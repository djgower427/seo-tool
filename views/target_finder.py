"""Target Finder — build company target lists from criteria.

Company search via Apollo's /mixed_companies/search consumes credits per call.
The endpoint returns matches split across two arrays — `accounts` (companies
already in your Apollo workspace) and `organizations` (new matches from
Apollo's broader DB) — and we display both, tagged by source.

Apollo's search endpoint returns slimmer records than enrichment: no industry,
exact employee count, or technology stack. Those require a per-company
enrichment call (use Full Site Status for a specific domain).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

import seo_usage
from seo_apollo import ApolloError, organization_search
from seo_keys import get_apollo_key

_SEARCH_CACHE_VERSION = 3


def _parse_money(s: str) -> int | None:
    """Parse human-friendly money strings into raw USD ints.

    Accepts '1M', '100K', '1.5B', '$10M', '10,000,000'. Returns None for
    empty input. Raises ValueError on a non-empty input that doesn't parse —
    caller surfaces the error to the user.
    """
    raw = s.strip()
    if not raw:
        return None
    cleaned = raw.upper().replace("$", "").replace(",", "").replace(" ", "")
    multiplier = 1
    if cleaned.endswith("K"):
        multiplier = 1_000
        cleaned = cleaned[:-1]
    elif cleaned.endswith("M"):
        multiplier = 1_000_000
        cleaned = cleaned[:-1]
    elif cleaned.endswith("B"):
        multiplier = 1_000_000_000
        cleaned = cleaned[:-1]
    try:
        return int(float(cleaned) * multiplier)
    except ValueError:
        raise ValueError(
            f"couldn't parse {raw!r} as a dollar amount (try '1M', '100K', '1.5B')"
        ) from None


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _cached_company_search(
    keywords: str,
    employees_min: int,
    employees_max: int,
    revenue_min: int | None,
    revenue_max: int | None,
    locations_key: tuple[str, ...],
    per_page: int,
    api_key: str,
    version: int,
) -> dict[str, Any]:
    del version
    return organization_search(
        api_key,
        keywords=keywords or None,
        employees_min=employees_min,
        employees_max=employees_max,
        revenue_min=revenue_min,
        revenue_max=revenue_max,
        locations=list(locations_key) or None,
        per_page=per_page,
    )


def _fmt_money(v: Any) -> str:
    """Format a raw USD number into 1.2M / 1.2B style. Apollo also gives us
    `organization_revenue_printed` which is already formatted — prefer that
    when present."""
    try:
        n = float(v)
    except (TypeError, ValueError):
        return "—"
    if n <= 0:
        return "—"
    if n >= 1_000_000_000:
        return f"${n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"${n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"${n / 1_000:.0f}K"
    return f"${n:,.0f}"


def _company_location(c: dict[str, Any]) -> str:
    """Flat city/state/country are populated on `accounts` but missing on most
    `organizations`. Probe the `organization_*` prefixed variants too."""
    parts = [
        c.get("city") or c.get("organization_city"),
        c.get("state") or c.get("organization_state"),
        c.get("country") or c.get("organization_country"),
    ]
    return ", ".join(p for p in parts if p) or "—"


def _render_company_summary(company: dict[str, Any]) -> None:
    revenue_printed = company.get("organization_revenue_printed")
    revenue = f"${revenue_printed}" if revenue_printed else _fmt_money(company.get("organization_revenue"))

    founded = company.get("founded_year")
    founded_str = str(int(founded)) if founded else "—"

    location = _company_location(company)

    phone_obj = company.get("primary_phone")
    phone = (
        phone_obj.get("number") if isinstance(phone_obj, dict) else company.get("phone")
    ) or "—"

    num_contacts = company.get("num_contacts")
    contacts_str = f"{int(num_contacts):,}" if num_contacts is not None else "—"

    growth_12mo = company.get("organization_headcount_twelve_month_growth")
    growth_str = (
        f"{growth_12mo * 100:+.1f}%" if isinstance(growth_12mo, (int, float)) else "—"
    )

    rows = [
        ("Revenue", revenue),
        ("HQ", location),
        ("Founded", founded_str),
        ("Phone", phone),
        ("# Contacts in Apollo", contacts_str),
        ("12mo headcount growth", growth_str),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Field", "Value"]),
        use_container_width=True,
        hide_index=True,
    )

    links = []
    for label, key in [
        ("Website", "website_url"),
        ("LinkedIn", "linkedin_url"),
        ("Twitter", "twitter_url"),
    ]:
        url = company.get(key)
        if url:
            links.append(f"[{label}]({url})")
    if links:
        st.markdown(" · ".join(links))

    with st.expander("Show raw company data", expanded=False):
        st.json(company)


def _dedupe_by_domain(
    companies: list[dict[str, Any]],
    already_seen: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Apollo can return multiple records per company (e.g. three CRM rows
    that all point to the same Apollo organization). Collapse by
    primary_domain — keep the first record, drop the rest. `already_seen`
    threads accumulated domains across multiple arrays so a domain that
    appears in `accounts` is hidden from `organizations`.

    Returns (deduped_list, removed_count).
    """
    seen = set(already_seen or ())
    out: list[dict[str, Any]] = []
    removed = 0
    for c in companies:
        domain = (c.get("primary_domain") or c.get("domain") or "").strip().lower()
        if not domain:
            out.append(c)  # nothing to dedupe by — keep as-is
            continue
        if domain in seen:
            removed += 1
            continue
        seen.add(domain)
        out.append(c)
    return out, removed


def _company_header(company: dict[str, Any], source: str) -> str:
    name = company.get("name") or "(unnamed)"
    domain = company.get("primary_domain") or company.get("domain") or ""
    tag = "🆕 New lead" if source == "organization" else "📁 Existing account"
    return f"{tag} · **{name}** — {domain}"


def render() -> None:
    st.title("🎯 Target Finder")
    st.caption(
        "Find companies matching your criteria. Company search consumes "
        "Apollo credits per submission. Results are split into new leads "
        "(not yet in your Apollo workspace) and existing accounts. To enrich "
        "a target with industry / employees / tech stack, run it through "
        "**Full Site Status**."
    )

    apollo_key = get_apollo_key()
    if not apollo_key:
        st.error(
            "`APOLLO_API_KEY` is not set. Add it to Streamlit Cloud Secrets "
            "or to `.streamlit/secrets.toml` locally."
        )
        st.stop()

    with st.form("target-finder-form"):
        keywords_input = st.text_input(
            "Industry / keywords / technology",
            placeholder="SaaS HubSpot",
            help=(
                "Free-text fuzzy search across company name, description, "
                "industry, and detected technologies. Combine terms freely — "
                "e.g. \"SaaS HubSpot\" matches B2B SaaS companies that Apollo "
                "has tagged as HubSpot users."
            ),
        )
        c1, c2 = st.columns(2)
        emp_min = c1.number_input("Min employees", 1, 100000, 50)
        emp_max = c2.number_input("Max employees", 1, 100000, 200)
        c3, c4 = st.columns(2)
        rev_min_raw = c3.text_input(
            "Min revenue",
            placeholder="1M",
            help="Accepts shorthand: 100K, 1M, 1.5B. Leave blank for no minimum.",
        )
        rev_max_raw = c4.text_input(
            "Max revenue",
            placeholder="100M",
            help="Accepts shorthand: 100K, 1M, 1.5B. Leave blank for no maximum.",
        )
        location_input = st.text_input(
            "HQ location",
            placeholder="United States",
            help="One country, state, or city. Apollo does fuzzy matching.",
        )
        max_companies = st.selectbox(
            "Companies per search",
            [10, 25, 50, 100],
            index=1,
        )

        submitted = st.form_submit_button("Find companies", type="primary")

    if submitted:
        try:
            rev_min = _parse_money(rev_min_raw)
            rev_max = _parse_money(rev_max_raw)
        except ValueError as e:
            st.error(str(e))
            return
        st.session_state["_target_finder_query"] = {
            "keywords": keywords_input.strip(),
            "emp_min": int(emp_min),
            "emp_max": int(emp_max),
            "rev_min": rev_min,
            "rev_max": rev_max,
            "location": location_input.strip(),
            "max_companies": int(max_companies),
        }

    query = st.session_state.get("_target_finder_query")
    if not query:
        return

    # Measure the Apollo credit spend around the search (cache hits cost 0), and
    # surface it at the top of the results.
    usage_slot = st.empty()
    tracker = seo_usage.start(apollo_key=apollo_key)
    with st.spinner("Searching Apollo for companies…"):
        try:
            result = _cached_company_search(
                query["keywords"],
                query["emp_min"],
                query["emp_max"],
                query.get("rev_min"),
                query.get("rev_max"),
                (query["location"],) if query["location"] else (),
                query["max_companies"],
                apollo_key,
                _SEARCH_CACHE_VERSION,
            )
        except ApolloError as e:
            tracker.finish()
            tracker.render(usage_slot)
            st.error(f"Apollo — {e}")
            return
    tracker.finish()
    tracker.render(usage_slot)

    orgs_raw = result.get("organizations") or []
    accounts_raw = result.get("accounts") or []
    pagination = result.get("pagination") or {}
    total_matching = pagination.get(
        "total_entries", len(orgs_raw) + len(accounts_raw)
    )

    # Dedupe accounts first, then exclude their domains from organizations so
    # the same company never shows in both sections.
    accounts, acc_removed = _dedupe_by_domain(accounts_raw)
    account_domains = {
        (c.get("primary_domain") or c.get("domain") or "").strip().lower()
        for c in accounts
    }
    orgs, org_removed = _dedupe_by_domain(orgs_raw, already_seen=account_domains)
    total_removed = acc_removed + org_removed

    if not orgs and not accounts:
        st.info("No companies matched those filters. Try broadening the criteria.")
        return

    dedup_note = (
        f" ({total_removed} duplicate row{'s' if total_removed != 1 else ''} collapsed)"
        if total_removed
        else ""
    )
    st.caption(
        f"Showing {len(orgs) + len(accounts)} of {total_matching:,} matches: "
        f"**{len(orgs)} new leads**, **{len(accounts)} existing accounts**"
        f"{dedup_note}."
    )

    if orgs:
        st.subheader("🆕 New leads")
        for company in orgs:
            with st.expander(_company_header(company, "organization")):
                _render_company_summary(company)

    if accounts:
        st.subheader("📁 Existing accounts")
        st.caption(
            "Already in your Apollo workspace. Useful for verifying coverage "
            "or revisiting dormant relationships."
        )
        for company in accounts:
            with st.expander(_company_header(company, "account")):
                _render_company_summary(company)

    with st.expander("Debug: raw Apollo search response"):
        st.json(result)


render()
