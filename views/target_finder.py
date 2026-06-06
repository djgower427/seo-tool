"""Target Finder — build company target lists from criteria.

Company search via Apollo's /mixed_companies/search consumes credits per call.
This page is company-discovery only: to dig into contacts at a target, use
the Find Contacts page with the domain.
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from seo_apollo import ApolloError, organization_search
from seo_keys import get_apollo_key

_SEARCH_CACHE_VERSION = 1


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _cached_company_search(
    keywords: str,
    employees_min: int,
    employees_max: int,
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
        locations=list(locations_key) or None,
        per_page=per_page,
    )


def _first(d: dict, *keys: str) -> Any:
    """Return the first truthy value across the given keys. Apollo's response
    shape varies between endpoints (search vs enrich), so we probe several
    common names per logical field."""
    for k in keys:
        v = d.get(k)
        if v not in (None, "", [], {}):
            return v
    return None


def _company_employees(c: dict[str, Any]) -> str:
    v = _first(
        c,
        "estimated_num_employees",
        "num_employees",
        "employee_count",
        "employees",
    )
    try:
        return f"{int(v):,}" if v is not None else "—"
    except (TypeError, ValueError):
        return str(v) if v else "—"


def _company_industry(c: dict[str, Any]) -> str:
    v = _first(c, "industry", "primary_industry", "industry_tag")
    if isinstance(v, str) and v:
        return v.title()
    return "—"


def _company_location(c: dict[str, Any]) -> str:
    # Try flat city/state/country first, then a nested headquarters object,
    # then a raw_address fallback.
    parts = [c.get("city"), c.get("state"), c.get("country")]
    flat = ", ".join(p for p in parts if p)
    if flat:
        return flat

    hq = c.get("headquarters") or c.get("primary_headquarters_address") or {}
    if isinstance(hq, dict):
        parts = [hq.get("city"), hq.get("state"), hq.get("country")]
        nested = ", ".join(p for p in parts if p)
        if nested:
            return nested

    raw = c.get("raw_address") or c.get("address")
    return raw or "—"


def _company_technologies(c: dict[str, Any]) -> list[str]:
    techs = _first(c, "technology_names", "current_technologies", "technologies") or []
    out: list[str] = []
    for t in techs:
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, dict) and t.get("name"):
            out.append(t["name"])
    return out


def _build_tech_html(tech_names: list[str]) -> str:
    """Show the first 8 techs inline; the rest collapse behind a native HTML
    <details> "see N more" toggle. Inline expand, no Streamlit rerun."""
    if len(tech_names) <= 8:
        return ", ".join(html.escape(t) for t in tech_names)

    first = ", ".join(html.escape(t) for t in tech_names[:8])
    rest = ", ".join(html.escape(t) for t in tech_names[8:])
    return (
        f"{first}, "
        f'<details style="display:inline">'
        f'<summary style="display:inline;cursor:pointer;color:#16a34a">'
        f"see {len(tech_names) - 8} more</summary> "
        f"{rest}</details>"
    )


def _render_company_summary(company: dict[str, Any]) -> None:
    rows = [
        ("Employees", _company_employees(company)),
        ("Industry", _company_industry(company)),
        ("HQ", _company_location(company)),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Field", "Value"]),
        use_container_width=True,
        hide_index=True,
    )

    tech_names = _company_technologies(company)
    if tech_names:
        st.markdown(
            "**Tech:** " + _build_tech_html(tech_names),
            unsafe_allow_html=True,
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

    # Per-company raw data expander so we can fix field mappings if the
    # defensive reads above missed anything.
    with st.expander("Show raw company data", expanded=False):
        st.json(company)


def render() -> None:
    st.title("🎯 Target Finder")
    st.caption(
        "Find companies matching your criteria. Company search consumes "
        "Apollo credits per submission. To dig into contacts at a target, "
        "open it on the Find Contacts page."
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
                "e.g. \"SaaS HubSpot\" matches B2B SaaS companies that "
                "Apollo has tagged as HubSpot users."
            ),
        )
        c1, c2 = st.columns(2)
        emp_min = c1.number_input("Min employees", 1, 100000, 50)
        emp_max = c2.number_input("Max employees", 1, 100000, 200)
        location_input = st.text_input(
            "HQ location",
            placeholder="United States",
            help="One country, state, or city. Apollo does fuzzy matching.",
        )
        max_companies = st.selectbox(
            "Companies per search",
            [10, 25, 50, 100],
            index=1,
            help=(
                "Apollo may silently return fewer than requested depending on "
                "your plan tier."
            ),
        )

        submitted = st.form_submit_button("Find companies", type="primary")

    if submitted:
        st.session_state["_target_finder_query"] = {
            "keywords": keywords_input.strip(),
            "emp_min": int(emp_min),
            "emp_max": int(emp_max),
            "location": location_input.strip(),
            "max_companies": int(max_companies),
        }

    query = st.session_state.get("_target_finder_query")
    if not query:
        return

    with st.spinner("Searching Apollo for companies…"):
        try:
            result = _cached_company_search(
                query["keywords"],
                query["emp_min"],
                query["emp_max"],
                (query["location"],) if query["location"] else (),
                query["max_companies"],
                apollo_key,
                _SEARCH_CACHE_VERSION,
            )
        except ApolloError as e:
            st.error(f"Apollo — {e}")
            return

    companies = result.get("organizations") or []
    pagination = result.get("pagination") or {}
    total = pagination.get("total_entries", len(companies))

    if not companies:
        st.info("No companies matched those filters. Try broadening the criteria.")
        return

    capped_note = ""
    if len(companies) < query["max_companies"]:
        capped_note = (
            f" — Apollo returned fewer than the {query['max_companies']} requested. "
            "Likely a plan-tier cap; refine filters or upgrade plan to see more."
        )

    st.caption(
        f"Showing {len(companies)} of {total:,} companies matching your "
        f"filters{capped_note}."
    )

    for company in companies:
        name = company.get("name") or "(unnamed)"
        domain = company.get("primary_domain") or company.get("website_url") or ""
        with st.expander(f"**{name}** — {domain}"):
            _render_company_summary(company)

    with st.expander("Debug: raw Apollo search response"):
        st.json(result)


render()
