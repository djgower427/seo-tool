"""Competitor / Market Mapping — map a seed company's market and find SEO openings.

Two halves, both keyed off one seed-company domain:

1. **Similar companies.** Apollo enriches the seed (industry, size, revenue,
   keywords), Claude turns that profile into an Apollo company-search query, and
   we run it to surface peers by industry + size. Apollo enrich and the company
   search each consume 1 Apollo credit.

2. **Steal-able keywords.** Semrush pulls the seed's organic keywords and our own
   (sketchdev.io by default), and we surface keywords the competitor ranks for
   that we don't — filtered to low difficulty and decent volume, i.e. the ones
   easiest to take. Semrush charges ~10 credits per keyword row, for both
   domains, so the keyword-pool sizes drive the cost.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from seo_apollo import ApolloError, organization_enrich, organization_search
from seo_claude import ClaudeError, build_similar_company_query
from seo_keys import (
    get_anthropic_key,
    get_apollo_key,
    get_semrush_key,
    normalize_domain,
)
from seo_semrush import SemrushError, domain_organic_keywords

DATABASES = ["us", "uk", "ca", "au", "de", "fr", "es", "it", "br", "in"]

# Sketch's own domain — the "us" side of the keyword-gap comparison. Editable on
# the page; this is just the default (inferred from the team's email domain).
DEFAULT_OUR_DOMAIN = "sketchdev.io"

# Bump when a cached call's args or return shape changes so old entries expire.
_CACHE_VERSION = 1


def _to_int(value: Any) -> int:
    try:
        return int(float(value)) if value not in (None, "") else 0
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _fmt_money(v: Any) -> str:
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


# ── Cached API calls ────────────────────────────────────────────────────────


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def _cached_enrich(domain: str, api_key: str, version: int) -> dict | None:
    del version
    return organization_enrich(domain, api_key)


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def _cached_build_query(
    profile_key: str, api_key: str, version: int
) -> dict[str, Any]:
    """profile_key is a JSON string so the cache key stays hashable + stable."""
    del version
    return build_similar_company_query(json.loads(profile_key), api_key)


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


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _cached_organic_keywords(
    domain: str, database: str, api_key: str, limit: int, version: int
) -> list[dict[str, str]]:
    del version
    return domain_organic_keywords(domain, database, api_key, limit=limit)


# ── Profile distillation + steal logic ──────────────────────────────────────


def _distill_profile(org: dict[str, Any]) -> dict[str, Any]:
    """Pull the fields Claude needs from a full Apollo enrichment record."""
    keywords = org.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []
    return {
        "name": org.get("name"),
        "industry": org.get("industry"),
        "keywords": [k for k in keywords if isinstance(k, str)][:25],
        "description": (org.get("short_description") or "")[:600] or None,
        "employees": org.get("estimated_num_employees"),
        "revenue": org.get("annual_revenue"),
    }


def _domain_of(company: dict[str, Any]) -> str:
    return (company.get("primary_domain") or company.get("domain") or "").strip().lower()


def _dedupe_companies(
    companies: list[dict[str, Any]], exclude: set[str]
) -> list[dict[str, Any]]:
    """Collapse duplicate rows by domain and drop any domain in `exclude`
    (used to hide the seed company from its own peer list)."""
    seen = set(exclude)
    out: list[dict[str, Any]] = []
    for c in companies:
        d = _domain_of(c)
        if d and d in seen:
            continue
        if d:
            seen.add(d)
        out.append(c)
    return out


def _steal_candidates(
    competitor_kws: list[dict[str, str]],
    our_kws: list[dict[str, str]],
    *,
    max_position: int,
    max_difficulty: float,
    min_volume: int,
) -> list[dict[str, Any]]:
    """Keywords the competitor ranks for that Sketch could plausibly steal.

    A candidate is kept when: the competitor ranks within `max_position`, the
    keyword clears `min_volume` and is at/under `max_difficulty`, and Sketch
    either doesn't rank for it (within the pulled set) or ranks strictly worse.
    Sorted easiest-first: lowest difficulty, then highest volume.
    """
    our_pos: dict[str, int] = {}
    for r in our_kws:
        ph = (r.get("Ph") or "").strip().lower()
        if ph:
            our_pos[ph] = _to_int(r.get("Po"))

    out: list[dict[str, Any]] = []
    for r in competitor_kws:
        ph = (r.get("Ph") or "").strip()
        if not ph:
            continue
        comp_pos = _to_int(r.get("Po"))
        if comp_pos == 0 or comp_pos > max_position:
            continue
        vol = _to_int(r.get("Nq"))
        if vol < min_volume:
            continue
        kd = _to_float(r.get("Kd"))
        if kd > max_difficulty:
            continue
        our_p = our_pos.get(ph.lower())
        if our_p is not None and our_p <= comp_pos:
            continue  # we already rank as well or better — not a steal
        out.append(
            {
                "Keyword": ph,
                "Competitor position": comp_pos,
                "Our position": our_p if our_p else None,
                "Volume": vol,
                "Difficulty": kd,
                "CPC": _to_float(r.get("Cp")),
                "URL": (r.get("Ur") or "").strip(),
            }
        )

    out.sort(key=lambda x: (x["Difficulty"], -x["Volume"]))
    return out


# ── Rendering ───────────────────────────────────────────────────────────────


def _render_query(query: dict[str, Any]) -> None:
    rev_min = query.get("revenue_min")
    rev_max = query.get("revenue_max")
    rev = (
        f"{_fmt_money(rev_min)} – {_fmt_money(rev_max)}"
        if rev_min or rev_max
        else "any"
    )
    locs = ", ".join(query.get("locations") or []) or "any"
    rows = [
        ("Keywords", query.get("keywords") or "—"),
        ("Employees", f"{query['employees_min']:,} – {query['employees_max']:,}"),
        ("Revenue", rev),
        ("Location", locs),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Field", "Value"]),
        use_container_width=True,
        hide_index=True,
    )
    if query.get("rationale"):
        st.caption(f"_{query['rationale']}_")


def _render_company(company: dict[str, Any]) -> None:
    name = company.get("name") or "(unnamed)"
    domain = _domain_of(company)
    st.markdown(f"**{name}** — {domain}" if domain else f"**{name}**")

    revenue_printed = company.get("organization_revenue_printed")
    revenue = (
        f"${revenue_printed}"
        if revenue_printed
        else _fmt_money(company.get("organization_revenue"))
    )
    loc_parts = [
        company.get("city") or company.get("organization_city"),
        company.get("state") or company.get("organization_state"),
        company.get("country") or company.get("organization_country"),
    ]
    location = ", ".join(p for p in loc_parts if p) or "—"
    employees = company.get("estimated_num_employees")
    emp_str = f"{int(employees):,}" if employees else "—"

    rows = [
        ("Employees", emp_str),
        ("Revenue", revenue),
        ("HQ", location),
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
    ]:
        url = company.get(key)
        if url:
            links.append(f"[{label}]({url})")
    if links:
        st.markdown(" · ".join(links))


def _render_steal_table(rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "URL": st.column_config.LinkColumn("Competitor URL"),
            "Difficulty": st.column_config.NumberColumn(
                "Difficulty", format="%.0f", help="Semrush keyword difficulty, 0–100 (lower = easier)"
            ),
            "Volume": st.column_config.NumberColumn("Volume", format="%d"),
            "CPC": st.column_config.NumberColumn("CPC", format="$%.2f"),
            "Our position": st.column_config.NumberColumn(
                "Our position", help="Blank = we don't rank for it in the pulled set"
            ),
        },
    )
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"steal-able-keywords-{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
    )


# ── Page ────────────────────────────────────────────────────────────────────


def render() -> None:
    st.title("🗺️ Competitor / Market Mapping")
    st.caption(
        "For a seed company, find similar companies (Apollo, structured by "
        "Claude) and the keywords Sketch could most easily steal from it "
        "(Semrush). Apollo enrich + company search cost 1 credit each; Semrush "
        "charges ~10 credits per keyword row, per domain."
    )

    apollo_key = get_apollo_key()
    anthropic_key = get_anthropic_key()
    semrush_key = get_semrush_key()

    missing = [
        name
        for name, key in [
            ("APOLLO_API_KEY", apollo_key),
            ("ANTHROPIC_API_KEY", anthropic_key),
            ("SEMRUSH_API_KEY", semrush_key),
        ]
        if not key
    ]
    if missing:
        st.error(
            "Missing API key(s): "
            + ", ".join(f"`{m}`" for m in missing)
            + ". Add them to Streamlit Cloud Secrets or `.streamlit/secrets.toml`."
        )
        st.stop()

    with st.form("competitor-mapping-form"):
        c1, c2 = st.columns(2)
        seed_input = c1.text_input(
            "Seed company domain",
            placeholder="stripe.com",
            help="The company whose market you want to map.",
        )
        our_input = c2.text_input(
            "Our domain",
            value=DEFAULT_OUR_DOMAIN,
            help="Sketch's domain — the 'us' side of the keyword-gap comparison.",
        )

        c3, c4, c5 = st.columns(3)
        database = c3.selectbox("Semrush database", DATABASES, index=DATABASES.index("us"))
        max_companies = c4.selectbox("Similar companies", [10, 25, 50, 100], index=1)
        keyword_pool = c5.number_input(
            "Keyword pool per domain",
            min_value=25,
            max_value=500,
            value=100,
            step=25,
            help=(
                "Organic keywords pulled from Semrush for each domain (~10 "
                "credits per row). Larger pools find more openings but cost more."
            ),
        )

        st.markdown("**Steal-able keyword filters**")
        f1, f2, f3 = st.columns(3)
        max_difficulty = f1.slider(
            "Max difficulty",
            min_value=0,
            max_value=100,
            value=40,
            help="Semrush keyword difficulty ceiling. Lower = easier to rank.",
        )
        max_position = f2.number_input(
            "Competitor ranks within top",
            min_value=1,
            max_value=100,
            value=20,
            help="Only consider keywords where the competitor ranks this high.",
        )
        min_volume = f3.number_input(
            "Min search volume", min_value=0, max_value=100000, value=50, step=10
        )

        submitted = st.form_submit_button("Map the market", type="primary")

    if submitted:
        if not seed_input.strip():
            st.error("Enter a seed company domain.")
            return
        st.session_state["_cm_query"] = {
            "seed": normalize_domain(seed_input),
            "ours": normalize_domain(our_input) if our_input.strip() else "",
            "database": database,
            "max_companies": int(max_companies),
            "keyword_pool": int(keyword_pool),
            "max_difficulty": float(max_difficulty),
            "max_position": int(max_position),
            "min_volume": int(min_volume),
        }

    q = st.session_state.get("_cm_query")
    if not q:
        return

    # ── 1. Enrich seed + build search query + find similar companies ────────
    st.header("Similar companies")

    with st.spinner(f"Enriching `{q['seed']}` via Apollo…"):
        try:
            seed_org = _cached_enrich(q["seed"], apollo_key, _CACHE_VERSION)
        except ApolloError as e:
            st.error(f"Apollo — {e}")
            seed_org = None

    if not seed_org:
        st.warning(
            f"Apollo has no record for `{q['seed']}`, so it can't profile the "
            "seed. Skipping the similar-companies search."
        )
    else:
        profile = _distill_profile(seed_org)
        with st.expander(f"Seed profile — {profile.get('name') or q['seed']}"):
            st.json(profile)

        with st.spinner("Asking Claude to structure an Apollo query…"):
            try:
                search_query = _cached_build_query(
                    json.dumps(profile, sort_keys=True), anthropic_key, _CACHE_VERSION
                )
            except ClaudeError as e:
                st.error(f"Claude — {e}")
                search_query = None

        if search_query:
            st.subheader("Claude's search criteria")
            _render_query(search_query)

            with st.spinner("Searching Apollo for similar companies…"):
                try:
                    result = _cached_company_search(
                        search_query["keywords"],
                        search_query["employees_min"],
                        search_query["employees_max"],
                        search_query["revenue_min"],
                        search_query["revenue_max"],
                        tuple(search_query["locations"]),
                        q["max_companies"],
                        apollo_key,
                        _CACHE_VERSION,
                    )
                except ApolloError as e:
                    st.error(f"Apollo — {e}")
                    result = None

            if result is not None:
                raw = (result.get("organizations") or []) + (
                    result.get("accounts") or []
                )
                companies = _dedupe_companies(raw, exclude={q["seed"]})
                pagination = result.get("pagination") or {}
                total = pagination.get("total_entries", len(companies))

                if not companies:
                    st.info(
                        "No similar companies matched. Try a larger seed company "
                        "or broaden the search."
                    )
                else:
                    st.caption(
                        f"Showing {len(companies)} of ~{total:,} matching companies."
                    )
                    for company in companies:
                        header = (
                            f"{company.get('name') or '(unnamed)'} — "
                            f"{_domain_of(company) or 'no domain'}"
                        )
                        with st.expander(header):
                            _render_company(company)

    # ── 2. Steal-able keywords via Semrush ──────────────────────────────────
    st.header("Keywords Sketch could steal")

    if not q["ours"]:
        st.info(
            "Set **Our domain** above to compare against the competitor and find "
            "steal-able keywords."
        )
        return

    st.caption(
        f"Comparing **{q['seed']}** (competitor) against **{q['ours']}** (us) in "
        f"the `{q['database']}` database."
    )

    with st.spinner("Pulling organic keywords from Semrush…"):
        try:
            competitor_kws = _cached_organic_keywords(
                q["seed"], q["database"], semrush_key, q["keyword_pool"], _CACHE_VERSION
            )
        except SemrushError as e:
            st.error(f"Semrush (competitor) — {e}")
            return
        try:
            our_kws = _cached_organic_keywords(
                q["ours"], q["database"], semrush_key, q["keyword_pool"], _CACHE_VERSION
            )
        except SemrushError as e:
            st.error(f"Semrush (us) — {e}")
            return

    if not competitor_kws:
        st.warning(
            f"Semrush returned no organic keywords for `{q['seed']}` in the "
            f"`{q['database']}` database. Try a different database."
        )
        return

    candidates = _steal_candidates(
        competitor_kws,
        our_kws,
        max_position=q["max_position"],
        max_difficulty=q["max_difficulty"],
        min_volume=q["min_volume"],
    )

    st.caption(
        f"Pulled {len(competitor_kws):,} competitor keywords and "
        f"{len(our_kws):,} of ours. **{len(candidates)} steal-able** after "
        f"filtering (difficulty ≤ {int(q['max_difficulty'])}, competitor in top "
        f"{q['max_position']}, volume ≥ {q['min_volume']})."
    )

    if not candidates:
        st.info(
            "No steal-able keywords under these filters. Loosen the difficulty "
            "or volume thresholds, or widen the competitor position cap."
        )
    else:
        _render_steal_table(candidates)


render()
