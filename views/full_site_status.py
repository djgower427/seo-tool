"""Full Site Status — site-wide health check powered by Semrush."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

import seo_usage
from seo_apollo import ApolloError, organization_enrich
from seo_keys import get_apollo_key, get_semrush_key, normalize_domain
from seo_semrush import (
    SemrushError,
    domain_overview,
    domain_rank_history,
    top_pages,
)

DATABASES = ["us", "uk", "ca", "au", "de", "fr", "es", "it", "br", "in"]


def _to_int(value: str | None) -> int:
    try:
        return int(float(value)) if value else 0
    except (TypeError, ValueError):
        return 0


def _to_float(value: str | None) -> float:
    try:
        return float(value) if value else 0.0
    except (TypeError, ValueError):
        return 0.0


# Bump when seo_semrush.py's parser/columns change in a way that affects
# the shape of cached return values. The version flows into the cache key,
# so older cached entries become unreachable on the next deploy.
_API_CACHE_VERSION = 5


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _cached_overview(
    domain: str, database: str, api_key: str, version: int
) -> dict[str, str] | None:
    del version  # only here to participate in the cache key
    return domain_overview(domain, database, api_key)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _cached_history(
    domain: str, database: str, api_key: str, limit: int, version: int
) -> list[dict[str, str]]:
    del version
    return domain_rank_history(domain, database, api_key, limit=limit)


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def _cached_apollo_org(
    domain: str, api_key: str, version: int
) -> dict | None:
    del version
    return organization_enrich(domain, api_key)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _cached_top_pages(
    domain: str,
    database: str,
    api_key: str,
    limit: int,
    keyword_pool: int,
    version: int,
) -> list[dict[str, str]]:
    del version
    return top_pages(domain, database, api_key, limit=limit, keyword_pool=keyword_pool)


def render_overview(overview: dict[str, str]) -> None:
    cols = st.columns(4)
    cols[0].metric("Rank", f"{_to_int(overview.get('Rk')):,}" if overview.get("Rk") else "—")
    cols[1].metric("Organic keywords", f"{_to_int(overview.get('Or')):,}")
    cols[2].metric("Est. organic traffic", f"{_to_int(overview.get('Ot')):,}")
    cols[3].metric("Est. traffic value", f"${_to_int(overview.get('Oc')):,}")


def _missing_cols_warning(df: pd.DataFrame, expected: list[str], label: str) -> bool:
    """If any expected columns are absent, render an inline warning + the raw frame.

    Returns True when columns are missing (caller should bail).
    """
    missing = [c for c in expected if c not in df.columns]
    if not missing:
        return False
    st.warning(
        f"{label}: Semrush response is missing expected column(s) {missing}. "
        f"Columns returned: {list(df.columns)}"
    )
    with st.expander("Raw response"):
        st.dataframe(df, use_container_width=True, hide_index=True)
    return True


def render_history(rows: list[dict[str, str]]) -> None:
    if not rows:
        st.info("No historical data returned for this domain.")
        return

    df = pd.DataFrame(rows)
    if _missing_cols_warning(df, ["Dt", "Ot", "Or", "Oc"], "Traffic history"):
        return

    df["Date"] = pd.to_datetime(df["Dt"], format="%Y%m%d", errors="coerce")
    df["Organic traffic"] = df["Ot"].apply(_to_int)
    df["Organic keywords"] = df["Or"].apply(_to_int)
    df["Traffic value"] = df["Oc"].apply(_to_float)
    df = df.dropna(subset=["Date"]).sort_values("Date")

    tabs = st.tabs(["Traffic", "Keywords", "Traffic value"])
    with tabs[0]:
        st.line_chart(df.set_index("Date")["Organic traffic"], height=320)
    with tabs[1]:
        st.line_chart(df.set_index("Date")["Organic keywords"], height=320)
    with tabs[2]:
        st.caption("Estimated monthly value of this organic traffic, in USD.")
        st.line_chart(df.set_index("Date")["Traffic value"], height=320)


def _fmt_int(v: Any) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return "—"


def _fmt_money(v: Any) -> str:
    """Apollo returns funding amounts as raw numbers (USD)."""
    try:
        n = float(v)
    except (TypeError, ValueError):
        return "—"
    if n >= 1_000_000_000:
        return f"${n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"${n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"${n / 1_000:.0f}K"
    return f"${n:,.0f}"


def render_apollo(org: dict) -> None:
    """Render Apollo enrichment as a flat key/value table, plus a compact links
    row and a tech-stack pill list. Anything missing shows as '—' — Apollo's
    coverage varies a lot by company size and region.
    """
    location_parts = [org.get("city"), org.get("state"), org.get("country")]
    location = ", ".join(p for p in location_parts if p) or "—"
    revenue = org.get("annual_revenue_printed") or _fmt_money(org.get("annual_revenue"))
    total_raised = org.get("total_funding_printed") or _fmt_money(org.get("total_funding"))
    industry = (org.get("industry") or "").title() or "—"
    phone_obj = org.get("primary_phone")
    phone = (
        phone_obj.get("number") if isinstance(phone_obj, dict) else org.get("phone")
    ) or "—"

    rows = [
        ("Employees", _fmt_int(org.get("estimated_num_employees"))),
        ("Industry", industry),
        ("HQ", location),
        ("Est. revenue", revenue),
        ("Total raised", total_raised),
        ("Latest funding stage", org.get("latest_funding_stage") or "—"),
        ("Latest round date", org.get("latest_funding_round_date") or "—"),
        ("Phone", phone),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Field", "Value"]),
        use_container_width=True,
        hide_index=True,
    )

    # Compact links row (clickable markdown, only rendered URLs that exist).
    links = []
    for label, key in [
        ("Website", "website_url"),
        ("LinkedIn", "linkedin_url"),
        ("Twitter", "twitter_url"),
        ("Facebook", "facebook_url"),
    ]:
        url = org.get(key)
        if url:
            links.append(f"[{label}]({url})")
    if links:
        st.markdown("**Links** · " + " · ".join(links))

    # Tech stack — wrap-friendly pill list, only if Apollo returned any.
    techs = org.get("technology_names") or org.get("current_technologies") or []
    tech_names = []
    for t in techs:
        if isinstance(t, str):
            tech_names.append(t)
        elif isinstance(t, dict) and t.get("name"):
            tech_names.append(t["name"])
    if tech_names:
        st.markdown(f"**Tech stack** — {len(tech_names)} technologies detected")
        st.markdown(" ".join(f"`{name}`" for name in tech_names))


def render_top_pages(rows: list[dict]) -> None:
    if not rows:
        st.info("No top pages returned for this domain.")
        return
    df = pd.DataFrame(
        [
            {
                "URL": r.get("Ur", ""),
                "Keywords": int(r.get("Keywords", 0)),
                "Traffic share (%)": float(r.get("TrafficShare", 0.0)),
            }
            for r in rows
        ]
    )
    st.caption(
        "Pages ranked by the share of the domain's estimated organic traffic "
        "they capture across their ranking keywords."
    )
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "URL": st.column_config.LinkColumn("URL"),
            "Traffic share (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"top-pages-{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
    )


def render() -> None:
    st.title("🌐 Full Site Status")
    st.caption("Domain-wide SEO snapshot via Semrush, with optional Apollo company enrichment.")

    api_key = get_semrush_key()
    if not api_key:
        st.error(
            "`SEMRUSH_API_KEY` is not set. Add it to **Settings → Secrets** in Streamlit Cloud, "
            "or to `.streamlit/secrets.toml` locally."
        )
        st.stop()

    with st.form("site-status-form"):
        c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
        domain_input = c1.text_input("Domain", placeholder="example.com")
        database = c2.selectbox("Database", DATABASES, index=DATABASES.index("us"))
        history_months = c3.number_input("History (months)", 3, 60, 24)
        page_limit = c4.number_input("Top pages", 5, 100, 25)
        keyword_pool = c5.number_input(
            "Keyword pool",
            min_value=25,
            max_value=500,
            value=100,
            step=25,
            help=(
                "Keywords pulled from Semrush and aggregated by URL to compute "
                "top pages. Each row costs 10 Semrush credits (100 = 1,000 credits). "
                "Sorted by traffic share, so the top ~25 pages are usually captured "
                "in the first 100–150 keywords."
            ),
        )
        apollo_key = get_apollo_key()
        enrich_with_apollo = st.checkbox(
            "Enrich with Apollo (1 credit per company; size, industry, funding, tech stack)",
            value=False,
            disabled=not apollo_key,
            help=(
                None
                if apollo_key
                else "Set APOLLO_API_KEY in .streamlit/secrets.toml to enable."
            ),
        )
        submitted = st.form_submit_button("Run check", type="primary")

    if not submitted or not domain_input.strip():
        return

    domain = normalize_domain(domain_input)

    # Reserve a spot at the top of the results for the consumed/remaining line,
    # then measure real Semrush (and optional Apollo) balance deltas around all
    # the billable calls below. Cache hits make no API call, so they read as
    # "0 used".
    usage_slot = st.empty()
    tracker = seo_usage.start(
        semrush_key=api_key,
        apollo_key=apollo_key if (enrich_with_apollo and apollo_key) else None,
    )

    def _safe_call(label: str, fn, *args):
        try:
            return fn(*args)
        except SemrushError as e:
            st.error(f"{label} — {e}")
        return None

    with st.spinner("Querying Semrush…"):
        overview = _safe_call(
            "Overview", _cached_overview, domain, database, api_key, _API_CACHE_VERSION
        )
        history = _safe_call(
            "History", _cached_history, domain, database, api_key,
            int(history_months), _API_CACHE_VERSION,
        )
        pages = _safe_call(
            "Top pages", _cached_top_pages, domain, database, api_key,
            int(page_limit), int(keyword_pool), _API_CACHE_VERSION,
        )

    if overview:
        st.subheader(f"Overview — {domain} ({database})")
        render_overview(overview)
    elif overview is not None:  # call succeeded but returned no rows
        st.warning(f"Semrush has no data for `{domain}` in the `{database}` database.")

    if history is not None:
        st.subheader("Traffic & keywords over time")
        render_history(history)

    if pages is not None:
        st.subheader(f"Top {len(pages)} pages by organic traffic share")
        render_top_pages(pages)

    apollo_org: dict | None = None
    apollo_ok = False
    if enrich_with_apollo and apollo_key:
        st.subheader(f"Apollo enrichment — {domain}")
        with st.spinner("Querying Apollo…"):
            try:
                apollo_org = _cached_apollo_org(domain, apollo_key, _API_CACHE_VERSION)
                apollo_ok = True
            except ApolloError as e:
                st.error(f"Apollo — {e}")
        if apollo_org:
            render_apollo(apollo_org)
        elif apollo_ok:
            st.info(f"Apollo has no record for `{domain}`.")

    tracker.finish()
    tracker.render(usage_slot)


render()
