"""Full Site Status — site-wide health check powered by Semrush."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from seo_semrush import (
    SemrushError,
    domain_overview,
    domain_rank_history,
    raw_request,
    top_pages,
)

DATABASES = ["us", "uk", "ca", "au", "de", "fr", "es", "it", "br", "in"]


def get_semrush_key() -> str | None:
    try:
        return st.secrets.get("SEMRUSH_API_KEY")
    except (FileNotFoundError, KeyError, AttributeError):
        return None


def normalize_domain(raw: str) -> str:
    """Accept either a bare domain or a URL; return the bare host."""
    raw = raw.strip().lower()
    if "://" in raw:
        raw = urlparse(raw).netloc or raw
    return raw.lstrip("www.")


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
    if _missing_cols_warning(df, ["Dt", "Ot", "Or"], "Traffic history"):
        return

    df["Date"] = pd.to_datetime(df["Dt"], format="%Y%m%d", errors="coerce")
    df["Organic traffic"] = df["Ot"].apply(_to_int)
    df["Organic keywords"] = df["Or"].apply(_to_int)
    df = df.dropna(subset=["Date"]).sort_values("Date")

    tabs = st.tabs(["Traffic", "Keywords"])
    with tabs[0]:
        st.line_chart(df.set_index("Date")["Organic traffic"], height=320)
    with tabs[1]:
        st.line_chart(df.set_index("Date")["Organic keywords"], height=320)


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
    st.caption("Domain-wide SEO snapshot via Semrush. GA4, GSC, and extra Semrush reports coming next.")

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
        force_refresh = st.checkbox(
            "Force refresh (bypass 6-hour cache; costs Semrush credits)"
        )
        debug_mode = st.checkbox(
            "Debug: also show raw Semrush responses (uncached; extra credits)"
        )
        submitted = st.form_submit_button("Run check", type="primary")

    if not submitted or not domain_input.strip():
        return

    if force_refresh:
        st.cache_data.clear()

    domain = normalize_domain(domain_input)

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

    if debug_mode:
        st.divider()
        st.subheader("Debug: raw Semrush responses")
        for label, params in [
            (
                "domain_ranks",
                {
                    "type": "domain_ranks",
                    "key": api_key,
                    "domain": domain,
                    "database": database,
                    "export_columns": "Dn,Rk,Or,Ot,Oc,Ad,At,Ac",
                },
            ),
            (
                "domain_rank_history",
                {
                    "type": "domain_rank_history",
                    "key": api_key,
                    "domain": domain,
                    "database": database,
                    "export_columns": "Rk,Or,Ot,Oc,Dt",
                    "display_limit": str(int(history_months)),
                },
            ),
            (
                "domain_organic (first 10)",
                {
                    "type": "domain_organic",
                    "key": api_key,
                    "domain": domain,
                    "database": database,
                    "export_columns": "Ph,Po,Nq,Ur,Tr",
                    "display_limit": "10",
                    "display_sort": "tr_desc",
                },
            ),
        ]:
            with st.expander(label):
                try:
                    body = raw_request(params)
                except SemrushError as e:
                    st.error(str(e))
                    continue
                if not body.strip():
                    st.info("(empty response body)")
                else:
                    st.code(body, language="text")


render()
