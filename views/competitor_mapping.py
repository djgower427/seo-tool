"""Competitor / Market Mapping — find SEO openings against a competitor.

For a seed (competitor) domain, Semrush pulls the seed's organic keywords and
our own (sketchdev.io by default), and we surface keywords the competitor ranks
for that we don't — filtered to low difficulty and decent volume, i.e. the ones
easiest to take. Semrush charges ~10 credits per keyword row, for both domains,
so the keyword-pool size drives the cost.

(An earlier version also profiled the seed via Apollo + Claude to find similar
companies; that section was removed pending a more useful design. The library
helpers it used — seo_claude.build_similar_company_query and
seo_apollo.organization_search — remain available.)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
import streamlit as st

from seo_claude import ClaudeError, recommend_keyword_to_steal, summarize_offerings
from seo_keys import get_anthropic_key, get_semrush_key, normalize_domain
from seo_review import extract_page_data, fetch_page
from seo_semrush import SemrushError, domain_organic_keywords

DATABASES = ["us", "uk", "ca", "au", "de", "fr", "es", "it", "br", "in"]

# Our own domain — the "us" side of the keyword-gap comparison, and the domain
# whose core offerings the recommendation is kept relevant to. Editable on the
# page; this default is inferred from the team's email domain.
DEFAULT_OUR_DOMAIN = "sketchdev.io"

# How many candidates (easiest-first) to hand the recommender.
_RECOMMEND_TOP_N = 40

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


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _cached_organic_keywords(
    domain: str, database: str, api_key: str, limit: int, version: int
) -> list[dict[str, str]]:
    del version
    return domain_organic_keywords(domain, database, api_key, limit=limit)


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def _cached_offerings(domain: str, api_key: str, version: int) -> str:
    """Fetch a domain's homepage and summarize its core offerings via Claude.

    Raises httpx.HTTPError (fetch failure) or ClaudeError (summary failure) —
    the caller decides how to surface those.
    """
    del version
    final_url, html = fetch_page(f"https://{domain}")
    page = extract_page_data(final_url, html)
    head = (
        f"Title: {page.get('title') or ''}\n"
        f"Meta description: {page.get('meta_description') or ''}\n\n"
    )
    return summarize_offerings(domain, head + (page.get("body_text") or ""), api_key)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _cached_recommendation(
    candidates_json: str,
    offerings: str,
    competitor: str,
    ours: str,
    api_key: str,
    version: int,
) -> str:
    """candidates_json is a JSON string so the cache key stays hashable."""
    del version
    return recommend_keyword_to_steal(
        json.loads(candidates_json),
        offerings,
        competitor_domain=competitor,
        our_domain=ours,
        api_key=api_key,
    )


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


def _offerings_field(domain: str, anthropic_key: str) -> str:
    """Editable 'Our core offerings' field, auto-filled from `domain`'s homepage.

    The summary is derived once per domain (cached) and stored in session state
    under a domain-scoped key, so the user can edit it without it being
    overwritten on rerun, and switching domains re-derives. Returns the current
    field text.
    """
    state_key = f"_cm_offerings::{domain}"
    if state_key not in st.session_state:
        with st.spinner(f"Reading {domain} to summarize our offerings…"):
            try:
                st.session_state[state_key] = _cached_offerings(
                    domain, anthropic_key, _CACHE_VERSION
                )
            except (httpx.HTTPError, ClaudeError) as e:
                st.warning(
                    f"Couldn't auto-summarize `{domain}` ({e}). Describe our core "
                    "offerings below instead."
                )
                st.session_state[state_key] = ""
    return st.text_area(
        "Our core offerings",
        key=state_key,
        height=140,
        help=(
            f"Auto-summarized from {domain}'s homepage. Only keywords relevant "
            "to these offerings are recommended — edit to refine."
        ),
    )


def render() -> None:
    st.title("🗺️ Competitor / Market Mapping")
    st.caption(
        "For a competitor domain, find the keywords Sketch could most easily "
        "steal — ones the competitor ranks for that we don't, filtered to low "
        "difficulty and decent volume — then get a Claude recommendation of "
        "which one to target (kept to our core offerings, auto-summarized from "
        "our domain) and what blog to write. Semrush charges ~10 credits per "
        "keyword row, per domain."
    )

    semrush_key = get_semrush_key()
    if not semrush_key:
        st.error(
            "`SEMRUSH_API_KEY` is not set. Add it to Streamlit Cloud Secrets or "
            "to `.streamlit/secrets.toml` locally."
        )
        st.stop()
    anthropic_key = get_anthropic_key()

    with st.form("competitor-mapping-form"):
        c1, c2 = st.columns(2)
        seed_input = c1.text_input(
            "Competitor domain",
            placeholder="stripe.com",
            help="The competitor whose keywords you want to mine.",
        )
        our_input = c2.text_input(
            "Our domain",
            value=DEFAULT_OUR_DOMAIN,
            help=(
                "Our domain — the 'us' side of the keyword-gap comparison. Its "
                "homepage is also used to auto-fill 'Our core offerings' below, "
                "which keeps the recommendation on-topic."
            ),
        )

        c3, c4 = st.columns(2)
        database = c3.selectbox("Semrush database", DATABASES, index=DATABASES.index("us"))
        keyword_pool = c4.number_input(
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

        submitted = st.form_submit_button("Find keywords", type="primary")

    if submitted:
        if not seed_input.strip():
            st.error("Enter a competitor domain.")
            return
        if not our_input.strip():
            st.error("Enter our domain to compare against.")
            return
        st.session_state["_cm_query"] = {
            "seed": normalize_domain(seed_input),
            "ours": normalize_domain(our_input),
            "database": database,
            "keyword_pool": int(keyword_pool),
            "max_difficulty": float(max_difficulty),
            "max_position": int(max_position),
            "min_volume": int(min_volume),
        }

    q = st.session_state.get("_cm_query")
    if not q:
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
        return

    # ── Recommendation ──────────────────────────────────────────────────────
    st.subheader("⭐ Recommendation")
    if not anthropic_key:
        st.info(
            "Set `ANTHROPIC_API_KEY` to auto-summarize your offerings and get a "
            "written recommendation of which keyword to target."
        )
    else:
        offerings = _offerings_field(q["ours"], anthropic_key)
        if not offerings.strip():
            st.info(
                "Add a description of our core offerings above to get a focused "
                "recommendation."
            )
        else:
            with st.spinner("Asking Claude which keyword to target…"):
                try:
                    rec = _cached_recommendation(
                        json.dumps(candidates[:_RECOMMEND_TOP_N], sort_keys=True),
                        offerings,
                        q["seed"],
                        q["ours"],
                        anthropic_key,
                        _CACHE_VERSION,
                    )
                except ClaudeError as e:
                    st.error(f"Claude — {e}")
                    rec = None
            if rec:
                st.markdown(rec)

    st.subheader("Steal-able keywords")
    _render_steal_table(candidates)


render()
