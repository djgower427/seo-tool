"""Find Contacts — search Apollo for people at a given company domain.

The People Search endpoint is free; email reveals cost 1 credit per person.
We separate these into a free search step and an explicit per-row reveal,
and track revealed person IDs in session state so the same person is never
billed twice within a session.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from seo_apollo import (
    SENIORITIES,
    ApolloError,
    people_search,
    person_match,
    raw_usage_stats,
)
import seo_usage
from seo_claude import ClaudeError, expand_job_function
from seo_keys import get_anthropic_key, get_apollo_key, normalize_domain

# Bump if the cached query / search return shape changes.
_SEARCH_CACHE_VERSION = 5

# 100 is Apollo's hard cap for /mixed_people/api_search per_page — anything
# higher 422s with "Per page not supported". Confirmed empirically.
_APOLLO_FETCH_MAX = 100

# Single per-session cache of best-known person data: search preview → basic
# enrichment (full names) → email reveal. Each step merges over the prior one,
# so we never re-pay for data we already have within a session.
_ENRICH_CACHE_KEY = "_apollo_enriched_people"  # dict[person_id, dict]


def _enriched_store() -> dict[str, dict[str, Any]]:
    if _ENRICH_CACHE_KEY not in st.session_state:
        st.session_state[_ENRICH_CACHE_KEY] = {}
    return st.session_state[_ENRICH_CACHE_KEY]


def _email_revealed(person: dict[str, Any]) -> bool:
    """True iff this cached person dict carries an unlocked personal email."""
    email = (person or {}).get("email") or ""
    return bool(email) and "not_unlocked" not in email


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _cached_people_search(
    domain: str,
    titles_key: tuple[str, ...],
    seniorities_key: tuple[str, ...],
    api_key: str,
    version: int,
) -> dict[str, Any]:
    """Always fetch Apollo's max (100). The view paginates client-side. Tuples
    in the signature because Streamlit's cache key needs hashable args."""
    del version
    return people_search(
        domain,
        api_key,
        titles=list(titles_key) or None,
        seniorities=list(seniorities_key) or None,
        page=1,
        per_page=_APOLLO_FETCH_MAX,
    )


# Title lists for a given function don't change day-to-day; cache for a week
# to avoid re-calling Claude on the same input.
@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def _cached_expand_function(
    function: str,
    seniorities_key: tuple[str, ...],
    api_key: str,
    version: int,
) -> tuple[str, ...]:
    del version
    titles = expand_job_function(
        function, api_key, seniorities=list(seniorities_key) or None
    )
    return tuple(titles)


def _person_row(p: dict[str, Any], enriched: dict[str, dict]) -> dict[str, Any]:
    """Flatten one Apollo person record into a table row. Prefers the enriched
    copy (full surname, unlocked email if revealed) over the search preview."""
    pid = p.get("id") or ""
    src = enriched.get(pid) or p  # enriched data wins where present
    org = src.get("organization") or p.get("organization") or {}
    location_parts = [src.get("city"), src.get("state"), src.get("country")]
    location = ", ".join(x for x in location_parts if x) or "—"

    first = (src.get("first_name") or "").strip()
    last = (src.get("last_name") or "").strip()
    full_name = f"{first} {last}".strip() or (src.get("name") or "").strip() or "—"

    email = src.get("email") or ""
    email_revealed = _email_revealed(src)

    return {
        "ID": pid,
        "Name": full_name,
        "Title": src.get("title") or "—",
        "Seniority": src.get("seniority") or "—",
        "Company": org.get("name") or "—",
        "Location": location,
        "LinkedIn": src.get("linkedin_url") or p.get("linkedin_url") or "",
        "Email": email if email_revealed else ("🔒 hidden" if email else "—"),
        "_revealed": email_revealed,
    }


def _render_results_table(rows: list[dict[str, Any]]) -> Any:
    """Return the Streamlit selection event so the caller can react to picks."""
    df = pd.DataFrame(rows)
    # Hide internal columns from display but keep them in df for lookups.
    display_cols = ["Name", "Title", "Seniority", "Company", "Location", "Email", "LinkedIn"]
    return st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        column_config={
            "LinkedIn": st.column_config.LinkColumn("LinkedIn", display_text="profile"),
            "Email": st.column_config.TextColumn("Email", width="medium"),
        },
    )


def _do_reveals(
    selected_ids: list[str],
    api_key: str,
) -> tuple[int, int, list[str]]:
    """Reveal emails for selected rows. Skips IDs whose cached record already
    has an unlocked email (avoids double-billing). Returns
    (newly_revealed_count, skipped_already_revealed_count, errors)."""
    enriched = _enriched_store()
    to_reveal = [
        pid for pid in selected_ids
        if pid and not _email_revealed(enriched.get(pid) or {})
    ]
    skipped = len(selected_ids) - len(to_reveal)
    errors: list[str] = []

    if not to_reveal:
        return 0, skipped, errors

    progress = st.progress(0.0, text=f"Revealing 0 / {len(to_reveal)}…")
    for i, pid in enumerate(to_reveal, start=1):
        try:
            person = person_match(pid, api_key, reveal_personal_emails=True)
            if person:
                enriched[pid] = person
        except ApolloError as e:
            errors.append(f"{pid}: {e}")
        progress.progress(i / len(to_reveal), text=f"Revealing {i} / {len(to_reveal)}…")
    progress.empty()

    return len(to_reveal) - len(errors), skipped, errors


def _auto_enrich(
    people: list[dict[str, Any]],
    api_key: str,
) -> tuple[int, list[str]]:
    """Call /people/match (without reveal_personal_emails) for any person not
    already in the enrichment cache. Returns (enriched_count, errors)."""
    enriched = _enriched_store()
    todo = [
        (p.get("id") or "")
        for p in people
        if p.get("id") and p["id"] not in enriched
    ]
    todo = [pid for pid in todo if pid]
    if not todo:
        return 0, []

    errors: list[str] = []
    progress = st.progress(
        0.0,
        text=f"Enriching 0 / {len(todo)} contacts via Apollo…",
    )
    for i, pid in enumerate(todo, start=1):
        try:
            person = person_match(pid, api_key, reveal_personal_emails=False)
            if person:
                enriched[pid] = person
        except ApolloError as e:
            errors.append(f"{pid}: {e}")
        progress.progress(
            i / len(todo),
            text=f"Enriching {i} / {len(todo)} contacts via Apollo…",
        )
    progress.empty()
    return len(todo) - len(errors), errors


def render() -> None:
    st.title("👥 Find Contacts")
    st.caption(
        "Search Apollo for people at a company. Search is free; "
        "revealing personal emails costs 1 Apollo credit per person."
    )

    api_key = get_apollo_key()
    if not api_key:
        st.error(
            "`APOLLO_API_KEY` is not set. Add it to **Settings → Secrets** in "
            "Streamlit Cloud, or to `.streamlit/secrets.toml` locally."
        )
        st.stop()

    # TEMPORARY debug: dump the raw Apollo usage_stats payload so we can see what
    # a master key actually returns and wire the real "remaining" figure. Safe to
    # remove once that's done — the response carries rate-limit data, not secrets.
    with st.expander("🔧 Apollo API usage (debug — temporary)"):
        st.caption(
            "Click to fetch the raw /usage_stats/api_usage_stats response "
            "(requires a master API key). Paste it back to finalize Apollo usage "
            "reporting."
        )
        if st.button("Fetch Apollo usage stats"):
            try:
                st.json(raw_usage_stats(api_key))
            except ApolloError as e:
                st.error(str(e))

    # Persist the user's per-page preference across searches so changing it
    # after a search sticks.
    if "_per_page" not in st.session_state:
        st.session_state["_per_page"] = 10

    with st.form("find-contacts-form"):
        domain_input = st.text_input(
            "Company domain", placeholder="stripe.com",
            help="One domain at a time. Apollo finds people whose current employer matches.",
        )
        function_input = st.text_input(
            "Job function",
            placeholder="e.g. marketing leadership, demand generation, head of growth",
            help=(
                "Plain-language description of the role. Claude expands this "
                "into ~6–20 concrete job titles before searching Apollo. "
                "Leave empty to search every contact at the domain."
            ),
        )
        seniorities = st.multiselect(
            "Seniority",
            options=SENIORITIES,
            default=[],
            help="Apollo's defined seniority buckets. Leave empty for all.",
        )
        submitted = st.form_submit_button("Search (free)", type="primary")

    # On submit: expand the function to titles via Claude (if non-empty),
    # then persist params so the page survives reruns after reveals.
    if submitted and domain_input.strip():
        function = function_input.strip()
        titles: list[str] = []
        if function:
            anthropic_key = get_anthropic_key()
            if not anthropic_key:
                st.error(
                    "`ANTHROPIC_API_KEY` is not set. Add it to Streamlit Cloud "
                    "Secrets, or leave Job function empty to skip title expansion."
                )
                return
            expand_tracker = seo_usage.start()
            with st.spinner(f"Asking Claude for titles related to “{function}”…"):
                try:
                    titles = list(
                        _cached_expand_function(
                            function,
                            tuple(seniorities),
                            anthropic_key,
                            _SEARCH_CACHE_VERSION,
                        )
                    )
                except ClaudeError as e:
                    expand_tracker.finish()
                    st.error(f"Claude — {e}")
                    return
            expand_tracker.finish()
            # Rendered by render_pending() at the top of the results below.
            seo_usage.stash_pending(expand_tracker.summary_md())

        st.session_state["_find_contacts_query"] = {
            "domain": normalize_domain(domain_input),
            "function": function,
            "titles": titles,
            "seniorities": seniorities,
            "page": 1,
            "per_page": int(st.session_state["_per_page"]),
        }

    query = st.session_state.get("_find_contacts_query")
    if not query:
        return

    # Show usage from the action that triggered this run — the title expansion
    # (this run) or a reveal/enrich from the previous run (stashed before its
    # st.rerun()). People search itself is free, so no tracker around it.
    seo_usage.render_pending(st.empty())

    with st.spinner("Searching Apollo…"):
        try:
            result = _cached_people_search(
                query["domain"],
                tuple(query["titles"]),
                tuple(query["seniorities"]),
                api_key,
                _SEARCH_CACHE_VERSION,
            )
        except ApolloError as e:
            st.error(f"Apollo — {e}")
            return

    # Surface the Claude-generated titles so the user can see what was actually
    # searched and tune their input.
    if query.get("function") and query.get("titles"):
        label = (
            f"Searched {len(query['titles'])} title variants of "
            f"“{query['function']}” (click to expand)"
        )
        with st.expander(label):
            st.markdown(" · ".join(f"`{t}`" for t in query["titles"]))

    all_people = result.get("people") or []

    if not all_people:
        st.info(f"No contacts found at `{query['domain']}` for those filters.")
        return

    # Client-side pagination over the cached result list.
    total = len(all_people)
    per_page = max(1, int(query["per_page"]))
    total_pages = max(1, -(-total // per_page))  # ceil div
    current_page = min(max(1, int(query["page"])), total_pages)
    start = (current_page - 1) * per_page
    end = start + per_page
    people = all_people[start:end]

    cap_note = (
        " — Apollo caps the free search at 100; refine filters to narrow further"
        if total >= _APOLLO_FETCH_MAX
        else ""
    )
    st.caption(
        f"Showing {start + 1}–{start + len(people)} of {total:,} contacts at "
        f"`{query['domain']}` (page {current_page} of {total_pages}){cap_note}."
    )

    # Surnames stay hidden until the user explicitly pays for them — Apollo's
    # /people/match costs 1 credit per row even without an email reveal.
    enriched = _enriched_store()

    # Above-table nav row: per-page selector on the left, prev/next on the right.
    # Buttons stay rendered (and disabled) on single-page results so the layout
    # doesn't shift between searches.
    nav_pp, _, nav_prev, nav_next = st.columns([2, 4, 1, 1])
    nav_pp.selectbox(
        "Results per page",
        [10, 25, 50, 100],
        key="_per_page",
    )
    if st.session_state["_per_page"] != query["per_page"]:
        query["per_page"] = int(st.session_state["_per_page"])
        query["page"] = 1
        st.session_state["_find_contacts_query"] = query
        st.rerun()

    prev_clicked = nav_prev.button(
        "← Prev",
        disabled=current_page <= 1,
        use_container_width=True,
    )
    next_clicked = nav_next.button(
        "Next →",
        disabled=current_page >= total_pages,
        use_container_width=True,
    )
    if prev_clicked:
        query["page"] = current_page - 1
        st.session_state["_find_contacts_query"] = query
        st.rerun()
    if next_clicked:
        query["page"] = current_page + 1
        st.session_state["_find_contacts_query"] = query
        st.rerun()

    rows = [_person_row(p, enriched) for p in people]
    event = _render_results_table(rows)

    selected_indices = list(getattr(event.selection, "rows", []) or [])
    selected_ids = [rows[i]["ID"] for i in selected_indices]
    already_revealed = sum(
        1 for pid in selected_ids if _email_revealed(enriched.get(pid) or {})
    )
    to_pay = len(selected_ids) - already_revealed

    # Count how many rows on this page still need a name reveal (i.e. aren't
    # already cached from a prior reveal). 1 credit per row.
    unrevealed_page_ids = [
        (p.get("id") or "") for p in people
        if p.get("id") and p["id"] not in enriched
    ]
    names_credit_cost = len(unrevealed_page_ids)

    c1, c2, _ = st.columns([2, 2, 3])
    reveal_emails_btn = c1.button(
        f"Reveal emails for {len(selected_ids)} selected ({to_pay} credits)",
        type="primary",
        disabled=len(selected_ids) == 0,
        help=(
            f"{already_revealed} of the selected rows were already revealed earlier "
            "in this session and won't be re-billed."
        ) if already_revealed else None,
    )
    reveal_names_btn = c2.button(
        f"Reveal full names for this page ({names_credit_cost} credits)",
        disabled=names_credit_cost == 0,
        help=(
            "Unlocks the surname Apollo withholds from the free search. "
            "Rows you've already revealed (via email reveal or earlier name "
            "reveal) aren't re-charged."
        ),
    )

    if reveal_emails_btn:
        reveal_tracker = seo_usage.start(apollo_key=api_key)
        ok, skipped, errors = _do_reveals(selected_ids, api_key)
        reveal_tracker.finish()
        seo_usage.stash_pending(reveal_tracker.summary_md())
        msg = f"Revealed {ok} new contact(s)."
        if skipped:
            msg += f" {skipped} were already revealed this session (no charge)."
        st.success(msg)
        for err in errors:
            st.error(f"Reveal failed — {err}")
        st.rerun()  # re-render table with revealed emails merged in

    if reveal_names_btn:
        enrich_tracker = seo_usage.start(apollo_key=api_key)
        ok, errors = _auto_enrich(people, api_key)
        enrich_tracker.finish()
        seo_usage.stash_pending(enrich_tracker.summary_md())
        st.success(f"Revealed full names for {ok} contact(s).")
        for err in errors:
            st.error(f"Name reveal failed — {err}")
        st.rerun()

    with st.expander("Debug: raw Apollo response"):
        st.json(result)


render()
