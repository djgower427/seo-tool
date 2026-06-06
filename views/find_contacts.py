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
    person_reveal,
)
from seo_keys import get_apollo_key, normalize_domain

# Bump if the cached search return shape changes.
_SEARCH_CACHE_VERSION = 1

# Keys into st.session_state
_REVEAL_CACHE_KEY = "_apollo_revealed_people"  # dict[person_id, dict]


def _revealed_store() -> dict[str, dict[str, Any]]:
    """Per-session map of person_id → revealed person dict. Stops double-billing
    if the user clicks reveal twice on the same row within a session."""
    if _REVEAL_CACHE_KEY not in st.session_state:
        st.session_state[_REVEAL_CACHE_KEY] = {}
    return st.session_state[_REVEAL_CACHE_KEY]


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _cached_people_search(
    domain: str,
    titles_key: tuple[str, ...],
    seniorities_key: tuple[str, ...],
    page: int,
    per_page: int,
    api_key: str,
    version: int,
) -> dict[str, Any]:
    """Search is free, so cache aggressively. Tuples in the signature because
    Streamlit's cache key needs hashable args."""
    del version
    return people_search(
        domain,
        api_key,
        titles=list(titles_key) or None,
        seniorities=list(seniorities_key) or None,
        page=page,
        per_page=per_page,
    )


def _parse_titles_input(raw: str) -> list[str]:
    """Comma- or newline-separated titles, stripped of whitespace and blanks."""
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    return [p for p in parts if p]


def _person_row(p: dict[str, Any], revealed: dict[str, dict]) -> dict[str, Any]:
    """Flatten one Apollo person preview into a table row. Merges in any
    revealed email/phone we already paid for this session."""
    pid = p.get("id") or ""
    org = p.get("organization") or {}
    location_parts = [p.get("city"), p.get("state"), p.get("country")]
    location = ", ".join(x for x in location_parts if x) or "—"

    # Prefer revealed data (with real email) if we have it; otherwise show the
    # masked email Apollo returns for unrevealed people.
    revealed_p = revealed.get(pid, {})
    email = revealed_p.get("email") or p.get("email") or ""
    # Apollo masks unrevealed emails as "email_not_unlocked@domain.com".
    email_revealed = bool(revealed_p) and "not_unlocked" not in (email or "")

    return {
        "ID": pid,
        "Name": p.get("name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or "—",
        "Title": p.get("title") or "—",
        "Seniority": p.get("seniority") or "—",
        "Company": org.get("name") or "—",
        "Location": location,
        "LinkedIn": p.get("linkedin_url") or "",
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
    """Call person_reveal for each unrevealed selected ID. Returns
    (revealed_count, skipped_already_revealed_count, errors)."""
    revealed = _revealed_store()
    to_reveal = [pid for pid in selected_ids if pid and pid not in revealed]
    skipped = len(selected_ids) - len(to_reveal)
    errors: list[str] = []

    if not to_reveal:
        return 0, skipped, errors

    progress = st.progress(0.0, text=f"Revealing 0 / {len(to_reveal)}…")
    for i, pid in enumerate(to_reveal, start=1):
        try:
            person = person_reveal(pid, api_key)
            if person:
                revealed[pid] = person
        except ApolloError as e:
            errors.append(f"{pid}: {e}")
        progress.progress(i / len(to_reveal), text=f"Revealing {i} / {len(to_reveal)}…")
    progress.empty()

    return len(to_reveal) - len(errors), skipped, errors


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

    with st.form("find-contacts-form"):
        c1, c2 = st.columns([2, 1])
        domain_input = c1.text_input(
            "Company domain", placeholder="stripe.com",
            help="One domain at a time. Apollo finds people whose current employer matches.",
        )
        seniorities = c2.multiselect(
            "Seniority",
            options=SENIORITIES,
            default=[],
            help="Apollo's defined seniority buckets. Leave empty for all.",
        )
        titles_raw = st.text_input(
            "Job titles (comma-separated)",
            placeholder="head of marketing, vp of growth, cmo",
            help="Apollo does fuzzy matching, so close variants are fine.",
        )
        c3, c4 = st.columns(2)
        per_page = c3.number_input("Results per page", 10, 100, 25, step=5)
        page = c4.number_input("Page", 1, 500, 1)
        submitted = st.form_submit_button("Search (free)", type="primary")

    # Persist last search params so the page survives reruns after reveals.
    if submitted and domain_input.strip():
        st.session_state["_find_contacts_query"] = {
            "domain": normalize_domain(domain_input),
            "titles": _parse_titles_input(titles_raw),
            "seniorities": seniorities,
            "page": int(page),
            "per_page": int(per_page),
        }

    query = st.session_state.get("_find_contacts_query")
    if not query:
        return

    with st.spinner("Searching Apollo…"):
        try:
            result = _cached_people_search(
                query["domain"],
                tuple(query["titles"]),
                tuple(query["seniorities"]),
                query["page"],
                query["per_page"],
                api_key,
                _SEARCH_CACHE_VERSION,
            )
        except ApolloError as e:
            st.error(f"Apollo — {e}")
            return

    people = result.get("people") or []
    pagination = result.get("pagination") or {}

    if not people:
        st.info(f"No contacts found at `{query['domain']}` for those filters.")
        return

    total = pagination.get("total_entries", len(people))
    total_pages = pagination.get("total_pages", 1)
    st.caption(
        f"Showing {len(people)} of {total:,} contacts at `{query['domain']}` "
        f"(page {query['page']} of {total_pages})."
    )

    revealed = _revealed_store()
    rows = [_person_row(p, revealed) for p in people]
    event = _render_results_table(rows)

    selected_indices = list(getattr(event.selection, "rows", []) or [])
    selected_ids = [rows[i]["ID"] for i in selected_indices]
    already_revealed = sum(1 for pid in selected_ids if pid in revealed)
    to_pay = len(selected_ids) - already_revealed

    c1, c2 = st.columns([1, 3])
    reveal_btn = c1.button(
        f"Reveal emails for {len(selected_ids)} selected ({to_pay} credits)",
        type="primary",
        disabled=len(selected_ids) == 0,
        help=(
            f"{already_revealed} of the selected rows were already revealed earlier "
            "in this session and won't be re-billed."
        ) if already_revealed else None,
    )

    if reveal_btn:
        ok, skipped, errors = _do_reveals(selected_ids, api_key)
        msg = f"Revealed {ok} new contact(s)."
        if skipped:
            msg += f" {skipped} were already revealed this session (no charge)."
        st.success(msg)
        for err in errors:
            st.error(f"Reveal failed — {err}")
        st.rerun()  # re-render table with revealed emails merged in

    with st.expander("Debug: raw Apollo response"):
        st.json(result)


render()
