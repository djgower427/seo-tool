"""Target Finder — build company target lists from criteria.

Company search via Apollo's /mixed_companies/search consumes ~1 credit per
submission. Per-company contact lookups use the free people api_search
endpoint and are triggered on demand (one click per company) so credits
aren't spent on companies the user isn't interested in.
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from seo_apollo import (
    SENIORITIES,
    ApolloError,
    organization_search,
    people_search,
)
from seo_claude import ClaudeError, expand_job_function
from seo_keys import get_anthropic_key, get_apollo_key

_SEARCH_CACHE_VERSION = 1

# Session state: domain → list of people dicts. Populated lazily when the user
# clicks "Show contacts" on a company; survives reruns so we don't re-fetch.
_CONTACTS_STORE_KEY = "_target_finder_contacts"


def _contacts_store() -> dict[str, list[dict[str, Any]]]:
    if _CONTACTS_STORE_KEY not in st.session_state:
        st.session_state[_CONTACTS_STORE_KEY] = {}
    return st.session_state[_CONTACTS_STORE_KEY]


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


def _render_company_summary(company: dict[str, Any]) -> None:
    location_parts = [company.get("city"), company.get("state"), company.get("country")]
    location = ", ".join(p for p in location_parts if p) or "—"
    industry = (company.get("industry") or "").title() or "—"
    employees = company.get("estimated_num_employees")
    employees_str = f"{int(employees):,}" if employees else "—"

    techs = company.get("technology_names") or company.get("current_technologies") or []
    tech_names: list[str] = []
    for t in techs:
        if isinstance(t, str):
            tech_names.append(t)
        elif isinstance(t, dict) and t.get("name"):
            tech_names.append(t["name"])

    rows = [
        ("Employees", employees_str),
        ("Industry", industry),
        ("HQ", location),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["Field", "Value"]),
        use_container_width=True,
        hide_index=True,
    )

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


def _build_tech_html(tech_names: list[str]) -> str:
    """Show the first 8 techs inline; the rest collapse behind a native HTML
    <details> "see N more" toggle. html.escape() because Apollo's data
    isn't user-controlled but neither is it strictly trusted."""
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


def _render_contacts_section(
    company: dict[str, Any],
    query: dict[str, Any],
    apollo_key: str,
) -> None:
    domain = company.get("primary_domain") or company.get("website_url") or ""
    if not domain:
        st.caption("No domain available for contact lookup.")
        return

    store = _contacts_store()
    if domain in store:
        people = store[domain]
        if not people:
            st.caption(
                "No contacts at this domain match the contact filters. "
                "Try widening Job function or removing seniority filters."
            )
            return
        rows = [
            {
                "Name (first)": p.get("first_name") or "—",
                "Title": p.get("title") or "—",
                "Seniority": p.get("seniority") or "—",
            }
            for p in people
        ]
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"To reveal full names + emails for these contacts, open the "
            f"**Find Contacts** page and search `{domain}`."
        )
        return

    # Not yet fetched — show button
    if st.button(
        f"Show contacts at {domain} (free)",
        key=f"contacts-{domain}",
    ):
        _fetch_contacts(domain, query, apollo_key)


def _fetch_contacts(
    domain: str,
    query: dict[str, Any],
    apollo_key: str,
) -> None:
    """Expand the job function via Claude (if set), then call Apollo. Store
    the result in session state and rerun."""
    function = query.get("function") or ""
    seniorities = query.get("seniorities") or []
    titles: list[str] = []

    if function:
        anthropic_key = get_anthropic_key()
        if not anthropic_key:
            st.error(
                "`ANTHROPIC_API_KEY` is not set. Leave Job function blank or "
                "add the key to Streamlit Cloud Secrets."
            )
            return
        try:
            titles = list(_cached_expand_function(
                function,
                tuple(seniorities),
                anthropic_key,
                _SEARCH_CACHE_VERSION,
            ))
        except ClaudeError as e:
            st.error(f"Claude — {e}")
            return

    try:
        result = people_search(
            domain,
            apollo_key,
            titles=titles or None,
            seniorities=seniorities or None,
            per_page=10,
        )
    except ApolloError as e:
        st.error(f"Apollo — {e}")
        return

    _contacts_store()[domain] = result.get("people") or []
    st.rerun()


def render() -> None:
    st.title("🎯 Target Finder")
    st.caption(
        "Find companies matching your criteria, then drill into key contacts. "
        "Company search consumes ~1 Apollo credit per submission. Contact "
        "lookups are free."
    )

    apollo_key = get_apollo_key()
    if not apollo_key:
        st.error(
            "`APOLLO_API_KEY` is not set. Add it to Streamlit Cloud Secrets "
            "or to `.streamlit/secrets.toml` locally."
        )
        st.stop()

    with st.form("target-finder-form"):
        st.markdown("**Company filters**")
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
        )

        st.markdown("**Contact filters** (used when expanding a company)")
        function_input = st.text_input(
            "Job function",
            placeholder="e.g. marketing leadership, head of growth",
            help=(
                "Plain-language description of the role. Claude expands this "
                "into concrete job titles before searching for contacts."
            ),
        )
        seniorities = st.multiselect(
            "Seniority",
            options=SENIORITIES,
            default=["c_suite", "vp", "head", "director"],
        )

        submitted = st.form_submit_button("Find companies (1 credit)", type="primary")

    if submitted:
        st.session_state["_target_finder_query"] = {
            "keywords": keywords_input.strip(),
            "emp_min": int(emp_min),
            "emp_max": int(emp_max),
            "location": location_input.strip(),
            "max_companies": int(max_companies),
            "function": function_input.strip(),
            "seniorities": seniorities,
        }
        # Wipe stale contacts so previous-search expansions don't leak in.
        st.session_state[_CONTACTS_STORE_KEY] = {}

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

    st.caption(
        f"Showing {len(companies)} of {total:,} companies matching your filters."
    )

    for company in companies:
        name = company.get("name") or "(unnamed)"
        domain = company.get("primary_domain") or company.get("website_url") or ""
        with st.expander(f"**{name}** — {domain}"):
            _render_company_summary(company)
            _render_contacts_section(company, query, apollo_key)

    with st.expander("Debug: raw Apollo response"):
        st.json(result)


render()
