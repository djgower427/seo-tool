"""Thin Apollo.io API client.

Docs: https://docs.apollo.io/reference
Apollo's REST API uses JSON; auth via the X-Api-Key header.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

APOLLO_BASE = "https://api.apollo.io/api/v1"

# Match the key anywhere it might leak into an error string: header value,
# URL fragment (some endpoints still accept ?api_key=), or accidental log lines.
_KEY_RE = re.compile(r"(api[_-]?key[=:\s\"']+)[^&\s\"']+", re.IGNORECASE)


def _redact(text: str) -> str:
    return _KEY_RE.sub(r"\1***", text)


class ApolloError(Exception):
    """Raised on HTTP errors, transport errors, or API-level errors."""


def _request(
    method: str,
    path: str,
    api_key: str,
    *,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{APOLLO_BASE}{path}"
    headers = {"X-Api-Key": api_key, "Accept": "application/json"}
    try:
        r = httpx.request(
            method, url, headers=headers, params=params, json=json, timeout=30.0
        )
    except httpx.HTTPError as e:
        # `from None` drops the chained traceback that could include the URL.
        raise ApolloError(f"network error: {_redact(str(e))}") from None

    if r.status_code == 401 or r.status_code == 403:
        raise ApolloError(
            f"HTTP {r.status_code}: Apollo rejected the API key "
            f"(check it's valid and your plan allows this endpoint)"
        )
    if r.status_code == 429:
        raise ApolloError(
            "HTTP 429: Apollo rate limit hit. Wait a minute and retry, or "
            "check your plan's per-minute / per-hour limits."
        )
    if r.status_code >= 400:
        body = _redact(r.text.strip() or r.reason_phrase)
        raise ApolloError(f"HTTP {r.status_code}: {body}")

    try:
        return r.json()
    except ValueError:
        raise ApolloError(f"non-JSON response: {_redact(r.text[:200])}") from None


def organization_enrich(domain: str, api_key: str) -> dict[str, Any] | None:
    """Look up a company by domain and return Apollo's organization record.

    Returns None if Apollo has no record for this domain. Costs 1 credit per
    call regardless of which fields the caller reads — the full payload (size,
    industry, funding, technologies, social links) comes back in one response.

    Apollo response shape: {"organization": {...}} on success; the org dict may
    be absent or null if no match. We return the inner dict directly so callers
    can use .get() to pick fields safely.
    """
    payload = _request(
        "GET",
        "/organizations/enrich",
        api_key,
        params={"domain": domain},
    )
    org = payload.get("organization")
    return org if isinstance(org, dict) and org else None


# Apollo's documented seniority enum. Surface in the UI so users don't guess.
SENIORITIES = [
    "owner",
    "founder",
    "c_suite",
    "partner",
    "vp",
    "head",
    "director",
    "manager",
    "senior",
    "entry",
    "intern",
]

def people_search(
    domain: str,
    api_key: str,
    *,
    titles: list[str] | None = None,
    seniorities: list[str] | None = None,
    page: int = 1,
    per_page: int = 25,
) -> dict[str, Any]:
    """Search for people at a given company domain. FREE endpoint — no credits.

    Returns the raw response so callers can use both `people` (the results
    list) and `pagination` (total counts, page info). Emails are NOT returned
    by this endpoint; each person record has an `id` to use with person_reveal().

    `titles` is passed to Apollo's person_titles[] filter, which does fuzzy
    server-side matching — pass several variants of a role (e.g. "VP Marketing"
    AND "Vice President of Marketing") for broader recall.
    """
    body: dict[str, Any] = {
        "q_organization_domains_list": [domain],
        "page": page,
        "per_page": per_page,
    }
    if titles:
        body["person_titles"] = titles
    if seniorities:
        body["person_seniorities"] = seniorities

    return _request(
        "POST",
        "/mixed_people/api_search",
        api_key,
        json=body,
    )


def organization_search(
    api_key: str,
    *,
    keywords: str | None = None,
    employees_min: int | None = None,
    employees_max: int | None = None,
    locations: list[str] | None = None,
    page: int = 1,
    per_page: int = 25,
) -> dict[str, Any]:
    """Search Apollo's company database. CONSUMES CREDITS per call (typically 1).

    Apollo's standard pagination knobs apply: up to 100 per_page, up to 500
    pages, 50K records hard cap. Returns the raw response with both
    `organizations` and `pagination` so the caller can paginate.

    `keywords` is a free-text query Apollo fuzzy-matches across company name,
    description, industry, and detected technologies — handy for filters like
    "SaaS HubSpot" without needing technology UIDs. `employees_min`/`_max`
    map to organization_num_employees_ranges[] in the shape Apollo expects.
    `locations` is a list of HQ location strings (countries, states, cities).
    """
    body: dict[str, Any] = {
        "page": page,
        "per_page": per_page,
    }
    if keywords:
        body["q_keywords"] = keywords
    if employees_min is not None or employees_max is not None:
        lo = employees_min if employees_min is not None else 1
        hi = employees_max if employees_max is not None else 100000
        body["organization_num_employees_ranges"] = [f"{lo},{hi}"]
    if locations:
        body["organization_locations"] = locations

    return _request(
        "POST",
        "/mixed_companies/search",
        api_key,
        json=body,
    )


def person_match(
    person_id: str,
    api_key: str,
    *,
    reveal_personal_emails: bool = False,
) -> dict[str, Any] | None:
    """Enrich one person via Apollo's /people/match endpoint.

    Without `reveal_personal_emails`, returns basic enrichment fields:
    un-obfuscated first/last name, title, seniority, organization, location,
    linkedin_url, etc. The search-preview endpoint returns last_name as just
    an initial; only /people/match gives you the full surname.

    With `reveal_personal_emails=True`, also returns the personal email and
    consumes additional credits per Apollo's pricing. Apollo won't reveal
    emails in GDPR-protected regions — no credit charged in that case, but
    the email field comes back masked or null.

    Apollo's docs are vague on whether the bare match call (without reveals)
    consumes credits — callers should watch their balance on the first
    request to confirm.

    Returns the `person` dict from Apollo's response, or None if not found.
    """
    payload = _request(
        "POST",
        "/people/match",
        api_key,
        json={
            "id": person_id,
            "reveal_personal_emails": reveal_personal_emails,
        },
    )
    person = payload.get("person")
    return person if isinstance(person, dict) and person else None


# Back-compat alias for the original name-revealing call.
def person_reveal(person_id: str, api_key: str) -> dict[str, Any] | None:
    """Reveal contact details (incl. personal email) for one person.
    Thin wrapper around person_match(..., reveal_personal_emails=True).
    """
    return person_match(person_id, api_key, reveal_personal_emails=True)
