"""Thin HubSpot API client.

Docs: https://developers.hubspot.com/docs/api/overview
Auth is a Private App access token sent as `Authorization: Bearer <token>`.
The token is long-lived, so no OAuth refresh is needed.

Covers our CRM (contacts, companies, deals) via the v3 Search API and marketing
campaign analytics via the Marketing Campaigns v3 API.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

import httpx

HUBSPOT_BASE = "https://api.hubapi.com"

# Redact bearer tokens / private-app tokens (pat-...) from any error string.
_KEY_RE = re.compile(r"(Bearer\s+|pat-[a-z0-9-]*)[A-Za-z0-9._-]+", re.IGNORECASE)


def _redact(text: str) -> str:
    return _KEY_RE.sub(r"\1***", text)


class HubSpotError(Exception):
    """Raised on HTTP errors, transport errors, or API-level errors."""


def _request(
    method: str,
    path: str,
    token: str,
    *,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{HUBSPOT_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        r = httpx.request(
            method, url, headers=headers, params=params, json=json, timeout=30.0
        )
    except httpx.HTTPError as e:
        raise HubSpotError(f"network error: {_redact(str(e))}") from None

    if r.status_code in (401, 403):
        raise HubSpotError(
            f"HTTP {r.status_code}: HubSpot rejected the request (check the "
            f"private-app token and that it has the needed scopes for this data)"
        )
    if r.status_code == 429:
        raise HubSpotError(
            "HTTP 429: HubSpot rate limit hit. Wait a moment and retry."
        )
    if r.status_code >= 400:
        body = _redact(r.text.strip() or r.reason_phrase)
        raise HubSpotError(f"HTTP {r.status_code}: {body}")

    try:
        return r.json()
    except ValueError:
        raise HubSpotError(f"non-JSON response: {_redact(r.text[:200])}") from None


# Default properties to return per object — readable fields for brief answers.
_CONTACT_PROPS = [
    "firstname", "lastname", "email", "jobtitle", "company",
    "lifecyclestage", "hs_lead_status",
]
_COMPANY_PROPS = [
    "name", "domain", "industry", "numberofemployees", "annualrevenue",
    "lifecyclestage", "city", "state", "country",
]
_DEAL_PROPS = [
    "dealname", "amount", "dealstage", "pipeline", "closedate",
    "hs_is_closed_won",
]

_DEFAULT_PROPS = {
    "contacts": _CONTACT_PROPS,
    "companies": _COMPANY_PROPS,
    "deals": _DEAL_PROPS,
}


def search_objects(
    object_type: str,
    query: str,
    token: str,
    *,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Full-text search one CRM object type (contacts | companies | deals).

    `query` is matched against the object's default searchable properties
    (HubSpot's `query` search param). Returns a list of records, each a flat
    dict of the record id plus its requested properties.
    """
    if object_type not in _DEFAULT_PROPS:
        raise HubSpotError(f"unsupported object type: {object_type}")
    payload = _request(
        "POST",
        f"/crm/v3/objects/{object_type}/search",
        token,
        json={
            "query": query,
            "properties": _DEFAULT_PROPS[object_type],
            "limit": min(int(limit), 100),
        },
    )
    out: list[dict[str, Any]] = []
    for r in payload.get("results", []):
        props = r.get("properties") or {}
        out.append({"id": r.get("id"), **{k: v for k, v in props.items() if v is not None}})
    return out


def list_campaigns(
    token: str,
    *,
    name_contains: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List marketing campaigns (id + name), optionally filtered by a name
    substring (matched client-side, case-insensitive)."""
    payload = _request(
        "GET",
        "/marketing/v3/campaigns",
        token,
        params={"properties": "hs_name", "limit": min(int(limit), 100)},
    )
    out: list[dict[str, Any]] = []
    needle = (name_contains or "").strip().lower()
    for c in payload.get("results", []):
        props = c.get("properties") or {}
        name = props.get("hs_name") or c.get("name") or ""
        if needle and needle not in name.lower():
            continue
        out.append({"id": c.get("id"), "name": name})
    return out


def campaign_metrics(
    token: str,
    campaign_guid: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Attribution metrics for one campaign (sessions, new/influenced contacts,
    etc.) over a date range. Dates are YYYY-MM-DD; default to the last 90 days."""
    if not end_date:
        end_date = date.today().isoformat()
    if not start_date:
        start_date = (date.today() - timedelta(days=90)).isoformat()
    return _request(
        "GET",
        f"/marketing/v3/campaigns/{campaign_guid}/reports/metrics",
        token,
        params={"startDate": start_date, "endDate": end_date},
    )
