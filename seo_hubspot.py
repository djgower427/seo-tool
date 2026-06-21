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


_DEAL_REPORT_PROPS = [
    "dealname", "amount", "dealstage", "pipeline", "closedate",
    "hs_closed_won_date", "hs_is_closed_won",
]


def list_properties(
    object_type: str,
    token: str,
    *,
    search: str | None = None,
    limit: int = 60,
) -> list[dict[str, Any]]:
    """List an object's properties (name, label, type, and enum options).

    Lets a caller discover the right field to filter on — e.g. find a "Deal
    Source" / "Lead Source" property and the exact value that means "Inbound".
    `search` filters to properties whose name or label contains the substring
    (case-insensitive). Reading properties needs a crm.schemas.{object}.read
    scope on the token.
    """
    payload = _request("GET", f"/crm/v3/properties/{object_type}", token)
    needle = (search or "").strip().lower()
    out: list[dict[str, Any]] = []
    for p in payload.get("results", []):
        name, label = p.get("name") or "", p.get("label") or ""
        if needle and needle not in name.lower() and needle not in label.lower():
            continue
        item: dict[str, Any] = {
            "name": name,
            "label": label,
            "type": p.get("type"),
            "fieldType": p.get("fieldType"),
        }
        options = p.get("options") or []
        if options:
            item["options"] = [
                {"label": o.get("label"), "value": o.get("value")}
                for o in options[:50]
            ]
        out.append(item)
        if len(out) >= limit:
            break
    return out


def aggregate_deals(
    token: str,
    *,
    only_closed_won: bool = False,
    dealstage: str | None = None,
    pipeline: str | None = None,
    date_property: str | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    extra_filters: list[dict[str, Any]] | None = None,
    max_records: int = 10000,
) -> dict[str, Any]:
    """Count and sum (by `amount`) deals matching structured filters.

    Uses the CRM Search API's filterGroups rather than full-text search, so it
    can filter by closed-won status, deal stage, pipeline, and a date range on a
    chosen date property (epoch-ms bounds). HubSpot returns an accurate `total`
    for the count; we paginate (100/page, up to `max_records`) to sum `amount`.

    `date_property` is e.g. "hs_closed_won_date" (when a deal entered closed
    won) or "closedate". Pass start_ms/end_ms as epoch milliseconds.

    Returns {count, summed_records, total_amount, truncated, deals(sample)}.
    """
    filters: list[dict[str, Any]] = []
    if only_closed_won:
        filters.append({"propertyName": "hs_is_closed_won", "operator": "EQ", "value": "true"})
    if dealstage:
        filters.append({"propertyName": "dealstage", "operator": "EQ", "value": dealstage})
    if pipeline:
        filters.append({"propertyName": "pipeline", "operator": "EQ", "value": pipeline})
    if date_property and start_ms is not None and end_ms is not None:
        filters.append({
            "propertyName": date_property, "operator": "BETWEEN",
            "value": str(start_ms), "highValue": str(end_ms),
        })
    elif date_property and start_ms is not None:
        filters.append({"propertyName": date_property, "operator": "GTE", "value": str(start_ms)})
    elif date_property and end_ms is not None:
        filters.append({"propertyName": date_property, "operator": "LTE", "value": str(end_ms)})
    if extra_filters:
        filters.extend(extra_filters)

    sort_prop = date_property or "createdate"
    filter_groups = [{"filters": filters}] if filters else []

    results: list[dict[str, Any]] = []
    after: str | None = None
    total: int | None = None
    while len(results) < max_records:
        body: dict[str, Any] = {
            "filterGroups": filter_groups,
            "properties": _DEAL_REPORT_PROPS,
            "sorts": [{"propertyName": sort_prop, "direction": "DESCENDING"}],
            "limit": 100,
        }
        if after:
            body["after"] = after
        payload = _request("POST", "/crm/v3/objects/deals/search", token, json=body)
        if total is None:
            total = payload.get("total")
        batch = payload.get("results", [])
        results.extend(batch)
        after = ((payload.get("paging") or {}).get("next") or {}).get("after")
        if not after or not batch:
            break

    total_amount = 0.0
    for r in results:
        amt = (r.get("properties") or {}).get("amount")
        try:
            total_amount += float(amt) if amt not in (None, "") else 0.0
        except (TypeError, ValueError):
            pass

    count = total if total is not None else len(results)
    sample = []
    for r in results[:25]:
        p = r.get("properties") or {}
        sample.append({
            "id": r.get("id"),
            "dealname": p.get("dealname"),
            "amount": p.get("amount"),
            "dealstage": p.get("dealstage"),
            "closedate": p.get("closedate"),
            "hs_closed_won_date": p.get("hs_closed_won_date"),
        })
    return {
        "count": count,
        "summed_records": len(results),
        "total_amount": round(total_amount, 2),
        "truncated": count > len(results),
        "deals": sample,
    }


def get_associations_batch(
    from_object_type: str,
    to_object_type: str,
    ids: list[str],
    token: str,
) -> dict[str, list[str]]:
    """Map each from-object id to its associated to-object ids (v4 batch read).

    e.g. from_object_type="deals", to_object_type="companies" returns
    {deal_id: [company_id, ...]}. Uses the v4 associations batch endpoint, so a
    list of deals resolves in one call. Reading associations needs the relevant
    crm.objects.*.read scopes on the token.
    """
    if not ids:
        return {}
    payload = _request(
        "POST",
        f"/crm/v4/associations/{from_object_type}/{to_object_type}/batch/read",
        token,
        json={"inputs": [{"id": str(i)} for i in ids]},
    )
    out: dict[str, list[str]] = {}
    for row in payload.get("results", []):
        frm = (row.get("from") or {}).get("id")
        if frm is None:
            continue
        to_ids = [
            str(t.get("toObjectId"))
            for t in (row.get("to") or [])
            if t.get("toObjectId") is not None
        ]
        out[str(frm)] = to_ids
    return out


def batch_read_objects(
    object_type: str,
    ids: list[str],
    token: str,
    *,
    properties: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch properties for a list of object ids in one batch call.

    Returns {id: {id, ...properties}} for the records HubSpot returns. Defaults
    to the object type's standard readable properties.
    """
    if not ids:
        return {}
    props = properties or _DEFAULT_PROPS.get(object_type, [])
    payload = _request(
        "POST",
        f"/crm/v3/objects/{object_type}/batch/read",
        token,
        json={"properties": props, "inputs": [{"id": str(i)} for i in ids]},
    )
    out: dict[str, dict[str, Any]] = {}
    for r in payload.get("results", []):
        rid = str(r.get("id"))
        p = r.get("properties") or {}
        out[rid] = {"id": r.get("id"), **{k: v for k, v in p.items() if v is not None}}
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


def traffic_sources(token: str, *, start: str, end: str) -> dict[str, Any]:
    """Website traffic broken down by source (organic, paid, direct, …) for a
    date range. `start`/`end` are YYYYMMDD strings (HubSpot's required format).

    Returns the raw payload, whose `breakdowns` list has one entry per source
    with a `visits` field (HubSpot's term for sessions) and `contacts`. Requires
    the private app to have an analytics / business-intelligence read scope.
    """
    return _request(
        "GET",
        "/analytics/v2/reports/sources/total",
        token,
        params={"start": start, "end": end},
    )


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
