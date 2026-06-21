"""Thin Google API client for Search Console and Google Ads (REST + OAuth).

Auth is a long-lived OAuth refresh token (stored in secrets): we exchange it for
short-lived access tokens server-side, cached in-process until ~expiry. No Google
SDKs — plain httpx, matching the rest of the app.

Search Console: Search Analytics (clicks/impressions/CTR/position, top
queries/pages). Google Ads: GAQL reports (campaign/ad-group/keyword metrics).
"""

from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import quote

import httpx

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GSC_BASE = "https://www.googleapis.com/webmasters/v3"
_ADS_BASE = "https://googleads.googleapis.com"
# Google Ads ships ~monthly; bump this when upgrading. (Current: v23, 2026.)
GOOGLE_ADS_API_VERSION = "v23"

# Redact OAuth secrets / tokens from any error string.
_SECRET_RE = re.compile(
    r"(access_token|refresh_token|client_secret|developer-token)[\"'=:\s]+[^&\s\"']+",
    re.IGNORECASE,
)


def _redact(text: str) -> str:
    return _SECRET_RE.sub(r"\1=***", text)


class GoogleError(Exception):
    """Raised on Google API HTTP/transport errors or auth failures."""


# refresh_token -> (access_token, expiry_epoch). Avoids re-minting a token for
# every tool call within one question.
_TOKEN_CACHE: dict[str, tuple[str, float]] = {}


def _access_token(creds: dict[str, str]) -> str:
    """Exchange the refresh token for an access token (cached until ~expiry)."""
    rt = creds["refresh_token"]
    cached = _TOKEN_CACHE.get(rt)
    if cached and cached[1] - 60 > time.time():
        return cached[0]
    try:
        r = httpx.post(
            _TOKEN_URL,
            data={
                "client_id": creds["client_id"],
                "client_secret": creds["client_secret"],
                "refresh_token": rt,
                "grant_type": "refresh_token",
            },
            timeout=30.0,
        )
    except httpx.HTTPError as e:
        raise GoogleError(f"token network error: {_redact(str(e))}") from None
    if r.status_code >= 400:
        raise GoogleError(
            f"OAuth token refresh failed (HTTP {r.status_code}): "
            f"{_redact(r.text[:300])} — check the client id/secret/refresh token."
        )
    payload = r.json()
    token = payload.get("access_token")
    if not token:
        raise GoogleError("OAuth response had no access_token")
    _TOKEN_CACHE[rt] = (token, time.time() + float(payload.get("expires_in", 3600)))
    return token


def _get(url: str, token: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    return _call("GET", url, token, headers=headers)


def _post(
    url: str, token: str, *, json: dict[str, Any], headers: dict[str, str] | None = None
) -> dict[str, Any]:
    return _call("POST", url, token, json=json, headers=headers)


def _call(
    method: str,
    url: str,
    token: str,
    *,
    json: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    h = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if headers:
        h.update(headers)
    try:
        r = httpx.request(method, url, headers=h, json=json, timeout=45.0)
    except httpx.HTTPError as e:
        raise GoogleError(f"network error: {_redact(str(e))}") from None
    if r.status_code in (401, 403):
        raise GoogleError(
            f"HTTP {r.status_code}: Google rejected the request (check OAuth "
            f"scopes / that the account has access). {_redact(r.text[:300])}"
        )
    if r.status_code >= 400:
        raise GoogleError(f"HTTP {r.status_code}: {_redact(r.text[:400])}")
    try:
        return r.json()
    except ValueError:
        raise GoogleError(f"non-JSON response: {_redact(r.text[:200])}") from None


# ── Search Console ───────────────────────────────────────────────────────────


def gsc_list_sites(creds: dict[str, str]) -> list[dict[str, str]]:
    """List Search Console properties the authorized user can access."""
    token = _access_token(creds)
    payload = _get(f"{_GSC_BASE}/sites", token)
    return [
        {"site_url": s.get("siteUrl"), "permission": s.get("permissionLevel")}
        for s in payload.get("siteEntry", [])
    ]


def gsc_search_analytics(
    creds: dict[str, str],
    site_url: str,
    start_date: str,
    end_date: str,
    *,
    dimensions: list[str] | None = None,
    row_limit: int = 25,
) -> dict[str, Any]:
    """Query Search Analytics for a property over a date range.

    `dimensions` e.g. ["query"], ["page"], ["date"], ["country"], ["device"], or
    a combination; pass [] for aggregated totals. Each row carries clicks,
    impressions, ctr, and position. Dates are YYYY-MM-DD.
    """
    token = _access_token(creds)
    body: dict[str, Any] = {
        "startDate": start_date,
        "endDate": end_date,
        "rowLimit": min(int(row_limit), 25000),
    }
    if dimensions:
        body["dimensions"] = dimensions
    url = f"{_GSC_BASE}/sites/{quote(site_url, safe='')}/searchAnalytics/query"
    payload = _post(url, token, json=body)
    rows = []
    for row in payload.get("rows", []):
        rows.append(
            {
                "keys": row.get("keys"),
                "clicks": row.get("clicks"),
                "impressions": row.get("impressions"),
                "ctr": row.get("ctr"),
                "position": row.get("position"),
            }
        )
    return {"dimensions": dimensions or [], "rows": rows}


# ── Google Ads ───────────────────────────────────────────────────────────────


def ads_search(
    creds: dict[str, str],
    customer_id: str,
    query: str,
    *,
    developer_token: str,
    login_customer_id: str | None = None,
) -> list[dict[str, Any]]:
    """Run a GAQL query against a Google Ads customer and return result rows.

    `customer_id` / `login_customer_id` may include dashes; they're stripped.
    Requires the developer token (header) plus OAuth. login_customer_id is the
    manager (MCC) account id when access is via a manager.
    """
    token = _access_token(creds)
    cid = re.sub(r"\D", "", customer_id)
    headers = {"developer-token": developer_token}
    if login_customer_id:
        headers["login-customer-id"] = re.sub(r"\D", "", login_customer_id)
    url = f"{_ADS_BASE}/{GOOGLE_ADS_API_VERSION}/customers/{cid}/googleAds:search"
    payload = _post(url, token, json={"query": query}, headers=headers)
    return payload.get("results", [])
