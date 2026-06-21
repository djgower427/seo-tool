"""Thin Semrush API client.

Docs: https://developer.semrush.com/api/
All endpoints return CSV (semicolon-delimited) with a header row.
"""

from __future__ import annotations

import re

import httpx

SEMRUSH_BASE = "https://api.semrush.com/"

# Free endpoint that returns the remaining API-unit balance as a plain integer.
# Does NOT itself consume units, so it's safe to call before and after a billable
# request to measure exact consumption.
SEMRUSH_UNITS_URL = "https://www.semrush.com/users/countapiunits.html"

_KEY_RE = re.compile(r"(key=)[^&\s]+", re.IGNORECASE)


def _redact(text: str) -> str:
    """Mask Semrush API keys in any string before it's surfaced to the UI."""
    return _KEY_RE.sub(r"\1***", text)


# Some endpoints (notably domain_rank_history) ignore `export_columns` and
# return full English headers instead of the short codes. Normalize back to
# short codes so the rest of the code can rely on a single naming scheme.
_HEADER_ALIASES = {
    "Rank": "Rk",
    "Organic Keywords": "Or",
    "Organic Traffic": "Ot",
    "Organic Cost": "Oc",
    "Adwords Keywords": "Ad",
    "Adwords Traffic": "At",
    "Adwords Cost": "Ac",
    "Date": "Dt",
    "Domain": "Dn",
    "Database": "Db",
    "Keyword": "Ph",
    "Position": "Po",
    "Previous Position": "Pp",
    "Position Difference": "Pd",
    "Search Volume": "Nq",
    "CPC": "Cp",
    "URL": "Ur",
    "Url": "Ur",
    "Traffic (%)": "Tr",
    "Traffic Cost (%)": "Tc",
    "Competition": "Co",
    "Number of Results": "Nr",
    "Trends": "Td",
    "Keyword Difficulty": "Kd",
}


def _normalize_headers(header: list[str]) -> list[str]:
    return [_HEADER_ALIASES.get(h.strip(), h.strip()) for h in header]


class SemrushError(Exception):
    """Raised when Semrush returns an ERROR response or empty data."""


def raw_request(params: dict[str, str]) -> str:
    """Same wire call as _request but returns the raw response text (redacted on error).

    Useful for a debug view of exactly what Semrush sent back.
    """
    report_type = params.get("type", "?")
    try:
        r = httpx.get(SEMRUSH_BASE, params=params, timeout=30.0)
    except httpx.HTTPError as e:
        raise SemrushError(f"[{report_type}] network error: {_redact(str(e))}") from None
    if r.status_code >= 400:
        body = _redact(r.text.strip() or r.reason_phrase)
        raise SemrushError(f"[{report_type}] HTTP {r.status_code}: {body}")
    return r.text


def _request(params: dict[str, str]) -> list[dict[str, str]]:
    """Call Semrush API and parse CSV into a list of dicts.

    Semrush surfaces validation errors two different ways: as HTTP 4xx with a
    plain-text body (e.g. bad column code), or as HTTP 200 with the body
    starting with "ERROR ..." (e.g. domain not in database, out of credits).
    We treat both as SemrushError so the UI gets the real message. All errors
    pass through _redact() so the API key never reaches the UI even if httpx
    includes the request URL in its exception message.
    """
    report_type = params.get("type", "?")
    try:
        r = httpx.get(SEMRUSH_BASE, params=params, timeout=30.0)
    except httpx.HTTPError as e:
        # `from None` drops the chained traceback that would otherwise contain
        # the un-redacted request URL.
        raise SemrushError(f"[{report_type}] network error: {_redact(str(e))}") from None

    text = r.text.strip()
    if r.status_code >= 400:
        body = _redact(text or r.reason_phrase)
        raise SemrushError(f"[{report_type}] HTTP {r.status_code}: {body}")
    if text.startswith("ERROR"):
        raise SemrushError(f"[{report_type}] {_redact(text)}")
    if not text:
        return []
    lines = text.splitlines()
    header = _normalize_headers(lines[0].split(";"))
    return [dict(zip(header, line.split(";"))) for line in lines[1:]]


def units_balance(api_key: str) -> int | None:
    """Remaining Semrush API units for this key, or None if it can't be read.

    Hits the free countapiunits endpoint, which returns the balance as a bare
    integer (e.g. "10000"). This call doesn't consume units. Returns None on any
    transport/parse error so callers (usage tracking) can degrade gracefully
    rather than break the page.
    """
    try:
        r = httpx.get(SEMRUSH_UNITS_URL, params={"key": api_key}, timeout=15.0)
    except httpx.HTTPError:
        return None
    if r.status_code >= 400:
        return None
    text = r.text.strip()
    try:
        return int(text)
    except ValueError:
        return None


def domain_organic_keywords(
    domain: str,
    database: str,
    api_key: str,
    *,
    limit: int = 100,
    sort: str = "tr_desc",
) -> list[dict[str, str]]:
    """Organic keywords a domain ranks for, with position, volume, and difficulty.

    Pulls the `domain_organic` report. Each row carries:
      Ph (keyword), Po (position), Nq (search volume), Kd (keyword difficulty,
      0-100), Cp (CPC), Co (competition), Tr (traffic share %), Ur (ranking URL).

    Costs 10 Semrush credits per returned row (same as the keyword pool used by
    Full Site Status). `sort` is a Semrush display_sort code — default `tr_desc`
    (most traffic first); pass `nq_desc` to bias toward high-volume keywords.

    Returns [] for a domain with no organic keywords in this database. Raises
    SemrushError on an API-level error (out of credits, bad column, etc.).
    """
    return _request(
        {
            "type": "domain_organic",
            "key": api_key,
            "domain": domain,
            "database": database,
            "export_columns": "Ph,Po,Nq,Kd,Cp,Co,Tr,Ur",
            "display_limit": str(limit),
            "display_sort": sort,
        }
    )


def domain_overview(domain: str, database: str, api_key: str) -> dict[str, str] | None:
    """Single-row snapshot: rank, organic keywords, organic traffic, ad stats."""
    rows = _request(
        {
            "type": "domain_ranks",
            "key": api_key,
            "domain": domain,
            "database": database,
            "export_columns": "Dn,Rk,Or,Ot,Oc,Ad,At,Ac",
        }
    )
    return rows[0] if rows else None


def domain_rank_history(
    domain: str, database: str, api_key: str, limit: int = 24
) -> list[dict[str, str]]:
    """Monthly history of rank, organic keywords, organic traffic, organic cost.

    `Dt` comes back as YYYYMMDD. We don't pass display_sort because dt_desc
    isn't a documented value for this endpoint; Semrush returns history in
    reverse-chronological order by default.
    """
    return _request(
        {
            "type": "domain_rank_history",
            "key": api_key,
            "domain": domain,
            "database": database,
            "export_columns": "Rk,Or,Ot,Oc,Dt",
            "display_limit": str(limit),
        }
    )


def top_pages(
    domain: str, database: str, api_key: str, limit: int = 25, keyword_pool: int = 100
) -> list[dict]:
    """Top pages by estimated organic traffic share.

    Semrush's standard Domain Analytics API has no direct "top pages" endpoint
    (that's part of the separate Trends API). We approximate it by pulling the
    top `keyword_pool` organic keywords for the domain and aggregating by URL:
    each page's score is the sum of `Tr` (traffic share %) across its ranking
    keywords. Returns the top `limit` URLs sorted by traffic share.

    If Semrush returns keyword rows but none expose a URL column, raises
    SemrushError with the column names it did return so the alias map can be
    extended.
    """
    rows = _request(
        {
            "type": "domain_organic",
            "key": api_key,
            "domain": domain,
            "database": database,
            "export_columns": "Ph,Po,Nq,Ur,Tr",
            "display_limit": str(keyword_pool),
            "display_sort": "tr_desc",
        }
    )

    if not rows:
        raise SemrushError(
            "domain_organic returned 0 rows. This is expected for domains with "
            "no organic keywords in this database, but if the overview shows "
            "organic traffic > 0 then your API plan may not include the "
            "domain_organic endpoint."
        )

    by_url: dict[str, dict] = {}
    for r in rows:
        url = (r.get("Ur") or "").strip()
        if not url:
            continue
        agg = by_url.setdefault(
            url, {"Ur": url, "Keywords": 0, "TrafficShare": 0.0}
        )
        agg["Keywords"] += 1
        try:
            agg["TrafficShare"] += float(r.get("Tr") or 0)
        except ValueError:
            pass

    if not by_url:
        sample_cols = list(rows[0].keys())
        raise SemrushError(
            f"got {len(rows)} keyword rows from domain_organic but none had a "
            f"non-empty URL (Ur) column. Columns returned: {sample_cols}"
        )

    return sorted(by_url.values(), key=lambda x: x["TrafficShare"], reverse=True)[:limit]
