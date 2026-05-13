"""Thin Semrush API client.

Docs: https://developer.semrush.com/api/
All endpoints return CSV (semicolon-delimited) with a header row.
"""

from __future__ import annotations

import httpx

SEMRUSH_BASE = "https://api.semrush.com/"


class SemrushError(Exception):
    """Raised when Semrush returns an ERROR response or empty data."""


def _request(params: dict[str, str]) -> list[dict[str, str]]:
    """Call Semrush API and parse CSV into a list of dicts.

    Semrush surfaces validation errors two different ways: as HTTP 4xx with a
    plain-text body (e.g. bad column code), or as HTTP 200 with the body
    starting with "ERROR ..." (e.g. domain not in database, out of credits).
    We treat both as SemrushError so the UI gets the real message.
    """
    r = httpx.get(SEMRUSH_BASE, params=params, timeout=30.0)
    text = r.text.strip()
    report_type = params.get("type", "?")
    if r.status_code >= 400:
        body = text or r.reason_phrase
        raise SemrushError(f"[{report_type}] HTTP {r.status_code}: {body}")
    if text.startswith("ERROR"):
        raise SemrushError(f"[{report_type}] {text}")
    if not text:
        return []
    lines = text.splitlines()
    header = lines[0].split(";")
    return [dict(zip(header, line.split(";"))) for line in lines[1:]]


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

    `Dt` comes back as YYYYMMDD.
    """
    return _request(
        {
            "type": "domain_rank_history",
            "key": api_key,
            "domain": domain,
            "database": database,
            "export_columns": "Rk,Or,Ot,Oc,Dt",
            "display_limit": str(limit),
            "display_sort": "dt_desc",
        }
    )


def top_pages(
    domain: str, database: str, api_key: str, limit: int = 25
) -> list[dict[str, str]]:
    """Top pages by estimated organic traffic.

    Columns: Ur (URL), Pc (# keywords), Tg (traffic), Tc (traffic cost).
    """
    return _request(
        {
            "type": "domain_organic_pages",
            "key": api_key,
            "domain": domain,
            "database": database,
            "export_columns": "Ur,Pc,Tg,Tc",
            "display_limit": str(limit),
            "display_sort": "tg_desc",
        }
    )
