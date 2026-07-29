"""Thin Claude/Anthropic helper for SEO tool features.

Two functions:
  - expand_job_function: turn a free-form job function (e.g. "marketing
    leadership") into a concrete list of job titles for Apollo's fuzzy-matched
    person_titles[] filter.
  - build_similar_company_query: turn a seed company's Apollo profile into the
    structured Apollo organization_search parameters that find similar companies
    (by industry + size).
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

import seo_usage

_MAX_TITLES = 20
_MODEL = "claude-haiku-4-5"


class ClaudeError(Exception):
    """Raised on Anthropic API errors or malformed responses."""


def expand_job_function(
    function: str,
    api_key: str,
    *,
    seniorities: list[str] | None = None,
) -> list[str]:
    """Use Claude to turn a job function into concrete job-title variants.

    e.g. "marketing leadership" → ["CMO", "VP Marketing", "Head of Marketing", ...].
    Caller passes the result to Apollo's person_titles[] filter — Apollo does
    fuzzy server-side matching, so including common variants ("VP Marketing"
    and "Vice President of Marketing") boosts recall.
    """
    client = anthropic.Anthropic(api_key=api_key)

    seniority_hint = ""
    if seniorities:
        seniority_hint = (
            f" Focus on titles consistent with these seniority levels: "
            f"{', '.join(seniorities)}."
        )

    prompt = (
        f'Generate concrete job titles for the function: "{function}".'
        f"{seniority_hint}\n\n"
        f"Requirements:\n"
        f"- 6 to {_MAX_TITLES} titles\n"
        f"- Real titles people put on LinkedIn (e.g. 'VP of Marketing', 'CMO'),"
        f" not department names ('Marketing') or vague phrases ('marketing leader')\n"
        f"- Include common variants ('VP Marketing' AND 'Vice President of Marketing')"
        f" so fuzzy matching catches both spellings\n"
        f"- Do not include the function itself as a title"
    )

    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "titles": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["titles"],
                        "additionalProperties": False,
                    },
                }
            },
        )
    except anthropic.AuthenticationError:
        raise ClaudeError(
            "Anthropic rejected the API key — check ANTHROPIC_API_KEY"
        ) from None
    except anthropic.RateLimitError:
        raise ClaudeError(
            "Anthropic rate limit hit — wait a minute and retry"
        ) from None
    except anthropic.APIError as e:
        raise ClaudeError(f"Anthropic API error: {e}") from None

    seo_usage.record_claude(_MODEL, response.usage)

    try:
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        raw_titles = data.get("titles", [])
    except (StopIteration, json.JSONDecodeError, AttributeError) as e:
        raise ClaudeError(f"could not parse Claude response: {e}") from None

    # Case-insensitive dedupe, preserving original casing of first occurrence.
    seen = set()
    out: list[str] = []
    for t in raw_titles:
        if not isinstance(t, str):
            continue
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(t.strip())
        if len(out) >= _MAX_TITLES:
            break
    return out


# Apollo's organization_search caps employee/revenue with these sentinels, so
# Claude returns concrete numbers and we never send an open-ended range.
_EMPLOYEE_CEILING = 100_000

_SIMILAR_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "string",
            "description": (
                "Free-text query for Apollo's q_keywords filter, describing the "
                "industry / business model / category — e.g. 'B2B payments API "
                "SaaS'. A few words, not a sentence."
            ),
        },
        "employees_min": {"type": "integer"},
        "employees_max": {"type": "integer"},
        "revenue_min": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "revenue_max": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "locations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "0-2 HQ location strings (country/state/city), or empty for no constraint.",
        },
        "rationale": {
            "type": "string",
            "description": "One sentence on how these criteria capture companies similar to the seed.",
        },
    },
    "required": [
        "keywords",
        "employees_min",
        "employees_max",
        "revenue_min",
        "revenue_max",
        "locations",
        "rationale",
    ],
    "additionalProperties": False,
}


def build_similar_company_query(
    profile: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    """Turn a seed company's Apollo profile into Apollo organization_search params.

    `profile` is a distilled dict from organization_enrich — e.g.
    {"name", "industry", "keywords", "description", "employees", "revenue"}.
    Claude reads it and returns an employee range bracketing the seed's size, a
    keyword query capturing its industry/category, and (optionally) a revenue
    range and location constraint — the shape Apollo's company search expects to
    surface peers.

    Returns a dict with keys: keywords (str), employees_min/_max (int),
    revenue_min/_max (int | None), locations (list[str]), rationale (str).
    Raises ClaudeError on API errors or a malformed response.
    """
    client = anthropic.Anthropic(api_key=api_key)

    prompt = (
        "You are given an Apollo company profile for a SEED company. Produce the "
        "parameters for an Apollo company search that finds OTHER companies "
        "similar to it — same industry/category and comparable size.\n\n"
        f"Seed company profile (JSON):\n{json.dumps(profile, indent=2)}\n\n"
        "Requirements:\n"
        "- keywords: a short free-text query capturing the seed's industry, "
        "business model, and category. Prefer the seed's industry and any "
        "keyword tags over its brand name (we want peers, not the seed itself).\n"
        "- employees_min / employees_max: an integer range that BRACKETS the "
        "seed's headcount so similarly-sized companies match. If the seed has "
        f"~N employees, a range like roughly N/3 to N*3 works. Cap max at "
        f"{_EMPLOYEE_CEILING}, floor min at 1.\n"
        "- revenue_min / revenue_max: raw USD integers bracketing the seed's "
        "revenue, or null for a side you can't reasonably infer. Use null for "
        "both if the seed's revenue is unknown.\n"
        "- locations: leave empty unless the seed is clearly tied to one region "
        "where peers would cluster; at most 2 entries.\n"
        "- rationale: one sentence explaining the criteria."
    )

    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {"type": "json_schema", "schema": _SIMILAR_QUERY_SCHEMA}
            },
        )
    except anthropic.AuthenticationError:
        raise ClaudeError(
            "Anthropic rejected the API key — check ANTHROPIC_API_KEY"
        ) from None
    except anthropic.RateLimitError:
        raise ClaudeError(
            "Anthropic rate limit hit — wait a minute and retry"
        ) from None
    except anthropic.APIError as e:
        raise ClaudeError(f"Anthropic API error: {e}") from None

    seo_usage.record_claude(_MODEL, response.usage)

    try:
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
    except (StopIteration, json.JSONDecodeError, AttributeError) as e:
        raise ClaudeError(f"could not parse Claude response: {e}") from None

    # Clamp the employee range into Apollo's accepted bounds defensively — the
    # schema can't express numeric min/max, so Claude could exceed them.
    emp_min = max(1, int(data.get("employees_min") or 1))
    emp_max = min(_EMPLOYEE_CEILING, int(data.get("employees_max") or _EMPLOYEE_CEILING))
    if emp_max < emp_min:
        emp_min, emp_max = emp_max, emp_min

    def _opt_int(v: Any) -> int | None:
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    locations = [
        s.strip()
        for s in (data.get("locations") or [])
        if isinstance(s, str) and s.strip()
    ]

    return {
        "keywords": (data.get("keywords") or "").strip(),
        "employees_min": emp_min,
        "employees_max": emp_max,
        "revenue_min": _opt_int(data.get("revenue_min")),
        "revenue_max": _opt_int(data.get("revenue_max")),
        "locations": locations,
        "rationale": (data.get("rationale") or "").strip(),
    }


# The recommendation is a short, judgment-heavy writeup (relevance filtering +
# prose), so it's worth the strongest model — the call is infrequent and the
# output is tiny, so cost stays trivial.
_RECOMMEND_MODEL = "claude-opus-4-8"

_RECOMMEND_SCHEMA = {
    "type": "object",
    "properties": {
        "keyword": {
            "type": "string",
            "description": (
                "The single recommended keyword to target, copied EXACTLY from "
                "the candidate list. Empty string if none of the candidates "
                "relate to our offerings."
            ),
        },
        "why": {
            "type": "string",
            "description": (
                "1-2 sentences on why this keyword is a good target — tie its "
                "relevance to our offerings together with its volume/difficulty. "
                "Empty when no keyword is recommended."
            ),
        },
        "blog_title": {
            "type": "string",
            "description": (
                "A concrete, compelling blog-post title we could publish to rank "
                "for the keyword. Empty when no keyword is recommended."
            ),
        },
        "blog_angle": {
            "type": "string",
            "description": (
                "1-2 sentences on the angle/content of that blog post and why it "
                "would capture the traffic. Empty when no keyword is recommended."
            ),
        },
        "no_fit_reason": {
            "type": "string",
            "description": (
                "When keyword is empty: one sentence on why none of the "
                "candidates relate to our offerings. Empty otherwise."
            ),
        },
    },
    "required": ["keyword", "why", "blog_title", "blog_angle", "no_fit_reason"],
    "additionalProperties": False,
}


def recommend_keyword_to_steal(
    candidates: list[dict[str, Any]],
    offerings: str,
    *,
    competitor_domain: str,
    our_domain: str,
    api_key: str,
) -> dict[str, str]:
    """Recommend which keyword to target and how, as structured fields.

    `candidates` are steal-able keyword rows (competitor ranks, we don't or rank
    worse), each with Keyword / Volume / Difficulty / Competitor position / Our
    position. `offerings` describes our core products/services — the model is
    told to recommend ONLY a keyword that clearly relates to them, and to leave
    `keyword` empty (with a `no_fit_reason`) if none do.

    Returns a dict with keys: keyword, why, blog_title, blog_angle,
    no_fit_reason. Raises ClaudeError on API errors or a malformed response.
    """
    client = anthropic.Anthropic(api_key=api_key)

    lines = []
    for c in candidates:
        our_pos = c.get("Our position")
        our_str = f"#{our_pos}" if our_pos else "not ranking"
        lines.append(
            f"- \"{c.get('Keyword')}\" — volume {c.get('Volume')}, "
            f"difficulty {c.get('Difficulty')}, competitor ranks "
            f"#{c.get('Competitor position')}, we are {our_str}"
        )
    candidate_block = "\n".join(lines) if lines else "(none)"

    prompt = (
        f"You advise {our_domain} on SEO. Below are keywords the competitor "
        f"{competitor_domain} ranks for that {our_domain} does not rank for, or "
        "ranks worse for — i.e. keywords we could try to steal organic traffic "
        "for.\n\n"
        f"Our core offerings:\n{offerings.strip()}\n\n"
        f"Candidate keywords (easiest first):\n{candidate_block}\n\n"
        "Recommend the ONE keyword we should target. Strict rule: only "
        "recommend a keyword that clearly relates to our core offerings above — "
        "ignore keywords about unrelated topics, the competitor's own brand, or "
        "products we don't sell, no matter how easy they look. If none of the "
        "candidates genuinely relate to our offerings, leave the keyword empty "
        "and explain why in no_fit_reason.\n\n"
        "Keep every field tight and concrete — these render as short labeled "
        "sections, not a paragraph."
    )

    try:
        response = client.messages.create(
            model=_RECOMMEND_MODEL,
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {"type": "json_schema", "schema": _RECOMMEND_SCHEMA}
            },
        )
    except anthropic.AuthenticationError:
        raise ClaudeError(
            "Anthropic rejected the API key — check ANTHROPIC_API_KEY"
        ) from None
    except anthropic.RateLimitError:
        raise ClaudeError(
            "Anthropic rate limit hit — wait a minute and retry"
        ) from None
    except anthropic.APIError as e:
        raise ClaudeError(f"Anthropic API error: {e}") from None

    seo_usage.record_claude(_RECOMMEND_MODEL, response.usage)

    try:
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
    except (StopIteration, json.JSONDecodeError, AttributeError) as e:
        raise ClaudeError(f"could not parse Claude response: {e}") from None

    return {
        "keyword": (data.get("keyword") or "").strip(),
        "why": (data.get("why") or "").strip(),
        "blog_title": (data.get("blog_title") or "").strip(),
        "blog_angle": (data.get("blog_angle") or "").strip(),
        "no_fit_reason": (data.get("no_fit_reason") or "").strip(),
    }


# Homepages are long; cap what we send to keep the summary call cheap.
_OFFERINGS_MAX_CHARS = 8000


def summarize_offerings(domain: str, page_text: str, api_key: str) -> str:
    """Summarize a domain's core products/services from its homepage text.

    `page_text` is the visible text scraped from the domain's homepage (title +
    meta + body). Returns a 2-4 sentence description of what the company offers
    and who it serves — the grounding the keyword recommender filters against.
    Raises ClaudeError on API errors or an empty response.
    """
    client = anthropic.Anthropic(api_key=api_key)

    prompt = (
        f"Below is the homepage text for {domain}. Summarize the company's core "
        "products, services, and offerings, and who its customers are, in 2-4 "
        "plain sentences. Focus on what they actually sell — ignore navigation, "
        "boilerplate, and marketing fluff. Do not invent details not supported "
        "by the text.\n\n"
        f"Homepage text:\n{page_text[:_OFFERINGS_MAX_CHARS]}"
    )

    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError:
        raise ClaudeError(
            "Anthropic rejected the API key — check ANTHROPIC_API_KEY"
        ) from None
    except anthropic.RateLimitError:
        raise ClaudeError(
            "Anthropic rate limit hit — wait a minute and retry"
        ) from None
    except anthropic.APIError as e:
        raise ClaudeError(f"Anthropic API error: {e}") from None

    seo_usage.record_claude(_MODEL, response.usage)

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise ClaudeError("Claude returned an empty offerings summary")
    return text


# Reading a messy spreadsheet's layout is real reasoning (find the header row
# among title rows, spot monthly columns to sum, subtotal rows to skip), so the
# infrequent layout + reconcile calls both use the strongest model.
_LAYOUT_MODEL = "claude-opus-4-8"

_ONE_SHEET_SCHEMA = {
    "type": "object",
    "properties": {
        "header_row": {
            "type": "integer",
            "description": "0-based index of the row holding column headers.",
        },
        "first_data_row": {
            "type": "integer",
            "description": "0-based index of the FIRST real line-item row (usually header_row + 1).",
        },
        "last_data_row": {
            "type": "integer",
            "description": (
                "0-based index of the LAST real line-item row. Everything below "
                "it — TOTAL / quarterly / half-year / service-line summary rows "
                "— must be excluded. These summary rows often park values inside "
                "month columns, so getting this right is critical."
            ),
        },
        "category_columns": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "0-based column indices whose values name the budget line / "
                "category. Usually one; more when the category is split across "
                "columns (e.g. Group + Line item)."
            ),
        },
        "period_columns": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "When the sheet spreads amounts across time columns (Jan…Dec, "
                "or weeks/quarters), the 0-based indices of those PERIOD columns "
                "in chronological order. Do NOT include any Total, Average, or "
                "summary column here. Empty when the sheet isn't a time matrix."
            ),
        },
        "amount_columns": {
            "type": "array",
            "items": {"type": "integer"},
            "description": (
                "The money column(s) to sum when the sheet is NOT a time matrix "
                "— e.g. a single 'Amount' in a transaction feed, or a single "
                "annual 'Total'. Leave EMPTY when period_columns is populated "
                "(the periods are the amounts). Never mix period columns with a "
                "total column here, or it double-counts."
            ),
        },
        "exclude_patterns": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Case-insensitive substrings identifying any stray subtotal/"
                "total rows to drop by their category label (e.g. 'total', "
                "'subtotal'). Empty if there are none."
            ),
        },
        "note": {
            "type": "string",
            "description": "One short sentence on how this sheet is laid out.",
        },
    },
    "required": [
        "header_row", "first_data_row", "last_data_row", "category_columns",
        "period_columns", "amount_columns", "exclude_patterns", "note",
    ],
    "additionalProperties": False,
}

_LAYOUTS_SCHEMA = {
    "type": "object",
    "properties": {
        "planned": _ONE_SHEET_SCHEMA,
        "actual": _ONE_SHEET_SCHEMA,
    },
    "required": ["planned", "actual"],
    "additionalProperties": False,
}


def _coerce_sheet_layout(data: Any) -> dict[str, Any]:
    """Defensively normalize one sheet-layout object from Claude."""
    d = data if isinstance(data, dict) else {}

    def _int_list(v: Any) -> list[int]:
        out: list[int] = []
        for x in v or []:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out

    def _opt_int(v: Any) -> int | None:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    try:
        header_row = int(d.get("header_row") or 0)
    except (TypeError, ValueError):
        header_row = 0

    return {
        "header_row": max(0, header_row),
        "first_data_row": _opt_int(d.get("first_data_row")),
        "last_data_row": _opt_int(d.get("last_data_row")),
        "category_columns": _int_list(d.get("category_columns")),
        "period_columns": _int_list(d.get("period_columns")),
        "amount_columns": _int_list(d.get("amount_columns")),
        "exclude_patterns": [
            str(p).strip() for p in (d.get("exclude_patterns") or []) if str(p).strip()
        ],
        "note": (str(d.get("note") or "")).strip(),
    }


def infer_layouts(
    planned_preview: list[list[str]],
    actual_preview: list[list[str]],
    api_key: str,
) -> dict[str, dict[str, Any]]:
    """Read how two spreadsheets are laid out so we can roll them up correctly.

    Each preview is a list of rows (outer index = row, inner index = 0-based
    column) sampled from the top of the raw sheet. Claude locates the header row
    and the category/amount columns for each — coping with title rows, monthly
    columns that must be summed, and subtotal rows to skip.

    Returns {"planned": layout, "actual": layout}, each layout a dict with
    header_row (int), category_columns / amount_columns (list[int] of column
    indices), exclude_patterns (list[str]), note (str). Raises ClaudeError on
    API errors or a malformed response.
    """
    client = anthropic.Anthropic(api_key=api_key)

    prompt = (
        "Two spreadsheets were uploaded to reconcile a marketing budget: a "
        "PLANNED BUDGET and a feed of ACTUAL SPEND from finance. Each preview is "
        "an object {total_rows, rows}, where each row is [row_index, [cells…]] — "
        "row_index is the TRUE 0-based index in the full sheet and cells are "
        "0-based columns left to right. To keep the preview small you're shown "
        "the first rows and the last rows; middle rows are omitted but the "
        "indices are exact, so you can still point at rows near the bottom.\n\n"
        "For EACH sheet determine:\n"
        "- header_row: the row index of the column headers (skip any title/blank "
        "rows above it).\n"
        "- first_data_row / last_data_row: the row-index span of the REAL line "
        "items. Crucially, exclude the block of summary rows at the bottom — a "
        "TOTAL row, quarterly/half-year totals, a service-line breakdown. Those "
        "summary rows frequently place values inside month columns, so if you "
        "include them the monthly sums blow up (often to exactly double).\n"
        "- category_columns: the column(s) naming each line item.\n"
        "- period_columns: if amounts are spread across time columns (Jan…Dec, "
        "weeks, quarters), their indices in chronological order — NEVER include "
        "a Total/Average/summary column among them.\n"
        "- amount_columns: only for a NON-time-matrix sheet (a single Amount in "
        "a transaction feed, or a single annual Total). Leave empty when "
        "period_columns is set.\n"
        "- exclude_patterns: substrings for any stray total rows within the data "
        "span.\n\n"
        f"PLANNED BUDGET preview:\n{json.dumps(planned_preview)}\n\n"
        f"ACTUAL SPEND preview:\n{json.dumps(actual_preview)}"
    )

    try:
        response = client.messages.create(
            model=_LAYOUT_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {"type": "json_schema", "schema": _LAYOUTS_SCHEMA}
            },
        )
    except anthropic.AuthenticationError:
        raise ClaudeError(
            "Anthropic rejected the API key — check ANTHROPIC_API_KEY"
        ) from None
    except anthropic.RateLimitError:
        raise ClaudeError(
            "Anthropic rate limit hit — wait a minute and retry"
        ) from None
    except anthropic.APIError as e:
        raise ClaudeError(f"Anthropic API error: {e}") from None

    seo_usage.record_claude(_LAYOUT_MODEL, response.usage)

    try:
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
    except (StopIteration, json.JSONDecodeError, AttributeError) as e:
        raise ClaudeError(f"could not parse Claude response: {e}") from None

    return {
        "planned": _coerce_sheet_layout(data.get("planned")),
        "actual": _coerce_sheet_layout(data.get("actual")),
    }


# Reconciliation is judgment-heavy (semantic category matching + prose flags),
# and the call is infrequent on a small payload, so use the strongest model.
_RECONCILE_MODEL = "claude-opus-4-8"

_RECONCILE_SCHEMA = {
    "type": "object",
    "properties": {
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Canonical category name for this reconciled line.",
                    },
                    "planned": {
                        "type": "number",
                        "description": "Planned budget for this category (0 if unbudgeted).",
                    },
                    "actual": {
                        "type": "number",
                        "description": "Actual spend for this category (0 if nothing spent).",
                    },
                    "variance": {
                        "type": "number",
                        "description": "actual - planned. Positive = overspend, negative = underspend.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["over", "under", "on_track", "unbudgeted", "unspent"],
                        "description": (
                            "over/under/on_track for budgeted categories; "
                            "unbudgeted = spend with no matching budget line; "
                            "unspent = budget line with no spend."
                        ),
                    },
                    "note": {
                        "type": "string",
                        "description": "Short flag/explanation, or empty when unremarkable.",
                    },
                },
                "required": ["category", "planned", "actual", "variance", "status", "note"],
                "additionalProperties": False,
            },
        },
        "summary": {
            "type": "string",
            "description": "2-4 sentences on overall over/under-spend against plan.",
        },
        "flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Notable inconsistencies to call out (biggest/most concerning first).",
        },
    },
    "required": ["categories", "summary", "flags"],
    "additionalProperties": False,
}


def reconcile_budget(
    planned: dict[str, float],
    actual: dict[str, float],
    api_key: str,
    *,
    currency: str = "USD",
) -> dict[str, Any]:
    """Reconcile a planned-budget rollup against an actual-spend rollup.

    `planned` and `actual` are {category: total} maps produced by
    seo_budget.rollup — already summed per category, so Claude only has to match
    categories across the two sheets (their names rarely align — "Google Ads" vs
    "Paid Search — Google"), compute per-category variance, classify over/under/
    unbudgeted/unspent, and surface inconsistencies.

    Returns {"categories": [...], "summary": str, "flags": [str]} — see
    _RECONCILE_SCHEMA. Raises ClaudeError on API errors or a malformed response.
    """
    client = anthropic.Anthropic(api_key=api_key)

    prompt = (
        "You are reconciling a marketing PLANNED BUDGET against ACTUAL SPEND "
        "pulled from finance. Each is a JSON map of category -> total (already "
        f"summed; amounts in {currency}).\n\n"
        f"PLANNED BUDGET:\n{json.dumps(planned, indent=2)}\n\n"
        f"ACTUAL SPEND:\n{json.dumps(actual, indent=2)}\n\n"
        "Produce a reconciliation:\n"
        "- Match categories across the two maps by MEANING, not exact string — "
        "e.g. 'Google Ads' in one and 'Paid Search — Google' in the other are "
        "the same line; merge them. Keep genuinely distinct categories separate.\n"
        "- For each reconciled category emit planned, actual, and "
        "variance = actual - planned (positive = OVERSPEND, negative = underspend).\n"
        "- status: 'over' if actual materially exceeds planned, 'under' if "
        "materially below, 'on_track' if close; 'unbudgeted' if there's spend "
        "but no matching budget line (planned = 0); 'unspent' if there's a "
        "budget line but no spend (actual = 0).\n"
        "- note: a short flag when useful (e.g. 'over by 38%'), else empty.\n"
        "- Include EVERY category that appears in either map exactly once.\n"
        "- summary: 2-4 sentences on whether we're over or under overall and the "
        "biggest drivers.\n"
        "- flags: the most concerning inconsistencies first — large overspends, "
        "spend in unbudgeted categories, budgeted lines with zero spend. Empty "
        "list if nothing stands out.\n"
        "Report amounts as plain numbers (no currency symbols or commas)."
    )

    try:
        response = client.messages.create(
            model=_RECONCILE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {"type": "json_schema", "schema": _RECONCILE_SCHEMA}
            },
        )
    except anthropic.AuthenticationError:
        raise ClaudeError(
            "Anthropic rejected the API key — check ANTHROPIC_API_KEY"
        ) from None
    except anthropic.RateLimitError:
        raise ClaudeError(
            "Anthropic rate limit hit — wait a minute and retry"
        ) from None
    except anthropic.APIError as e:
        raise ClaudeError(f"Anthropic API error: {e}") from None

    seo_usage.record_claude(_RECONCILE_MODEL, response.usage)

    try:
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
    except (StopIteration, json.JSONDecodeError, AttributeError) as e:
        raise ClaudeError(f"could not parse Claude response: {e}") from None

    def _num(v: Any) -> float:
        try:
            return round(float(v), 2)
        except (TypeError, ValueError):
            return 0.0

    categories = []
    for c in data.get("categories") or []:
        if not isinstance(c, dict):
            continue
        planned_v = _num(c.get("planned"))
        actual_v = _num(c.get("actual"))
        categories.append(
            {
                "category": (str(c.get("category") or "")).strip() or "(unnamed)",
                "planned": planned_v,
                "actual": actual_v,
                # Trust our own arithmetic for variance over the model's.
                "variance": round(actual_v - planned_v, 2),
                "status": (str(c.get("status") or "")).strip() or "on_track",
                "note": (str(c.get("note") or "")).strip(),
            }
        )

    flags = [str(f).strip() for f in (data.get("flags") or []) if str(f).strip()]

    return {
        "categories": categories,
        "summary": (str(data.get("summary") or "")).strip(),
        "flags": flags,
    }
