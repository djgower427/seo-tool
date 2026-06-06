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
