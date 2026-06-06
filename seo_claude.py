"""Thin Claude/Anthropic helper for SEO tool features.

One function so far: turn a free-form job function (e.g. "marketing leadership")
into a concrete list of job titles to pass to Apollo's fuzzy-matched
person_titles[] filter.
"""

from __future__ import annotations

import json

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
