"""Tool-using question-answering agent for the Magic Eight Ball page.

Claude is given a small toolbox wired to the app's existing data sources
(Apollo firmographics/contacts/company-search, Semrush SEO/keyword data). It
picks the relevant tool(s), we execute them with the app's API keys, feed the
results back, and Claude returns a brief answer grounded in what the tools
returned.

Adding a new data source (e.g. HubSpot) is a matter of writing a handler +
schema and registering it in _build_toolbox — the loop and the view don't change.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import anthropic

from seo_apollo import (
    organization_enrich,
    organization_search,
    people_search,
)
from seo_claude import ClaudeError
from seo_hubspot import (
    aggregate_deals,
    campaign_metrics,
    list_campaigns,
    list_properties,
    search_objects,
    traffic_sources,
)
from seo_keys import normalize_domain
from seo_semrush import (
    domain_organic_keywords,
    domain_overview,
    top_pages,
)

_AGENT_MODEL = "claude-opus-4-8"
_MAX_STEPS = 6  # tool-call rounds before we give up
_MAX_ROWS = 25  # cap list results so tool output stays compact

_SYSTEM = (
    "You are the Magic Eight Ball inside an internal marketing app. Answer the "
    "user's question by retrieving REAL data with the available tools:\n"
    "- Apollo tools: an EXTERNAL prospecting database — company firmographics "
    "(size, revenue, industry, tech), contacts/people at any company, and "
    "company search by criteria. Use for prospecting / market questions about "
    "companies we may not work with.\n"
    "- HubSpot tools: OUR OWN CRM and marketing — our contacts, companies, and "
    "deals, marketing campaign analytics, and our WEBSITE TRAFFIC by source "
    "(organic, paid, direct, etc.) for any date range. Use when the question is "
    "about 'our' records, pipeline/deals, customers, campaigns, or our site's "
    "traffic. For year-over-year or YTD traffic, call the traffic tool twice "
    "(this period and the matching period last year) and compare.\n"
    "- Semrush tools: SEO data — organic traffic/keywords, keyword difficulty, "
    "and top pages for a domain.\n\n"
    "Pick the right source: 'our deals/pipeline/customers/campaigns' → HubSpot; "
    "'find/look up a company or its people in the market' → Apollo; 'SEO / "
    "traffic / keywords' → Semrush. Call the tool(s), and base your answer ONLY "
    "on what they return — never invent numbers or facts. Keep the final answer "
    "brief: 2-4 plain sentences, leading with the direct answer. If the "
    "available tools can't answer the question, say so in one sentence and name "
    "the kind of data source that would be needed."
)


# ── Result distillation (keep tool output small) ─────────────────────────────


def _distill_enrichment(org: dict[str, Any]) -> dict[str, Any]:
    techs = org.get("technology_names") or org.get("current_technologies") or []
    tech_names = [t for t in techs if isinstance(t, str)][:25]
    loc = ", ".join(
        p for p in [org.get("city"), org.get("state"), org.get("country")] if p
    )
    return {
        "name": org.get("name"),
        "domain": org.get("primary_domain") or org.get("website_url"),
        "industry": org.get("industry"),
        "employees": org.get("estimated_num_employees"),
        "annual_revenue": org.get("annual_revenue") or org.get("annual_revenue_printed"),
        "total_funding": org.get("total_funding_printed") or org.get("total_funding"),
        "latest_funding_stage": org.get("latest_funding_stage"),
        "founded_year": org.get("founded_year"),
        "hq": loc or None,
        "description": (org.get("short_description") or "")[:500] or None,
        "technologies": tech_names,
    }


def _distill_search_company(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": c.get("name"),
        "domain": c.get("primary_domain") or c.get("domain"),
        "employees": c.get("estimated_num_employees"),
        "revenue": c.get("organization_revenue_printed")
        or c.get("organization_revenue"),
        "industry": c.get("industry"),
    }


def _distill_person(p: dict[str, Any]) -> dict[str, Any]:
    org = p.get("organization") or {}
    return {
        "name": (f"{p.get('first_name') or ''} {p.get('last_name') or ''}").strip()
        or p.get("name"),
        "title": p.get("title"),
        "seniority": p.get("seniority"),
        "company": org.get("name"),
        "linkedin": p.get("linkedin_url"),
    }


# ── Tool handlers ────────────────────────────────────────────────────────────
# Each handler takes the parsed tool input plus the relevant API key(s) and
# returns a JSON-serializable result (or raises; the loop turns that into an
# error tool_result so Claude can adapt).


def _apollo_company_enrich(args: dict[str, Any], *, apollo_key: str) -> Any:
    domain = normalize_domain(args["domain"])
    org = organization_enrich(domain, apollo_key)
    if not org:
        return {"found": False, "domain": domain}
    return {"found": True, **_distill_enrichment(org)}


def _apollo_company_search(args: dict[str, Any], *, apollo_key: str) -> Any:
    location = args.get("location")
    result = organization_search(
        apollo_key,
        keywords=args.get("keywords") or None,
        employees_min=args.get("employees_min"),
        employees_max=args.get("employees_max"),
        revenue_min=args.get("revenue_min"),
        revenue_max=args.get("revenue_max"),
        locations=[location] if location else None,
        per_page=min(int(args.get("limit", _MAX_ROWS)), _MAX_ROWS),
    )
    companies = (result.get("organizations") or []) + (result.get("accounts") or [])
    pagination = result.get("pagination") or {}
    return {
        "total_matches": pagination.get("total_entries", len(companies)),
        "companies": [_distill_search_company(c) for c in companies[:_MAX_ROWS]],
    }


def _apollo_people_search(args: dict[str, Any], *, apollo_key: str) -> Any:
    domain = normalize_domain(args["domain"])
    result = people_search(
        domain,
        apollo_key,
        titles=args.get("titles") or None,
        seniorities=args.get("seniorities") or None,
        per_page=min(int(args.get("limit", _MAX_ROWS)), _MAX_ROWS),
    )
    people = result.get("people") or []
    pagination = result.get("pagination") or {}
    return {
        "total_contacts": pagination.get("total_entries", len(people)),
        "people": [_distill_person(p) for p in people[:_MAX_ROWS]],
        "note": "Emails are not included (revealing them costs Apollo credits).",
    }


def _semrush_domain_overview(args: dict[str, Any], *, semrush_key: str) -> Any:
    domain = normalize_domain(args["domain"])
    db = args.get("database", "us")
    row = domain_overview(domain, db, semrush_key)
    if not row:
        return {"found": False, "domain": domain, "database": db}
    return {
        "found": True,
        "domain": domain,
        "database": db,
        "rank": row.get("Rk"),
        "organic_keywords": row.get("Or"),
        "organic_traffic": row.get("Ot"),
        "organic_traffic_value_usd": row.get("Oc"),
    }


def _semrush_organic_keywords(args: dict[str, Any], *, semrush_key: str) -> Any:
    domain = normalize_domain(args["domain"])
    db = args.get("database", "us")
    limit = min(int(args.get("limit", _MAX_ROWS)), _MAX_ROWS)
    rows = domain_organic_keywords(domain, db, semrush_key, limit=limit)
    return {
        "domain": domain,
        "database": db,
        "keywords": [
            {
                "keyword": r.get("Ph"),
                "position": r.get("Po"),
                "volume": r.get("Nq"),
                "difficulty": r.get("Kd"),
                "url": r.get("Ur"),
            }
            for r in rows[:limit]
        ],
    }


def _semrush_top_pages(args: dict[str, Any], *, semrush_key: str) -> Any:
    domain = normalize_domain(args["domain"])
    db = args.get("database", "us")
    limit = min(int(args.get("limit", _MAX_ROWS)), _MAX_ROWS)
    rows = top_pages(domain, db, semrush_key, limit=limit)
    return {
        "domain": domain,
        "database": db,
        "top_pages": [
            {
                "url": r.get("Ur"),
                "keywords": r.get("Keywords"),
                "traffic_share_pct": round(float(r.get("TrafficShare", 0.0)), 2),
            }
            for r in rows[:limit]
        ],
    }


def _hubspot_search(
    object_type: str, args: dict[str, Any], *, hubspot_token: str
) -> Any:
    results = search_objects(
        object_type,
        args.get("query", ""),
        hubspot_token,
        limit=min(int(args.get("limit", _MAX_ROWS)), _MAX_ROWS),
    )
    return {"object": object_type, "count": len(results), "results": results}


def _date_to_ms(d: str, *, end_of_day: bool = False) -> int:
    """Parse a YYYY-MM-DD date into epoch milliseconds (UTC)."""
    dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        dt = dt + timedelta(days=1) - timedelta(milliseconds=1)
    return int(dt.timestamp() * 1000)


def _hubspot_deals_report(args: dict[str, Any], *, hubspot_token: str) -> Any:
    only_closed_won = bool(args.get("only_closed_won"))
    # When entering closed-won is the question, hs_closed_won_date is the right
    # field; otherwise default to closedate.
    date_property = args.get("date_property") or (
        "hs_closed_won_date" if only_closed_won else "closedate"
    )

    start_ms = end_ms = None
    since_days = args.get("since_days")
    if since_days is not None:
        now = datetime.now(timezone.utc)
        start_ms = int((now - timedelta(days=int(since_days))).timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)
    else:
        if args.get("start_date"):
            start_ms = _date_to_ms(args["start_date"])
        if args.get("end_date"):
            end_ms = _date_to_ms(args["end_date"], end_of_day=True)

    # Translate the agent's simplified property filters into HubSpot's format.
    extra_filters = []
    for f in args.get("property_filters") or []:
        prop = f.get("property")
        if not prop:
            continue
        hf: dict[str, Any] = {"propertyName": prop, "operator": (f.get("operator") or "EQ").upper()}
        if f.get("values") is not None:
            hf["values"] = f["values"]
        elif f.get("value") is not None:
            hf["value"] = str(f["value"])
        extra_filters.append(hf)

    has_date_filter = start_ms is not None or end_ms is not None
    result = aggregate_deals(
        hubspot_token,
        only_closed_won=only_closed_won,
        dealstage=args.get("dealstage"),
        pipeline=args.get("pipeline"),
        date_property=date_property if has_date_filter else None,
        start_ms=start_ms,
        end_ms=end_ms,
        extra_filters=extra_filters or None,
    )
    result["filter"] = {
        "only_closed_won": only_closed_won,
        "date_property": date_property if has_date_filter else None,
        "since_days": since_days,
        "start_date": args.get("start_date"),
        "end_date": args.get("end_date"),
    }
    result["note"] = (
        "total_amount sums the `amount` field across matching deals (portal "
        "default currency). If truncated is true, total_amount covers only the "
        "deals fetched, not all of them."
    )
    return result


def _hubspot_deal_properties(args: dict[str, Any], *, hubspot_token: str) -> Any:
    props = list_properties(
        "deals",
        hubspot_token,
        search=args.get("search"),
        limit=min(int(args.get("limit", 60)), 100),
    )
    return {"count": len(props), "properties": props}


def _hubspot_traffic(args: dict[str, Any], *, hubspot_token: str) -> Any:
    start = datetime.strptime(args["start_date"], "%Y-%m-%d").strftime("%Y%m%d")
    end = datetime.strptime(args["end_date"], "%Y-%m-%d").strftime("%Y%m%d")
    payload = traffic_sources(hubspot_token, start=start, end=end)

    sources = []
    organic_visits = 0
    for b in payload.get("breakdowns") or []:
        name = b.get("breakdown")
        visits = b.get("visits")
        sources.append(
            {"source": name, "visits": visits, "contacts": b.get("contacts")}
        )
        if name and "organic" in str(name).lower():
            try:
                organic_visits += int(visits or 0)
            except (TypeError, ValueError):
                pass

    totals = payload.get("totals") or {}
    return {
        "start_date": args["start_date"],
        "end_date": args["end_date"],
        "organic_visits": organic_visits,
        "total_visits": totals.get("visits"),
        "sources": sources,
        "note": "'visits' is HubSpot's term for sessions.",
    }


def _hubspot_list_campaigns(args: dict[str, Any], *, hubspot_token: str) -> Any:
    campaigns = list_campaigns(
        hubspot_token,
        name_contains=args.get("name_contains"),
        limit=min(int(args.get("limit", _MAX_ROWS)), _MAX_ROWS),
    )
    return {"count": len(campaigns), "campaigns": campaigns}


def _hubspot_campaign_metrics(args: dict[str, Any], *, hubspot_token: str) -> Any:
    return campaign_metrics(
        hubspot_token,
        args["campaign_id"],
        start_date=args.get("start_date"),
        end_date=args.get("end_date"),
    )


# ── Toolbox assembly ─────────────────────────────────────────────────────────

_DOMAIN_PROP = {"type": "string", "description": "Company domain, e.g. stripe.com"}
_DB_PROP = {
    "type": "string",
    "description": "Semrush regional database (us, uk, ca, au, de, fr, es, it, br, in). Defaults to us.",
}


def _build_toolbox(
    apollo_key: str | None,
    semrush_key: str | None,
    hubspot_token: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Callable[[dict[str, Any]], Any]]]:
    """Return (tool schemas, name->handler) for the data sources whose keys are
    set, so Claude is only offered tools it can actually run."""
    tools: list[dict[str, Any]] = []
    handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}

    if apollo_key:
        tools += [
            {
                "name": "apollo_company_enrich",
                "description": (
                    "Look up one company by domain and return firmographics: "
                    "size, revenue, industry, funding, HQ, and detected "
                    "technologies. Use for questions about a specific company."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"domain": _DOMAIN_PROP},
                    "required": ["domain"],
                },
            },
            {
                "name": "apollo_company_search",
                "description": (
                    "Find companies matching criteria (free-text keywords for "
                    "industry/category, employee range, revenue range in USD, HQ "
                    "location). Use for 'find/list companies that…' questions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "string",
                            "description": "Industry / category / technology terms, e.g. 'B2B SaaS payments'.",
                        },
                        "employees_min": {"type": "integer"},
                        "employees_max": {"type": "integer"},
                        "revenue_min": {"type": "integer", "description": "Min annual revenue in USD."},
                        "revenue_max": {"type": "integer", "description": "Max annual revenue in USD."},
                        "location": {"type": "string", "description": "HQ country, state, or city."},
                        "limit": {"type": "integer", "description": "Max companies to return (<=25)."},
                    },
                    "required": [],
                },
            },
            {
                "name": "apollo_people_search",
                "description": (
                    "Find people/contacts at a company domain, optionally "
                    "filtered by job titles or seniority. Use for 'who works at…' "
                    "or 'find the head of X at…' questions. Emails not included."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "domain": _DOMAIN_PROP,
                        "titles": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Job-title variants to match, e.g. ['VP Marketing','CMO'].",
                        },
                        "seniorities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Apollo seniorities, e.g. ['c_suite','vp','director'].",
                        },
                        "limit": {"type": "integer", "description": "Max people to return (<=25)."},
                    },
                    "required": ["domain"],
                },
            },
        ]
        handlers.update(
            {
                "apollo_company_enrich": lambda a: _apollo_company_enrich(a, apollo_key=apollo_key),
                "apollo_company_search": lambda a: _apollo_company_search(a, apollo_key=apollo_key),
                "apollo_people_search": lambda a: _apollo_people_search(a, apollo_key=apollo_key),
            }
        )

    if semrush_key:
        tools += [
            {
                "name": "semrush_domain_overview",
                "description": (
                    "SEO snapshot for a domain: organic keyword count, estimated "
                    "monthly organic traffic, and its estimated USD value. Use "
                    "for 'how much traffic does X get' questions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"domain": _DOMAIN_PROP, "database": _DB_PROP},
                    "required": ["domain"],
                },
            },
            {
                "name": "semrush_organic_keywords",
                "description": (
                    "Top organic keywords a domain ranks for, with position, "
                    "search volume, and keyword difficulty. Use for 'what "
                    "keywords does X rank for' questions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "domain": _DOMAIN_PROP,
                        "database": _DB_PROP,
                        "limit": {"type": "integer", "description": "Max keywords (<=25)."},
                    },
                    "required": ["domain"],
                },
            },
            {
                "name": "semrush_top_pages",
                "description": (
                    "A domain's top pages by share of organic traffic. Use for "
                    "'which pages drive X's traffic' questions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "domain": _DOMAIN_PROP,
                        "database": _DB_PROP,
                        "limit": {"type": "integer", "description": "Max pages (<=25)."},
                    },
                    "required": ["domain"],
                },
            },
        ]
        handlers.update(
            {
                "semrush_domain_overview": lambda a: _semrush_domain_overview(a, semrush_key=semrush_key),
                "semrush_organic_keywords": lambda a: _semrush_organic_keywords(a, semrush_key=semrush_key),
                "semrush_top_pages": lambda a: _semrush_top_pages(a, semrush_key=semrush_key),
            }
        )

    if hubspot_token:
        _hs_query = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text search (name, email, domain, etc.).",
                },
                "limit": {"type": "integer", "description": "Max records (<=25)."},
            },
            "required": ["query"],
        }
        tools += [
            {
                "name": "hubspot_search_contacts",
                "description": (
                    "Search OUR HubSpot CRM contacts (name, email, title, "
                    "company, lifecycle stage). Use for questions about our "
                    "contacts/leads/customers."
                ),
                "input_schema": _hs_query,
            },
            {
                "name": "hubspot_search_companies",
                "description": (
                    "Search OUR HubSpot CRM companies (name, domain, industry, "
                    "size, revenue, lifecycle stage). Use for questions about "
                    "companies in our CRM."
                ),
                "input_schema": _hs_query,
            },
            {
                "name": "hubspot_search_deals",
                "description": (
                    "Full-text search OUR HubSpot CRM deals by name/keyword "
                    "(e.g. find deals for a specific company). For COUNTS, "
                    "REVENUE TOTALS, or DATE/STAGE-filtered questions, use "
                    "hubspot_deals_report instead."
                ),
                "input_schema": _hs_query,
            },
            {
                "name": "hubspot_deals_report",
                "description": (
                    "Count deals and sum their revenue (the `amount` field) with "
                    "structured filters — the right tool for questions like 'how "
                    "many closed-won deals in the last 365 days and total "
                    "revenue'. Returns count, total_amount, and a sample. "
                    "Set only_closed_won=true for won deals; date filtering uses "
                    "hs_closed_won_date (when a deal ENTERED closed-won) by "
                    "default for closed-won queries, or closedate otherwise — "
                    "override with date_property if needed."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "only_closed_won": {
                            "type": "boolean",
                            "description": "Filter to deals currently in a closed-won stage.",
                        },
                        "since_days": {
                            "type": "integer",
                            "description": "Restrict to deals whose date_property is within the last N days (e.g. 365).",
                        },
                        "start_date": {"type": "string", "description": "YYYY-MM-DD lower bound (alternative to since_days)."},
                        "end_date": {"type": "string", "description": "YYYY-MM-DD upper bound."},
                        "date_property": {
                            "type": "string",
                            "description": "Which date field to filter on: hs_closed_won_date, closedate, or createdate.",
                        },
                        "dealstage": {"type": "string", "description": "Filter to a specific deal stage id (optional)."},
                        "pipeline": {"type": "string", "description": "Filter to a specific pipeline id (optional)."},
                        "property_filters": {
                            "type": "array",
                            "description": (
                                "Filter on ANY deal property (e.g. deal source / "
                                "lead source = Inbound, region, type). Discover "
                                "the exact property name and valid values first "
                                "with hubspot_deal_properties."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "property": {"type": "string", "description": "Internal property name, e.g. deal_source."},
                                    "operator": {
                                        "type": "string",
                                        "description": "EQ, NEQ, IN, NOT_IN, CONTAINS_TOKEN, HAS_PROPERTY, NOT_HAS_PROPERTY. Default EQ.",
                                    },
                                    "value": {"type": "string", "description": "Value for EQ/NEQ/CONTAINS_TOKEN."},
                                    "values": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Values for IN/NOT_IN.",
                                    },
                                },
                                "required": ["property"],
                            },
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "hubspot_deal_properties",
                "description": (
                    "List deal properties (name, label, type, and allowed "
                    "values for dropdowns). Use this to find the right field and "
                    "value before filtering — e.g. search 'source' or 'lead' to "
                    "find how inbound/outbound is recorded, then pass it to "
                    "hubspot_deals_report's property_filters."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "search": {"type": "string", "description": "Only return properties whose name/label contains this text."},
                        "limit": {"type": "integer", "description": "Max properties (<=100)."},
                    },
                    "required": [],
                },
            },
            {
                "name": "hubspot_website_traffic",
                "description": (
                    "Our website's traffic (sessions, called 'visits') broken "
                    "down by source — organic, paid, direct, referral, social, "
                    "email — for a date range. Use for questions about our site's "
                    "organic/overall traffic. For YTD-vs-last-year, call this "
                    "twice with the two date ranges and compare the results."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "YYYY-MM-DD (inclusive)."},
                        "end_date": {"type": "string", "description": "YYYY-MM-DD (inclusive)."},
                    },
                    "required": ["start_date", "end_date"],
                },
            },
            {
                "name": "hubspot_list_campaigns",
                "description": (
                    "List our HubSpot marketing campaigns (id + name), "
                    "optionally filtered by a name substring. Use this first to "
                    "find a campaign's id before fetching its metrics."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name_contains": {
                            "type": "string",
                            "description": "Only return campaigns whose name contains this text.",
                        },
                        "limit": {"type": "integer", "description": "Max campaigns (<=25)."},
                    },
                    "required": [],
                },
            },
            {
                "name": "hubspot_campaign_metrics",
                "description": (
                    "Attribution metrics for one campaign (sessions, new & "
                    "influenced contacts, etc.) over a date range. Get the "
                    "campaign_id from hubspot_list_campaigns first."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "campaign_id": {"type": "string", "description": "Campaign id (GUID)."},
                        "start_date": {"type": "string", "description": "YYYY-MM-DD. Defaults to 90 days ago."},
                        "end_date": {"type": "string", "description": "YYYY-MM-DD. Defaults to today."},
                    },
                    "required": ["campaign_id"],
                },
            },
        ]
        handlers.update(
            {
                "hubspot_search_contacts": lambda a: _hubspot_search("contacts", a, hubspot_token=hubspot_token),
                "hubspot_search_companies": lambda a: _hubspot_search("companies", a, hubspot_token=hubspot_token),
                "hubspot_search_deals": lambda a: _hubspot_search("deals", a, hubspot_token=hubspot_token),
                "hubspot_deals_report": lambda a: _hubspot_deals_report(a, hubspot_token=hubspot_token),
                "hubspot_deal_properties": lambda a: _hubspot_deal_properties(a, hubspot_token=hubspot_token),
                "hubspot_website_traffic": lambda a: _hubspot_traffic(a, hubspot_token=hubspot_token),
                "hubspot_list_campaigns": lambda a: _hubspot_list_campaigns(a, hubspot_token=hubspot_token),
                "hubspot_campaign_metrics": lambda a: _hubspot_campaign_metrics(a, hubspot_token=hubspot_token),
            }
        )

    return tools, handlers


def answer_question(
    question: str,
    *,
    anthropic_key: str,
    apollo_key: str | None = None,
    semrush_key: str | None = None,
    hubspot_token: str | None = None,
) -> dict[str, Any]:
    """Answer a free-text question by letting Claude call the app's data tools.

    Returns {"answer": str, "tools_used": [{"name", "input"}, ...]}. Raises
    ClaudeError on Anthropic API errors. Individual tool failures are fed back
    to the model as error results rather than raised, so it can adapt.
    """
    client = anthropic.Anthropic(api_key=anthropic_key)
    tools, handlers = _build_toolbox(apollo_key, semrush_key, hubspot_token)

    # Give the model today's date so it can resolve relative ranges (YTD, last
    # year, last 30 days) itself.
    system = f"{_SYSTEM}\n\nToday's date is {datetime.now().date().isoformat()}."

    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    tools_used: list[dict[str, Any]] = []

    for _ in range(_MAX_STEPS):
        try:
            response = client.messages.create(
                model=_AGENT_MODEL,
                max_tokens=2048,
                system=system,
                tools=tools,
                messages=messages,
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

        if response.stop_reason != "tool_use":
            answer = "".join(
                b.text for b in response.content if b.type == "text"
            ).strip()
            return {
                "answer": answer or "(no answer produced)",
                "tools_used": tools_used,
            }

        # Execute the requested tools and feed results back.
        messages.append({"role": "assistant", "content": response.content})
        results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tools_used.append({"name": block.name, "input": block.input})
            handler = handlers.get(block.name)
            if handler is None:
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: unknown tool {block.name}",
                        "is_error": True,
                    }
                )
                continue
            try:
                output = handler(block.input)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(output, default=str),
                    }
                )
            except Exception as e:  # surface tool errors to the model, don't crash
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error running {block.name}: {e}",
                        "is_error": True,
                    }
                )
        messages.append({"role": "user", "content": results})

    return {
        "answer": (
            "I couldn't reach a final answer within the step limit. Try a more "
            "specific question."
        ),
        "tools_used": tools_used,
    }
