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
from seo_google import (
    GoogleError,
    ads_search,
    gsc_list_sites,
    gsc_search_analytics,
)
from seo_hubspot import (
    aggregate_deals,
    batch_read_objects,
    campaign_metrics,
    get_associations_batch,
    list_campaigns,
    list_properties,
    search_objects,
    traffic_sources,
)
from seo_keys import normalize_domain
import seo_usage
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
    "(this period and the matching period last year) and compare. To get the "
    "companies or contacts ASSOCIATED with deals (or any CRM records), call "
    "hubspot_get_associations with the record ids — e.g. take the deal ids from "
    "hubspot_deals_report and resolve their associated companies/contacts.\n"
    "- Google Search Console tools: OUR site's organic Google Search performance "
    "— clicks, impressions, CTR, average position, and top queries/pages, for a "
    "date range.\n"
    "- Google Ads tools: OUR paid search performance — campaign / ad group / "
    "keyword impressions, clicks, cost, and conversions for a date range.\n"
    "- Semrush tools: THIRD-PARTY SEO estimates — organic traffic/keywords, "
    "keyword difficulty, and top pages for any domain.\n\n"
    "Pick the right source: 'our deals/pipeline/customers/campaigns' → HubSpot; "
    "'our organic Google clicks/impressions/queries' → Google Search Console; "
    "'our ad spend/paid performance' → Google Ads; 'find/look up a company or "
    "its people in the market' → Apollo; third-party 'SEO estimate / keyword "
    "difficulty / a competitor's traffic' → Semrush. Call the tool(s), and base "
    "your answer ONLY "
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


_HS_OBJECT_TYPES = {"deals", "companies", "contacts"}


def _hubspot_associations(args: dict[str, Any], *, hubspot_token: str) -> Any:
    """Resolve the records associated with a set of CRM records (e.g. each
    deal's associated companies/contacts), returning their details."""
    from_type = (args.get("from_object_type") or "deals").lower()
    to_type = (args.get("to_object_type") or "").lower()
    if from_type not in _HS_OBJECT_TYPES or to_type not in _HS_OBJECT_TYPES:
        return {
            "error": f"object types must be one of {sorted(_HS_OBJECT_TYPES)}",
        }
    ids = [str(i).strip() for i in (args.get("object_ids") or []) if str(i).strip()]
    if not ids:
        return {"error": "object_ids is required (e.g. deal ids from hubspot_deals_report)"}
    ids = ids[:_MAX_ROWS]
    per = min(int(args.get("limit", 10)), _MAX_ROWS)

    assoc = get_associations_batch(from_type, to_type, ids, hubspot_token)

    # Batch-read the details of every associated record once, then map back.
    wanted: list[str] = []
    for tids in assoc.values():
        wanted.extend(tids[:per])
    details = batch_read_objects(to_type, list(dict.fromkeys(wanted)), hubspot_token)

    out = []
    for fid in ids:
        tids = assoc.get(fid, [])[:per]
        out.append(
            {
                "from_id": fid,
                to_type: [details.get(t, {"id": t}) for t in tids],
            }
        )
    return {
        "from_object_type": from_type,
        "to_object_type": to_type,
        "associations": out,
        "note": (
            "Empty lists mean no associations of that type are recorded on the "
            "record (or the token lacks read scope for it)."
        ),
    }


def _gsc_search(
    args: dict[str, Any], *, google_oauth: dict[str, str], default_site: str | None
) -> Any:
    site = args.get("site_url") or default_site
    if not site:
        raise GoogleError(
            "No Search Console site given and no default GSC_SITE_URL is set — "
            "call gsc_list_sites to find the property URL, then pass site_url."
        )
    dims = args.get("dimensions")
    if dims is None:
        dims = ["query"]
    data = gsc_search_analytics(
        google_oauth,
        site,
        args["start_date"],
        args["end_date"],
        dimensions=dims,
        row_limit=min(int(args.get("row_limit", 25)), 100),
    )
    data["site_url"] = site
    return data


def _gsc_sites(args: dict[str, Any], *, google_oauth: dict[str, str]) -> Any:
    return {"sites": gsc_list_sites(google_oauth)}


def _distill_ads_row(r: dict[str, Any]) -> dict[str, Any]:
    camp = r.get("campaign") or {}
    ag = r.get("adGroup") or {}
    kw = (r.get("adGroupCriterion") or {}).get("keyword") or {}
    m = r.get("metrics") or {}

    def _i(v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    def _f(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    row: dict[str, Any] = {
        "campaign": camp.get("name"),
        "impressions": _i(m.get("impressions")),
        "clicks": _i(m.get("clicks")),
        "cost": round(_i(m.get("costMicros")) / 1_000_000, 2),
        "conversions": round(_f(m.get("conversions")), 2),
        "ctr": round(_f(m.get("ctr")), 4),
        "avg_cpc": round(_i(m.get("averageCpc")) / 1_000_000, 2),
    }
    if ag.get("name"):
        row["ad_group"] = ag.get("name")
    if kw.get("text"):
        row["keyword"] = kw.get("text")
    return row


_ADS_LEVELS = {
    "campaign": ("campaign", "campaign.name, campaign.status"),
    "ad_group": ("ad_group", "campaign.name, ad_group.name, ad_group.status"),
    "keyword": ("keyword_view", "campaign.name, ad_group.name, ad_group_criterion.keyword.text"),
}


def _ads_report(
    args: dict[str, Any], *, google_oauth: dict[str, str], ads_config: dict[str, str]
) -> Any:
    level = (args.get("level") or "campaign").lower()
    resource, select_dims = _ADS_LEVELS.get(level, _ADS_LEVELS["campaign"])

    since_days = args.get("since_days")
    if since_days is not None:
        end = datetime.now(timezone.utc).date()
        start_s = (end - timedelta(days=int(since_days))).isoformat()
        end_s = end.isoformat()
    else:
        start_s, end_s = args.get("start_date"), args.get("end_date")
    if not start_s or not end_s:
        raise GoogleError("Provide either since_days, or both start_date and end_date.")

    limit = min(int(args.get("limit", 25)), 100)
    metrics = (
        "metrics.impressions, metrics.clicks, metrics.cost_micros, "
        "metrics.conversions, metrics.ctr, metrics.average_cpc"
    )
    query = (
        f"SELECT {select_dims}, {metrics} FROM {resource} "
        f"WHERE segments.date BETWEEN '{start_s}' AND '{end_s}' "
        f"ORDER BY metrics.cost_micros DESC LIMIT {limit}"
    )
    results = ads_search(
        google_oauth,
        ads_config["customer_id"],
        query,
        developer_token=ads_config["developer_token"],
        login_customer_id=ads_config.get("login_customer_id"),
    )
    rows = [_distill_ads_row(r) for r in results]
    return {
        "level": level,
        "start_date": start_s,
        "end_date": end_s,
        "rows": rows,
        "totals": {
            "cost": round(sum(r["cost"] for r in rows), 2),
            "clicks": sum(r["clicks"] for r in rows),
            "impressions": sum(r["impressions"] for r in rows),
            "conversions": round(sum(r["conversions"] for r in rows), 2),
        },
        "note": "cost is in account currency; only the top rows by cost are returned.",
    }


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
    google_oauth: dict[str, str] | None,
    gsc_site: str | None,
    google_ads_config: dict[str, str] | None,
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
            {
                "name": "hubspot_get_associations",
                "description": (
                    "Get the CRM records associated with a set of records — e.g. "
                    "the companies and/or contacts linked to specific deals. Pass "
                    "the record ids (deal ids come back in hubspot_deals_report's "
                    "sample and hubspot_search_deals results) and the type you "
                    "want back. Returns each record's associated objects with "
                    "their details. Call once per target type (companies, then "
                    "contacts) if you need both."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "object_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Record ids to look up associations for (e.g. deal ids).",
                        },
                        "to_object_type": {
                            "type": "string",
                            "description": "Associated type to return: companies, contacts, or deals.",
                        },
                        "from_object_type": {
                            "type": "string",
                            "description": "Type of the object_ids: deals (default), companies, or contacts.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max associated records per id (<=25, default 10).",
                        },
                    },
                    "required": ["object_ids", "to_object_type"],
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
                "hubspot_get_associations": lambda a: _hubspot_associations(a, hubspot_token=hubspot_token),
            }
        )

    if google_oauth:
        tools += [
            {
                "name": "gsc_search_analytics",
                "description": (
                    "Our site's organic Google Search performance from Search "
                    "Console: clicks, impressions, CTR, and average position over "
                    "a date range. Break down by dimensions (query, page, "
                    "country, device, date) for top queries/pages, or pass an "
                    "empty dimensions list for totals. For period-over-period, "
                    "call twice and compare."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "YYYY-MM-DD (inclusive)."},
                        "end_date": {"type": "string", "description": "YYYY-MM-DD (inclusive)."},
                        "dimensions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "e.g. ['query'], ['page'], ['date']; [] for totals only.",
                        },
                        "row_limit": {"type": "integer", "description": "Max rows (<=100)."},
                        "site_url": {
                            "type": "string",
                            "description": "Property URL (sc-domain:example.com or https://example.com/). Omit to use the default; use gsc_list_sites to discover.",
                        },
                    },
                    "required": ["start_date", "end_date"],
                },
            },
            {
                "name": "gsc_list_sites",
                "description": "List the Search Console properties we have access to (use to find the site_url).",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
        ]
        handlers.update(
            {
                "gsc_search_analytics": lambda a: _gsc_search(a, google_oauth=google_oauth, default_site=gsc_site),
                "gsc_list_sites": lambda a: _gsc_sites(a, google_oauth=google_oauth),
            }
        )

        if google_ads_config:
            tools.append(
                {
                    "name": "google_ads_report",
                    "description": (
                        "Our Google Ads (paid search) performance — impressions, "
                        "clicks, cost, conversions, CTR, avg CPC — over a date "
                        "range, broken down by campaign (default), ad_group, or "
                        "keyword. Returns per-row metrics plus totals."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "level": {
                                "type": "string",
                                "description": "campaign | ad_group | keyword. Default campaign.",
                            },
                            "since_days": {"type": "integer", "description": "Last N days (alternative to start/end)."},
                            "start_date": {"type": "string", "description": "YYYY-MM-DD."},
                            "end_date": {"type": "string", "description": "YYYY-MM-DD."},
                            "limit": {"type": "integer", "description": "Max rows by cost (<=100)."},
                        },
                        "required": [],
                    },
                }
            )
            handlers["google_ads_report"] = lambda a: _ads_report(
                a, google_oauth=google_oauth, ads_config=google_ads_config
            )

    return tools, handlers


def answer_question(
    question: str,
    *,
    anthropic_key: str,
    history: list[dict[str, Any]] | None = None,
    apollo_key: str | None = None,
    semrush_key: str | None = None,
    hubspot_token: str | None = None,
    google_oauth: dict[str, str] | None = None,
    gsc_site: str | None = None,
    google_ads_config: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Answer a free-text question by letting Claude call the app's data tools.

    Pass `history` (the `messages` returned by a previous call) to continue a
    conversation — Claude then has the earlier questions, tool calls, and
    results as context for a follow-up. The new question is appended as a user
    turn and the final answer as an assistant turn.

    Returns {"answer": str, "tools_used": [{"name", "input"}, ...], "messages":
    [...]}, where `messages` is the full updated conversation to feed back in as
    `history` next time. `tools_used` covers only this turn. Raises ClaudeError
    on Anthropic API errors. Individual tool failures are fed back to the model
    as error results rather than raised, so it can adapt.
    """
    client = anthropic.Anthropic(api_key=anthropic_key)
    tools, handlers = _build_toolbox(
        apollo_key, semrush_key, hubspot_token, google_oauth, gsc_site, google_ads_config
    )

    # Give the model today's date so it can resolve relative ranges (YTD, last
    # year, last 30 days) itself.
    system = f"{_SYSTEM}\n\nToday's date is {datetime.now().date().isoformat()}."

    messages: list[dict[str, Any]] = list(history) if history else []
    messages.append({"role": "user", "content": question})
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

        seo_usage.record_claude(_AGENT_MODEL, response.usage)

        if response.stop_reason != "tool_use":
            answer = "".join(
                b.text for b in response.content if b.type == "text"
            ).strip()
            # Record the final assistant turn so a follow-up call has full context.
            messages.append({"role": "assistant", "content": response.content})
            return {
                "answer": answer or "(no answer produced)",
                "tools_used": tools_used,
                "messages": messages,
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

    fallback = (
        "I couldn't reach a final answer within the step limit. Try a more "
        "specific question."
    )
    # Close the conversation on an assistant turn so history stays valid (the
    # loop left it ending on a tool_result user turn).
    messages.append({"role": "assistant", "content": fallback})
    return {
        "answer": fallback,
        "tools_used": tools_used,
        "messages": messages,
    }
