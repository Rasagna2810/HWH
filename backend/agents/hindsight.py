import json
import logging
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import get_settings
from core.hindsight import get_hindsight_client
from core.llm import get_llm
from database.mongo import get_hindsight_matches, save_hindsight_entry

logger = logging.getLogger(__name__)
SYSTEM_PROMPT = """You are the Planner Agent for AegisAI, an autonomous B2B prospect intelligence platform. You learn from past campaign results to make smarter plans each time. Always respond with ONLY a raw JSON object. No markdown, no explanation."""


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text or "").replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found: {cleaned[:200]}")
    return json.loads(match.group(0))


async def _get_hindsight(campaign: dict[str, Any]) -> str:
    industry = str(campaign.get("industry", "")).strip()
    country = str(campaign.get("country", "")).strip()
    persona = str(campaign.get("target_persona") or campaign.get("persona", "")).strip()

    past = await get_hindsight_matches(industry=industry, country=country, persona=persona, limit=5)
    if not past:
        past = await get_hindsight_matches(industry=industry, limit=3)

    if not past:
        return "No past hindsight available for this campaign type."

    lines = ["LESSONS FROM PAST CAMPAIGNS:"]
    for p in past:
        lessons = p.get("lessons", {}) or {}
        metrics = p.get("metrics", {}) or {}
        created = str(p.get("created_at", ""))[:10]
        lines.append(f"\nCampaign: {p.get('campaign_name', 'Unknown')} ({created})")
        lines.append(f"  Results: {metrics.get('total_leads', 'n/a')} leads, avg score {metrics.get('avg_score', 'n/a')}")
        lines.append(f"  What worked: {lessons.get('what_worked', [])}")
        lines.append(f"  What failed: {lessons.get('what_failed', [])}")
        lines.append(f"  Best queries: {lessons.get('best_queries', [])}")
        lines.append(f"  Worst queries: {lessons.get('worst_queries', [])}")
        lines.append(f"  Best signals: {lessons.get('best_signals', [])}")
        lines.append(f"  Avoid: {lessons.get('avoid_next_time', [])}")
        lines.append(f"  Do differently: {lessons.get('do_next_time', [])}")
        lines.append(f"  Summary: {lessons.get('summary', p.get('summary', ''))}")
    return "\n".join(lines)


def _build_initial_prompt(campaign: dict[str, Any], hindsight_context: str) -> str:
    return f"""
Analyze this B2B sales campaign and build a smart execution plan.
Use the hindsight from past campaigns to avoid mistakes and repeat successes.

CAMPAIGN:
- Name: {campaign.get('name')}
- Industry: {campaign.get('industry')}
- Country: {campaign.get('country')}
- Company Size: {campaign.get('company_size')}
- Revenue Range: {campaign.get('revenue')}
- Target Persona: {campaign.get('target_persona')}
- Business Goals: {campaign.get('business_goals')}
- ICP Rules: {campaign.get('icp_rules')}

{hindsight_context}

Using the hindsight above, return ONLY raw JSON:
{{
  "reasoning": "2-3 sentences on strategy, referencing what past campaigns taught you",
  "search_queries": ["query1", "query2", "query3"],
  "target_count": 30,
  "validation_rules": {{
    "must_have_website": true,
    "must_match_industry": true,
    "must_match_country": false,
    "min_employees": 10,
    "max_employees": 5000
  }},
  "priority_signals": ["recent funding", "hiring engineers"],
  "scoring_weights": {{
    "growth": 0.35,
    "technology": 0.25,
    "funding": 0.25,
    "hiring": 0.15
  }},
  "skip_agents": [],
  "min_score_threshold": 65,
  "persona_depth": "standard",
  "outreach_angle": "One sentence on outreach angle informed by past lessons"
}}
"""


def _build_replan_prompt(campaign: dict[str, Any], state: dict[str, Any], errors: list, found: int, validated: int, hindsight_context: str) -> str:
    plan = state.get("plan", {})
    return f"""
Previous plan failed. Adapt it using hindsight.

CAMPAIGN: industry={campaign.get('industry')}, country={campaign.get('country')}, persona={campaign.get('target_persona')}
RESULTS: found={found}, validated={validated}
ERRORS: {errors}
PREVIOUS QUERIES: {plan.get('search_queries', [])}

{hindsight_context}

Make broader queries. Avoid what failed before. Return ONLY raw JSON:
{{
  "reasoning": "Why you changed the plan referencing past lessons",
  "search_queries": ["broader query 1", "alternative query 2", "news query 3"],
  "target_count": 20,
  "validation_rules": {{
    "must_have_website": true,
    "must_match_industry": false,
    "must_match_country": false,
    "min_employees": 5,
    "max_employees": 10000
  }},
  "priority_signals": ["any growth signal"],
  "scoring_weights": {{"growth": 0.35, "technology": 0.25, "funding": 0.25, "hiring": 0.15}},
  "skip_agents": [],
  "min_score_threshold": 50,
  "persona_depth": "standard",
  "outreach_angle": "Revised outreach angle from lessons"
}}
"""


def _validate_plan(plan: dict[str, Any]):
    plan.setdefault("search_queries", ["SaaS companies USA"])
    plan.setdefault("target_count", 20)
    plan.setdefault("skip_agents", [])
    plan.setdefault("min_score_threshold", 60)
    plan.setdefault("priority_signals", ["growth", "hiring"])
    plan.setdefault("scoring_weights", {"growth": 0.3, "technology": 0.25, "funding": 0.25, "hiring": 0.2})
    plan.setdefault("validation_rules", {})
    plan.setdefault("persona_depth", "standard")
    plan.setdefault("outreach_angle", "Introduce your solution")
    plan.setdefault("reasoning", "Plan generated")
    if not isinstance(plan["search_queries"], list) or len(plan["search_queries"]) == 0:
        raise ValueError("search_queries must be a non-empty list")


def _fallback_plan(campaign: dict[str, Any], iteration: int) -> dict[str, Any]:
    industry = campaign.get("industry", "SaaS")
    country = campaign.get("country", "USA")
    persona = campaign.get("target_persona", "CTO")
    queries = [
        f"{industry} companies {country}",
        f"{industry} startups hiring {persona}",
        f"best {industry} software companies {country} 2024",
    ]
    if iteration > 0:
        queries += [f"{industry} technology companies funding 2024", f"top {industry} vendors {country}"]

    return {
        "reasoning": f"Fallback plan iteration {iteration + 1}",
        "search_queries": queries,
        "target_count": 20,
        "validation_rules": {"must_have_website": True, "must_match_industry": iteration == 0, "must_match_country": False, "min_employees": 5, "max_employees": 10000},
        "priority_signals": ["hiring", "funding", "growth"],
        "scoring_weights": {"growth": 0.3, "technology": 0.25, "funding": 0.25, "hiring": 0.2},
        "skip_agents": [],
        "min_score_threshold": 50 if iteration > 0 else 60,
        "persona_depth": "standard",
        "outreach_angle": f"Introduce solution to {persona} in {industry}",
    }


async def run(state: dict[str, Any]) -> dict[str, Any]:
    campaign = state.get("campaign", {})
    errors = state.get("errors", [])
    found = len(state.get("companies", []))
    validated = len(state.get("validated", []))
    iteration = state.get("planner_iteration", 0)
    state.setdefault("logs", [])
    state.setdefault("errors", [])

    hindsight_context = await _get_hindsight(campaign)

    if iteration == 0:
        prompt = SYSTEM_PROMPT + "\n\n" + _build_initial_prompt(campaign, hindsight_context)
    else:
        prompt = SYSTEM_PROMPT + "\n\n" + _build_replan_prompt(campaign, state, errors, found, validated, hindsight_context)

    try:
        llm = get_llm()
        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        plan = _extract_json(response.content)
        _validate_plan(plan)
    except Exception as e:
        plan = _fallback_plan(campaign, iteration)
        state["errors"].append(f"Planner error iteration {iteration}: {e}")

    state["plan"] = plan
    state["planner_iteration"] = iteration + 1
    state["skip_agents"] = plan.get("skip_agents", [])
    state["scoring_weights"] = plan.get("scoring_weights", {})
    state["validation_rules"] = plan.get("validation_rules", {})

    reasoning = plan.get("reasoning", "")
    state["logs"].append(f"Planner [{iteration + 1}]: {reasoning}")
    state["logs"].append(f"Planner [{iteration + 1}]: {len(plan.get('search_queries', []))} queries, skip={plan.get('skip_agents', [])}, min_score={plan.get('min_score_threshold', 60)}")
    return state


async def save_recommendation_hindsight(record: dict[str, Any]) -> str:
    payload = {
        "source_key": record.get("source_key", ""),
        "customer_id": record.get("customer_id", ""),
        "customer_name": record.get("customer_name") or record.get("campaign_name", "Unknown"),
        "industry": record.get("industry", ""),
        "country": record.get("country", ""),
        "target_persona": record.get("target_persona") or record.get("persona", ""),
        "campaign_name": record.get("campaign_name") or record.get("customer_name") or "Unknown",
        "problem": record.get("problem", ""),
        "summary": record.get("summary", record.get("outcome", "")),
        "outcome": record.get("outcome", ""),
        "recommendation": record.get("recommendation", ""),
        "expected_outcome": record.get("expected_outcome", ""),
        "success_metric": record.get("success_metric", ""),
        "decision": record.get("decision", ""),
        "feedback": record.get("feedback", ""),
        "reviewer": record.get("reviewer", ""),
        "todo_status": record.get("todo_status", ""),
        "approved_at": record.get("approved_at", ""),
        "done_at": record.get("done_at", ""),
        "metrics": record.get("metrics", {}) or {},
        "lessons": record.get("lessons", {}) or {},
        "accepted": record.get("accepted", record.get("decision") == "approved"),
        "created_at": record.get("created_at") or datetime.utcnow().isoformat(),
    }
    local_id = await save_hindsight_entry(payload)
    client = get_hindsight_client()
    if client is None:
        logger.info("Hindsight API disabled; recommendation saved to MongoDB fallback")
        return local_id

    settings = get_settings()
    source_key = payload.get("source_key") or local_id
    tags = []
    if payload.get("customer_id"):
        tags.append(f"customer:{payload['customer_id']}")
    industry_tag = re.sub(r"[^a-z0-9]+", "-", payload.get("industry", "").lower()).strip("-")
    if industry_tag:
        tags.append(f"industry:{industry_tag}")

    content_fields = (
        ("Customer", payload.get("customer_name")),
        ("Industry", payload.get("industry")),
        ("Problem", payload.get("problem")),
        ("Decision", payload.get("decision")),
        ("Recommendation", payload.get("recommendation")),
        ("Recorded at", payload.get("created_at")),
        ("Approved at", payload.get("approved_at")),
        ("Completed at", payload.get("done_at")),
        ("Todo status", payload.get("todo_status")),
        ("Reviewer", payload.get("reviewer")),
        ("Reviewer feedback", payload.get("feedback")),
        ("Expected outcome", payload.get("expected_outcome")),
        ("Actual outcome", payload.get("outcome")),
        ("Success metric", payload.get("success_metric")),
        ("Lessons", json.dumps(payload.get("lessons", {}), ensure_ascii=True)),
    )
    content = "Customer-success recommendation record:\n" + "\n".join(
        f"{label}: {value}" for label, value in content_fields if value
    )
    metadata = {
        key: str(payload.get(key) or "")
        for key in (
            "source_key", "customer_id", "customer_name", "industry", "country",
            "decision", "recommendation", "feedback", "outcome", "success_metric", "reviewer",
            "todo_status", "created_at", "approved_at", "done_at",
        )
    }
    event_time = payload.get("done_at") or payload.get("created_at") or payload.get("approved_at")
    try:
        event_timestamp = datetime.fromisoformat(str(event_time).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        event_timestamp = None

    try:
        response = await client.aretain(
            bank_id=settings.hindsight_bank_id,
            content=content,
            timestamp=event_timestamp,
            context="customer success recommendation decision and outcome",
            document_id=source_key,
            metadata=metadata,
            tags=tags,
        )
        if getattr(response, "success", True):
            logger.info("[Hindsight] Retained recommendation %s", source_key)
        else:
            logger.warning("[Hindsight] Retain was not successful for %s", source_key)
    except Exception as exc:
        logger.warning(
            "[Hindsight] Retain failed for %s; MongoDB fallback is preserved: %s",
            source_key,
            exc,
        )
    return local_id


async def recall_recommendation_hindsight(
    customer_id: str,
    industry: str = "",
    limit: int = 8,
) -> list[dict[str, Any]]:
    client = get_hindsight_client()
    if client is None:
        return []

    tags = []
    if customer_id:
        tags.append(f"customer:{customer_id}")
    industry_tag = re.sub(r"[^a-z0-9]+", "-", industry.lower()).strip("-")
    if industry_tag:
        tags.append(f"industry:{industry_tag}")
    if not tags:
        return []

    settings = get_settings()
    try:
        response = await client.arecall(
            bank_id=settings.hindsight_bank_id,
            query=(
                "What customer-success recommendations were approved or rejected, "
                "what feedback was given, and what outcomes or lessons followed?"
            ),
            types=["world", "experience", "observation"],
            max_tokens=4096,
            tags=tags,
            tags_match="any_strict",
        )
    except Exception as exc:
        logger.warning("[Hindsight] Recall failed; MongoDB history will be used: %s", exc)
        return []

    memories = []
    seen = set()
    for fact in response.results:
        metadata = getattr(fact, "metadata", None) or {}
        source_key = metadata.get("source_key") or getattr(fact, "id", "")
        if source_key in seen:
            continue
        seen.add(source_key)
        memories.append({
            "source_key": source_key,
            "decision": metadata.get("decision", "unknown"),
            "recommendation": metadata.get("recommendation") or fact.text,
            "feedback": metadata.get("feedback", ""),
            "outcome": metadata.get("outcome", ""),
            "todo_status": metadata.get("todo_status") or ("done" if metadata.get("outcome") else ""),
            "success_metric": metadata.get("success_metric", ""),
            "created_at": (
                metadata.get("done_at")
                or metadata.get("created_at")
                or getattr(fact, "mentioned_at", "")
            ),
            "done_at": metadata.get("done_at", ""),
            "memory": fact.text,
        })
        if len(memories) >= limit:
            break

    logger.info("[Hindsight] Recalled %s recommendation memories", len(memories))
    return memories


async def save_accepted_hindsight(record: dict[str, Any]) -> str:
    """Backward-compatible alias for the recommendation hindsight writer."""
    return await save_recommendation_hindsight(record)


async def record_accepted_hindsight(record: dict[str, Any]) -> str:
    return await save_recommendation_hindsight(record)
