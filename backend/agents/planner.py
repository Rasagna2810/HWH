import json
import re
from datetime import datetime

from bson import ObjectId
from langchain_core.messages import HumanMessage, SystemMessage

from agents.hindsight import recall_recommendation_hindsight
from core.database import get_db
from core.llm import get_llm
from database.mongo import get_hindsight_matches
from models.state import AgentState

_SYSTEM = """You are the Workflow Planner for a reusable B2B SaaS Customer Success Decision Intelligence Platform.

Your responsibility is ONLY to decide the analysis strategy.

Do NOT perform business reasoning.
Do NOT generate recommendations.
Do NOT generate knowledge base search queries.

The Internal Knowledge Agent will retrieve enterprise knowledge.
The Reasoning Agent will analyze the customer.
The Recommendation Agent will generate next best actions.

Return ONLY valid JSON:

{
  "needs_kb_search": true,
  "kb_query": "specific search query for internal knowledge",
  "priority": "high|medium|low",
  "focus": "renewal|expansion|health|technical|escalation",
  "rationale": "One sentence explaining the workflow decision."
}

Decision Rules:

needs_kb_search = true when:
- Technical issues require troubleshooting guidance.
- Customer interactions mention product functionality.
- Internal playbooks may help decision making.
- Business processes (renewal, onboarding, expansion, escalation) require company-specific guidance.

Rules for kb_query - be SPECIFIC, include:
  - Product names / integrations mentioned (Epic, Salesforce, SSO)
  - Problem type (integration failure, low adoption, renewal risk)
  - Industry (healthcare, logistics, SaaS)
  - Urgency signal (45 days, critical, at risk)

Priority Rules

HIGH Priority
- The same critical issue has occurred again after being marked as completed.
- Customer has reported the same issue multiple times within the last 60 days.
- Customer has sent multiple follow-up emails without satisfactory resolution.
- Contract renewal is within 45 days and unresolved business-critical issues exist.
- Executive stakeholders (CTO, CEO, VP, etc.) are involved in escalations.
- Previous recommendations related to this issue were rejected or not completed.
- The issue is blocking the customer business operations.

MEDIUM Priority
- New issue reported for the first time.
- Feature request has been pending for more than 30 days.
- Customer requested product improvements or roadmap updates.
- Contract renewal is within 45-90 days.
- Previous recommendations are still pending.
- Customer reports moderate dissatisfaction but business operations continue normally.

LOW Priority
- Previous recommendations have been completed successfully.
- No recurring issues are found.
- Customer interactions are mostly positive.
- Customer is requesting information or minor enhancements.
- Renewal is more than 90 days away.
- No unresolved critical tickets or escalations exist.

focus:
Choose ONE primary business objective:
- renewal
- expansion
- health
- technical
- escalation

Output JSON only."""


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found: {text[:200]}")


def _fallback_plan(customer: dict, days_to_renewal) -> dict:
    hs = customer.get("health_score", 100)
    dtr = days_to_renewal or 999
    tags = customer.get("tags", [])
    return {
        "needs_kb_search": hs < 70 or dtr < 90 or "renewal-risk" in tags,
        "kb_query": f"{customer.get('name', '')} {customer.get('industry', '')} renewal risk integration",
        "priority": "high" if (hs < 50 or dtr < 45) else "medium" if (hs < 70 or dtr < 90) else "low",
        "focus": "renewal" if dtr < 90 else "health",
        "rationale": "Derived from CRM data (Gemini parse failed)",
    }


def _format_hindsight(entries: list[dict]) -> str:
    if not entries:
        return "No prior recommendation decisions or outcomes are available yet."

    return "\n".join(
        "- "
        f"Decision: {entry.get('decision', 'unknown')}; "
        f"Recommendation: {entry.get('recommendation', 'N/A')}; "
        f"Feedback: {entry.get('feedback') or 'none provided'}; "
        f"Actual outcome: {entry.get('outcome') or 'not completed yet'}; "
        f"Success metric: {entry.get('success_metric') or 'not specified'}"
        for entry in entries
    )


def _format_hindsight_evidence(entry: dict) -> str:
    evidence = [
        f"decision={entry.get('decision')}" if entry.get("decision") else "",
        f"action={entry.get('recommendation')}" if entry.get("recommendation") else "",
        f"feedback={entry.get('feedback')}" if entry.get("feedback") else "",
        f"outcome={entry.get('outcome')}" if entry.get("outcome") else "",
        f"recalled fact={entry.get('memory')}" if entry.get("memory") else "",
    ]
    summary = re.sub(r"\s+", " ", "; ".join(value for value in evidence if value)).strip()
    if not summary:
        return ""
    return f"[Planner] Hindsight API evidence: {summary[:280]}"


def _legacy_recommendation_memories(reports: list[dict]) -> list[dict]:
    """Adapt older recommendation documents into the hindsight prompt shape."""
    memories = []
    for report in reports:
        for recommendation in report.get("recommendations", []):
            status = str(recommendation.get("status", "")).lower()
            todo_status = str(recommendation.get("todo_status", "")).lower()
            if status not in {"approved", "rejected", "accepted", "declined"}:
                continue

            decision = "rejected" if status in {"rejected", "declined"} else "approved"
            memories.append({
                "source_key": f"recommendation:{report.get('_id')}:{recommendation.get('id', recommendation.get('_id', 'unknown'))}",
                "decision": decision,
                "recommendation": recommendation.get("title") or recommendation.get("recommendation", "N/A"),
                "feedback": (
                    recommendation.get("rejection_reason")
                    or recommendation.get("reviewer_feedback")
                    or recommendation.get("feedback", "")
                ),
                "outcome": recommendation.get("outcome", "") if todo_status == "done" else "",
                "success_metric": recommendation.get("success_metric", ""),
                "todo_status": todo_status,
                "created_at": recommendation.get("done_at") or report.get("created_at", ""),
                "done_at": recommendation.get("done_at", ""),
            })
    return memories


async def planner_agent(state: AgentState) -> AgentState:
    customer_name = state["customer_name"]
    customer_id = state["customer_id"]
    logs: list[str] = []
    db = get_db()
    llm = get_llm()

    try:
        oid = ObjectId(customer_id)
        customer = await db.customers.find_one({"_id": oid})
    except Exception:
        customer = await db.customers.find_one({"name": customer_name})

    if not customer:
        customer = {}

    industry = str(customer.get("industry", "")).strip()
    hindsight_entries = []
    hindsight_api_entries = []
    hindsight_api_count = 0
    hindsight_memory_count = 0
    legacy_memory_count = 0
    try:
        hindsight_api_entries = await recall_recommendation_hindsight(
            customer_id=customer_id,
            industry=industry,
            limit=8,
        )
        hindsight_api_count = len(hindsight_api_entries)
        industry_history = await get_hindsight_matches(industry=industry, limit=8) if industry else []
        customer_history = await get_hindsight_matches(customer_id=customer_id, limit=8)
        hindsight_memory_count = len(customer_history) + len(industry_history)
        seen_hindsight = set()
        for entry in [*hindsight_api_entries, *customer_history, *industry_history]:
            key = entry.get("source_key") or str(entry.get("_id", ""))
            if key and key not in seen_hindsight:
                hindsight_entries.append(entry)
                seen_hindsight.add(key)

        # Older recommendations were stored only in `recommendations` before
        # hindsight persistence was wired in. Use those as a migration-safe
        # fallback so historical approvals/rejections are not ignored.
        reports = await db.recommendations.find({
            "$or": [
                {"customer_id": customer_id},
                {"customer_name": customer_name},
            ]
        }).sort("created_at", -1).limit(30).to_list(length=30)
        legacy_entries = _legacy_recommendation_memories(reports)
        legacy_memory_count = len(legacy_entries)
        for entry in legacy_entries:
            key = entry["source_key"]
            if key not in seen_hindsight:
                hindsight_entries.append(entry)
                seen_hindsight.add(key)
    except Exception as exc:
        logs.append(f"[Planner] Hindsight read failed; continuing without it: {exc}")
    hindsight_entries.sort(
        key=lambda entry: str(
            entry.get("done_at")
            or entry.get("created_at")
            or entry.get("mentioned_at")
            or ""
        ),
        reverse=True,
    )
    hindsight_entries = hindsight_entries[:10]
    hindsight_note = _format_hindsight(hindsight_entries)

    recent = await db.interactions.find({"customer_name": customer_name}).sort("date", -1).limit(3).to_list(length=3)
    recent_snippets = " | ".join(i.get("content", "")[:150] for i in recent)

    contract_end = customer.get("contract_end", "")
    days_to_renewal = None
    if contract_end:
        try:
            end_dt = datetime.fromisoformat(contract_end[:10])
            days_to_renewal = (end_dt - datetime.utcnow()).days
        except Exception:
            pass

    past = state.get("past_recommendations", [])
    past_note = ""
    if past:
        last = past[-1]
        past_note = f"\nPrevious analysis: {last.get('summary', 'N/A')}"

    logs.append(f"[Planner] CRM loaded - Days to renewal: {days_to_renewal}, ARR: ${customer.get('arr', 0):,}")
    logs.append(
        f"[Planner] Hindsight loaded: {len(hindsight_entries)} records "
        f"(Hindsight API recall: {hindsight_api_count}, MongoDB history: "
        f"{hindsight_memory_count}, legacy recommendation history: {legacy_memory_count})"
    )
    logs.extend(
        evidence_log
        for entry in hindsight_api_entries[:3]
        if (evidence_log := _format_hindsight_evidence(entry))
    )

    user_prompt = f"""Plan the analysis strategy for this customer:

Customer: {customer_name}
Industry: {customer.get('industry', 'Unknown')}
ARR: ${customer.get('arr', 0):,}
Health Score: {customer.get('health_score', 'Unknown')}/100
Status: {customer.get('status', 'Unknown')}
Tier: {customer.get('tier', 'Unknown')}
Days to Contract Renewal: {days_to_renewal if days_to_renewal is not None else 'Unknown'}
Tags: {', '.join(customer.get('tags', []))}

Recent interactions (last 3):
{recent_snippets if recent_snippets else 'No recent interactions'}

Past recommendation decisions, reviewer feedback, and actual outcomes:
{hindsight_note}

Use rejected feedback to avoid repeating unsuitable actions. Use completed outcomes
to identify what worked or failed; do not assume approval alone means success.
{past_note}

Generate the analysis strategy JSON."""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
        plan = _extract_json(response.content)
    except Exception as e:
        logs.append(f"[Planner] Gemini error: {e} - using fallback")
        plan = _fallback_plan(customer, days_to_renewal)

    pipeline = ["internal_knowledge", "reasoning", "report"]

    logs.append(f"[Planner] Strategy: {plan.get('focus', '?').upper()} | Priority: {plan.get('priority', '?').upper()}")
    logs.append(f"[Planner] KB search: {plan.get('needs_kb_search')} - query: \"{plan.get('kb_query', '')}\"")
    logs.append(f"[Planner] Pipeline: {' -> '.join(pipeline)}")
    logs.append(f"[Planner] Rationale: {plan.get('rationale', '')}")

    return {
        **state,
        "plan": pipeline,
        "current_step": 0,
        "status": "running",
        "hindsight_entries": hindsight_entries[:10],
        "agent_logs": logs,
        "customer_context": {
            **state.get("customer_context", {}),
            "kb_query": plan.get("kb_query", customer_name),
            "needs_kb_search": plan.get("needs_kb_search", True),
            "plan_focus": plan.get("focus", "health"),
            "plan_priority": plan.get("priority", "medium"),
            "customer": {**customer, "_id": str(customer.get("_id", ""))},
            "days_to_renewal": days_to_renewal,
        },
    }
