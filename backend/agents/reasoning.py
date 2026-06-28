import json
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, SystemMessage
from models.state import AgentState
from core.llm import get_llm
 
_SYSTEM = """You are a Senior Customer Success AI Analyst for a B2B SaaS company.
 
You receive raw customer data and must output a complete analysis + prioritised action plan in ONE response.
 
Be specific and evidence-based. Every recommendation must cite the exact data that justifies it.
Output ONLY valid JSON, no markdown fences."""
 
_COMBINED_PROMPT = """Analyse this customer and generate Next Best Actions.
 
=== CUSTOMER ===
{customer_json}
 
=== RECENT INTERACTIONS ({interaction_count} total, newest first) ===
{interactions_text}
 
=== KNOWLEDGE BASE ARTICLES MATCHED ===
{kb_text}
 
=== MEMORY: PAST RECOMMENDATIONS ===
{past_text}
 
=== TODAY'S DATE ===
{today}
 
INSTRUCTIONS:
1. Calculate health_score (0-100) and status from the interaction data — do NOT use the stored values.
   - Start at 100. Deduct for: open critical tickets (-20), renewal risk mentioned (-15),
     competitor evaluation (-20), low adoption mentioned (-10), escalation emails (-10),
     unanswered urgent emails (-10). Add for: positive check-ins (+5), expansion signals (+5).
2. For each recommendation, set a realistic due_date (ISO YYYY-MM-DD) based on urgency:
   immediate = within 2 days, this-week = within 7 days, this-month = within 30 days, this-quarter = within 90 days.
3. Set priority: P1 = immediate + high business impact, P2 = this-week/this-month, P3 = this-quarter.
4. expected_outcome must be a concrete, measurable result (not vague).
 
Return this exact JSON structure:
{{
  "calculated_health": {{
    "score": 42,
    "status": "At Risk",
    "trend": "declining|stable|improving",
    "reasoning": "one sentence explaining the score"
  }},
  "risks": [
    {{
      "title": "short title",
      "severity": "critical|high|medium|low",
      "description": "specific description citing data",
      "evidence": "exact quote or data point from interactions",
      "business_impact": "$ or qualitative impact"
    }}
  ],
  "opportunities": [
    {{
      "title": "short title",
      "type": "expansion|renewal|advocacy|upsell",
      "description": "specific opportunity",
      "potential_value": "$ estimate",
      "evidence": "data point supporting this"
    }}
  ],
  "recommendations": [
    {{
      "id": "nba-001",
      "rank": 1,
      "title": "Verb + who + specific action",
      "category": "risk-mitigation|expansion|relationship|technical|strategic",
      "urgency": "immediate|this-week|this-month|this-quarter",
      "priority": "P1|P2|P3",
      "due_date": "2025-01-15",
      "description": "exactly what to do and why in 2-3 sentences",
      "owner": "CSM|Engineering|Account Executive|Executive",
      "expected_outcome": "specific measurable result e.g. Epic integration restored, adoption rises from 34% to 60% in 30 days",
      "business_impact": "$ or risk reduction",
      "evidence": [
        {{"source": "meeting_notes|ticket|email|kb_article", "detail": "specific text"}}
      ],
      "execution_steps": ["Step 1", "Step 2", "Step 3"],
      "risks_if_not_done": "consequence of inaction",
      "success_metric": "how to know it worked"
    }}
  ],
  "key_insights": ["insight 1", "insight 2", "insight 3"],
  "executive_summary": "3-4 sentences covering situation, risk, and recommended posture",
  "total_risk_exposure": "$480K ARR at risk or qualitative",
  "total_opportunity_value": "$80K expansion potential or qualitative"
}}"""
 
 
def _clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        if raw.endswith("```"):
            raw = raw[:-3]
    return raw.strip()
 
 
async def reasoning_agent(state: AgentState) -> AgentState:
    llm       = get_llm()
    ctx       = state.get("customer_context", {})
    past_recs = state.get("past_recommendations", [])
    logs: list[str] = []
 
    customer     = ctx.get("customer", {})
    interactions = ctx.get("interactions", [])
    kb_articles  = ctx.get("kb_articles", [])
    interaction_lines = []
    for i in interactions[:15]:
        interaction_lines.append(
            f"[{i.get('type','?').upper()} | {str(i.get('date',''))[:10]} | by {i.get('author','?')}]\n"
            f"{i.get('content', '')}"
        )
    interactions_text = "\n\n".join(interaction_lines) if interaction_lines else "No interactions on record."
 
    kb_lines = []
    for a in kb_articles:
        kb_lines.append(f"[{a.get('type','?').upper()}] {a.get('title','')}\n{a.get('content','')[:400]}")
    kb_text = "\n\n".join(kb_lines) if kb_lines else "No KB articles matched."
 
    # ── Past recommendations (memory) ─────────────────────────
    past_lines = []
    for r in past_recs[:2]:
        completed = [a.get("title","") for a in r.get("recommendations",[]) if a.get("todo_status") == "done"]
        past_lines.append(
            f"Analysis from {str(r.get('created_at',''))[:10]}: {r.get('summary','')}\n"
            f"Completed actions: {', '.join(completed) if completed else 'none'}"
        )
    past_text = "\n\n".join(past_lines) if past_lines else "No previous recommendations."
 
    prompt = _COMBINED_PROMPT.format(
        customer_json = json.dumps({
            "name":         customer.get("name"),
            "industry":     customer.get("industry"),
            "arr":          customer.get("arr"),
            "tier":         customer.get("tier"),
            "contract_end": str(customer.get("contract_end",""))[:10],
            "contacts":     customer.get("contacts", []),
        }, indent=2),
        interaction_count = len(interactions),
        interactions_text = interactions_text,
        kb_text           = kb_text,
        past_text         = past_text,
        today             = datetime.utcnow().strftime("%Y-%m-%d"),
    )
 
    logs.append("[Reasoning] Running combined analysis + recommendations (1 Gemini call)...")
 
    response = await llm.ainvoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=prompt),
    ])
 
    raw = _clean_json(response.content)
 
    try:
        result = json.loads(raw)
    except Exception as e:
        logs.append(f"[Reasoning] JSON parse failed: {str(e)[:80]}")
        result = {
            "calculated_health":    {"score": 50, "status": "Unknown", "trend": "stable", "reasoning": "Parse failed"},
            "risks":                [],
            "opportunities":        [],
            "recommendations":      [],
            "key_insights":         ["LLM response parse failed"],
            "executive_summary":    "Analysis failed — check server logs",
            "total_risk_exposure":  "Unknown",
            "total_opportunity_value": "Unknown",
        }
 
    recs   = result.get("recommendations", [])
    risks  = result.get("risks", [])
    opps   = result.get("opportunities", [])
    calc   = result.get("calculated_health", {})
 
    logs.append(f"[Reasoning] Calculated health: {calc.get('score','?')}/100 — {calc.get('status','?')} ({calc.get('trend','?')})")
    logs.append(f"[Reasoning] {len(risks)} risks, {len(opps)} opportunities")
    logs.append(f"[Reasoning] {len(recs)} Next Best Actions generated")
    for r in recs:
        logs.append(
            f"[Reasoning] {r.get('priority','?')} [{r.get('urgency','?').upper()}] "
            f"{r.get('title','?')} — due {r.get('due_date','?')}"
        )
 
    analysis = {
        "calculated_health":      calc,
        "risks":                  risks,
        "opportunities":          opps,
        "key_insights":           result.get("key_insights", []),
        "executive_summary":      result.get("executive_summary", ""),
        "total_risk_exposure":    result.get("total_risk_exposure", ""),
        "total_opportunity_value":result.get("total_opportunity_value", ""),
        "reasoning_summary":      result.get("executive_summary", ""),
        "information_gaps":       [],
        "business_context":       {},
    }
 
    return {
        **state,
        "analysis":        analysis,
        "recommendations": recs,
        "agent_logs":      logs,
    }
 