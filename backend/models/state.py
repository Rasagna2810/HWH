from typing import TypedDict, Annotated, Any
import operator


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────
    customer_id: str
    customer_name: str
    user_email: str

    # ── Planner ────────────────────────────────────────
    plan: list[str]
    current_step: int

    # ── Agent outputs ──────────────────────────────────
    customer_context: dict        # Internal Knowledge Agent
    external_intel: dict          # External Research Agent
    analysis: dict                # Reasoning Agent
    recommendations: list[dict]   # Recommendation Agent
    report: dict                  # Report Agent

    # ── Streaming / observability ──────────────────────
    # Annotated with operator.add so LangGraph merges lists
    agent_logs: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]

    # ── Memory ─────────────────────────────────────────
    past_recommendations: list[dict]

    # ── Lifecycle ──────────────────────────────────────
    # running | awaiting_approval | approved | rejected | complete
    status: str