import json as _json
from datetime import datetime
from uuid import uuid4
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from jose import jwt
from core.auth import get_current_user
from core.config import get_settings
from core.database import get_db
from core.orchestrator import run_workflow_streaming
from agents.hindsight import save_recommendation_hindsight
 
router   = APIRouter(prefix="/analysis", tags=["analysis"])
settings = get_settings()
 
 
def _oid(v: str) -> ObjectId | None:
    try:
        return ObjectId(v)
    except Exception:
        return None
 
 
def _ser(doc: dict) -> dict:
    if doc and "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    return doc


async def _customer_for_report(db, report: dict) -> dict:
    customer_id = report.get("customer_id", "")
    customer_oid = _oid(customer_id) if customer_id else None
    customer = await db.customers.find_one({"_id": customer_oid}) if customer_oid else None
    if not customer and report.get("customer_name"):
        customer = await db.customers.find_one({"name": report["customer_name"]})
    return customer or {}


def _hindsight_record(report: dict, customer: dict, recommendation: dict, decision: str,
                      reviewer: str, timestamp: str, feedback: str = "") -> dict:
    title = recommendation.get("title") or recommendation.get("recommendation") or "Recommendation"
    return {
        "source_key": f"recommendation:{report.get('_id')}:{recommendation.get('id')}",
        "customer_id": report.get("customer_id", ""),
        "customer_name": report.get("customer_name", "Unknown"),
        "campaign_name": report.get("customer_name", "Unknown"),
        "industry": customer.get("industry", ""),
        "country": customer.get("country", ""),
        "target_persona": recommendation.get("target_persona") or recommendation.get("persona", ""),
        "problem": recommendation.get("problem") or recommendation.get("risk") or report.get("summary", ""),
        "summary": recommendation.get("description") or recommendation.get("summary") or title,
        "recommendation": title,
        "expected_outcome": recommendation.get("expected_outcome", ""),
        "success_metric": recommendation.get("success_metric", ""),
        "outcome": recommendation.get("outcome", ""),
        "decision": decision,
        "feedback": feedback,
        "reviewer": reviewer,
        "approved_by": recommendation.get("approved_by", ""),
        "approved_at": recommendation.get("approved_at", ""),
        "rejected_by": recommendation.get("rejected_by", ""),
        "rejected_at": recommendation.get("rejected_at", ""),
        "done_by": recommendation.get("done_by", ""),
        "done_at": recommendation.get("done_at", ""),
        "todo_status": recommendation.get("todo_status", ""),
        "metrics": recommendation.get("metrics", {}) or {},
        "lessons": {
            "what_worked": recommendation.get("what_worked") or [],
            "what_failed": recommendation.get("what_failed") or [],
            "best_queries": recommendation.get("best_queries") or [],
            "worst_queries": recommendation.get("worst_queries") or [],
            "best_signals": recommendation.get("best_signals") or [],
            "avoid_next_time": recommendation.get("avoid_next_time") or [],
            "do_next_time": recommendation.get("do_next_time") or [],
            "summary": recommendation.get("outcome") or feedback or recommendation.get("description") or "",
        },
        "accepted": decision == "approved",
        "created_at": timestamp,
    }
 
 
@router.websocket("/ws/{customer_id}")
async def analysis_ws(websocket: WebSocket, customer_id: str):
    await websocket.accept()
 
    try:
        raw        = await websocket.receive_text()
        auth_data  = _json.loads(raw)
        token      = auth_data.get("token", "")
        payload    = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        user_email = payload.get("sub")
        if not user_email:
            await websocket.send_json({"type": "error", "message": "Invalid auth token"})
            await websocket.close(); return
    except Exception as e:
        await websocket.send_json({"type": "error", "message": f"Auth failed: {str(e)}"})
        await websocket.close(); return
 
    db  = get_db()
    oid = _oid(customer_id)
    customer = await db.customers.find_one({"_id": oid}) if oid else None
    if not customer:
        await websocket.send_json({"type": "error", "message": "Customer not found"})
        await websocket.close(); return
 
    final_state = None
    try:
        async for event in run_workflow_streaming(customer_id, customer["name"], user_email):
            if event.get("type") == "final_state":
                final_state = event.get("data"); continue
            await websocket.send_json(event)
 
        if final_state and final_state.get("report"):
            recs = final_state.get("recommendations", [])
            # Stamp each recommendation with todo_status=None at creation time
            for r in recs:
                r.setdefault("id", str(uuid4()))
                r.setdefault("todo_status", None)      # None → not yet reviewed
                r.setdefault("status", "awaiting_approval")  # awaiting_approval | approved | rejected
                r.setdefault("rejection_reason", None)
 
            # Update customer's calculated health in MongoDB
            calc = final_state.get("analysis", {}).get("calculated_health", {})
            if calc.get("score") is not None:
                await db.customers.update_one(
                    {"_id": oid},
                    {"$set": {
                        "health_score": calc["score"],
                        "status":       calc.get("status", "Unknown"),
                    }}
                )
 
            doc = {
                "customer_id":    customer_id,
                "customer_name":  customer["name"],
                "created_at":     datetime.utcnow().isoformat(),
                "created_by":     user_email,
                "summary":        final_state.get("analysis", {}).get("executive_summary", ""),
                "recommendations": recs,
                "report":         final_state["report"],
                "status":         "awaiting_approval",
            }
            result    = await db.recommendations.insert_one(doc)
            report_id = str(result.inserted_id)
 
            await websocket.send_json({
                "type":    "report_ready",
                "agent":   "orchestrator",
                "message": "Report saved — ready for review",
                "data":    {"report_id": report_id, "recommendation_count": len(recs)},
            })
 
    except WebSocketDisconnect:
        print(f"[WS] Disconnected: {customer_id}")
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "agent": "orchestrator", "message": str(e)})
        except Exception:
            pass
 
 
@router.get("/reports/{customer_id}")
async def get_reports(customer_id: str, user=Depends(get_current_user)):
    db   = get_db()
    docs = await db.recommendations.find({"customer_id": customer_id}).sort("created_at", -1).limit(10).to_list(10)
    return [_ser(d) for d in docs]
 
 
@router.get("/report/{report_id}")
async def get_report(report_id: str, user=Depends(get_current_user)):
    db  = get_db()
    oid = _oid(report_id)
    if not oid: raise HTTPException(400, "Invalid report id")
    doc = await db.recommendations.find_one({"_id": oid})
    if not doc: raise HTTPException(404, "Report not found")
    return _ser(doc)

 
@router.post("/report/{report_id}/approve")
async def approve_report(report_id: str, body: dict, user=Depends(get_current_user)):
    """
    action: approve_all | reject_all | approve | reject
    Approved recs → status=approved, todo_status=pending  (appear in Todos)
    Rejected recs → status=rejected, rejection_reason saved
    """
    db  = get_db()
    oid = _oid(report_id)
    if not oid: raise HTTPException(400, "Invalid report id")
    doc = await db.recommendations.find_one({"_id": oid})
    if not doc: raise HTTPException(404, "Report not found")
 
    action   = body.get("action", "")
    rec_ids  = body.get("recommendation_ids", [])
    reason   = body.get("rejection_reason", body.get("feedback", ""))
    recs     = doc.get("recommendations", [])
    now      = datetime.utcnow().isoformat()
    reviewer = user.get("name", user.get("email", ""))
    selected: list[dict] = []
 
    def _approve(r):
        r.update({"status": "approved", "todo_status": "pending",
                  "approved_by": reviewer, "approved_at": now})
 
    def _reject(r):
        r.update({"status": "rejected", "todo_status": None,
                  "rejection_reason": reason,
                  "rejected_by": reviewer, "rejected_at": now})
 
    if action == "approve_all":
        for r in recs:
            _approve(r)
            selected.append(r)
        new_status = "approved"
    elif action == "reject_all":
        for r in recs:
            _reject(r)
            selected.append(r)
        new_status = "rejected"
    elif action in {"approve", "reject"}:
        for r in recs:
            if r.get("id") in rec_ids:
                if action == "approve":
                    _approve(r)
                else:
                    _reject(r)
                selected.append(r)
        if not selected:
            raise HTTPException(404, "No matching recommendation IDs found")
        approved   = sum(1 for r in recs if r.get("status") == "approved")
        new_status = "partially_approved" if approved else "awaiting_approval"
    else:
        raise HTTPException(400, "action must be approve_all, reject_all, approve, or reject")
 
    await db.recommendations.update_one(
        {"_id": oid},
        {"$set": {"recommendations": recs, "status": new_status,
                  "reviewed_by": reviewer, "reviewed_at": now}},
    )

    customer = await _customer_for_report(db, doc)
    persistence_errors = []
    decision = "approved" if action in {"approve", "approve_all"} else "rejected"
    for recommendation in selected:
        try:
            await save_recommendation_hindsight(_hindsight_record(
                doc, customer, recommendation, decision, reviewer, now,
                feedback=reason,
            ))
        except Exception as exc:
            persistence_errors.append(str(exc))
    if persistence_errors:
        raise HTTPException(
            503,
            detail=(
                "The decision was saved to recommendations, but hindsight persistence failed. "
                "Retry the same approval/rejection request; writes are idempotent. "
                f"Error: {persistence_errors[0]}"
            ),
        )
    return {"status": new_status}
@router.get("/todos")
async def get_all_todos(user=Depends(get_current_user)):
    """
    All approved recommendations across all customers with todo_status=pending|done|cancelled.
    """
    db   = get_db()
    docs = await db.recommendations.find({"status": {"$in": ["approved", "partially_approved"]}}).sort("created_at", -1).to_list(100)
    todos = []
    for doc in docs:
        doc_id = str(doc["_id"])
        for r in doc.get("recommendations", []):
            if r.get("status") == "approved":
                todos.append({
                    **r,
                    "report_id":    doc_id,
                    "customer_id":  doc.get("customer_id"),
                    "customer_name":doc.get("customer_name"),
                    "report_date":  doc.get("created_at","")[:10],
                })
    return todos
 
 
@router.get("/todos/{customer_id}")
async def get_customer_todos(customer_id: str, user=Depends(get_current_user)):
    """Approved todos for one customer."""
    db   = get_db()
    docs = await db.recommendations.find({
        "customer_id": customer_id,
        "status": {"$in": ["approved", "partially_approved"]},
    }).sort("created_at", -1).to_list(20)
 
    todos = []
    for doc in docs:
        doc_id = str(doc["_id"])
        for r in doc.get("recommendations", []):
            if r.get("status") == "approved":
                todos.append({
                    **r,
                    "report_id":    doc_id,
                    "customer_name":doc.get("customer_name"),
                    "report_date":  doc.get("created_at","")[:10],
                })
    return todos
 
 
@router.post("/todos/done")
async def mark_todo_done(body: dict, user=Depends(get_current_user)):
    """
    Mark a todo as done. Saves the outcome text.
    report_id, recommendation_id, outcome (required)
    """
    db     = get_db()
    oid    = _oid(body.get("report_id", ""))
    rec_id = body.get("recommendation_id", "")
    outcome= body.get("outcome", "").strip()
 
    if not oid:   raise HTTPException(400, "Invalid report_id")
    if not outcome: raise HTTPException(400, "outcome is required")
 
    doc  = await db.recommendations.find_one({"_id": oid})
    if not doc: raise HTTPException(404, "Report not found")

    recs = doc.get("recommendations", [])
    now  = datetime.utcnow().isoformat()
    matched = None
    for r in recs:
        if r.get("id") == rec_id:
            r.update({
                "todo_status":    "done",
                "outcome":        outcome,
                "done_by":        user.get("name", ""),
                "done_at":        now,
            })
            matched = r

    if not matched:
        raise HTTPException(404, "Recommendation not found")
 
    await db.recommendations.update_one({"_id": oid}, {"$set": {"recommendations": recs}})

    customer = await _customer_for_report(db, doc)
    try:
        await save_recommendation_hindsight(_hindsight_record(
            doc,
            customer,
            matched,
            "approved",
            user.get("name", ""),
            now,
        ))
    except Exception as exc:
        raise HTTPException(
            503,
            detail=(
                "The todo outcome was saved to recommendations, but hindsight persistence failed. "
                "Retry submitting the outcome; writes are idempotent. "
                f"Error: {exc}"
            ),
        ) from exc
    return {"status": "done", "outcome": outcome}
 
 
@router.post("/todos/cancel")
async def cancel_todo(body: dict, user=Depends(get_current_user)):
    """
    Cancel a todo. Saves the cancellation reason.
    report_id, recommendation_id, reason (required)
    """
    db     = get_db()
    oid    = _oid(body.get("report_id", ""))
    rec_id = body.get("recommendation_id", "")
    reason = body.get("reason", "").strip()
 
    if not oid:    raise HTTPException(400, "Invalid report_id")
    if not reason: raise HTTPException(400, "reason is required")
 
    doc  = await db.recommendations.find_one({"_id": oid})
    if not doc: raise HTTPException(404, "Report not found")
 
    recs = doc.get("recommendations", [])
    now  = datetime.utcnow().isoformat()
    for r in recs:
        if r.get("id") == rec_id:
            r.update({
                "todo_status":        "cancelled",
                "cancellation_reason": reason,
                "cancelled_by":       user.get("name", ""),
                "cancelled_at":       now,
            })
 
    await db.recommendations.update_one({"_id": oid}, {"$set": {"recommendations": recs}})
    return {"status": "cancelled", "reason": reason}