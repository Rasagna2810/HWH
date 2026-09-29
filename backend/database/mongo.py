from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from core.database import get_db


def get_hindsight_collection():
    return get_db()["hindsight_logs"]


async def save_hindsight_entry(entry: dict[str, Any]) -> str:
    collection = get_hindsight_collection()
    payload = {
        **entry,
        "created_at": entry.get("created_at") or datetime.utcnow().isoformat(),
    }

    # A stable key makes API retries safe and lets a later outcome enrich the
    # original approval/rejection decision rather than create a duplicate.
    source_key = payload.get("source_key")
    if source_key:
        result = await collection.update_one(
            {"source_key": source_key},
            {"$set": payload},
            upsert=True,
        )
        if result.upserted_id:
            return str(result.upserted_id)
        existing = await collection.find_one({"source_key": source_key}, {"_id": 1})
        if existing:
            return str(existing["_id"])

    result = await collection.insert_one(payload)
    return str(result.inserted_id)


async def get_hindsight_matches(
    industry: str = "",
    country: str = "",
    persona: str = "",
    customer_id: str = "",
    limit: int = 5,
):
    collection = get_hindsight_collection()
    query: dict[str, Any] = {}

    if industry:
        query["industry"] = {"$regex": f"^{re.escape(industry.strip())}$", "$options": "i"}
    if country:
        query["country"] = {"$regex": f"^{re.escape(country.strip())}$", "$options": "i"}
    if persona:
        query["target_persona"] = {"$regex": f"^{re.escape(persona.strip())}$", "$options": "i"}
    if customer_id:
        query["customer_id"] = customer_id

    return await collection.find(query).sort("created_at", -1).limit(limit).to_list(length=limit)


hindsight_col = get_hindsight_collection()