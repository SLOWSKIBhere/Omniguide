"""Non-blocking optional Firestore telemetry."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("omniguide.telemetry")
_db = None
_db_attempted = False


def _get_db():
    global _db, _db_attempted
    if _db_attempted:
        return _db
    _db_attempted = True
    try:
        from google.cloud import firestore
        _db = firestore.AsyncClient(project=os.getenv("GCP_PROJECT_ID", "omniguide-hackathon"))
    except Exception as exc:
        logger.warning("Firestore unavailable; telemetry disabled: %s", exc)
        _db = None
    return _db


async def log_interaction(
    session_id: str,
    user_query: str,
    observer_output: str,
    guide_response: str,
    token_count: int,
    latency_ms: float,
    metadata: Optional[dict[str, Any]] = None,
):
    try:
        db = _get_db()
        if db is None:
            return
        await db.collection("agent_telemetry").document().set({
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_query": user_query,
            "observer_output": observer_output,
            "guide_response": guide_response,
            "token_count": token_count,
            "latency_ms": round(latency_ms, 2),
            "metadata": metadata or {},
            "version": "2.1.0",
        })
    except Exception as exc:
        logger.warning("Telemetry skipped: %s", exc)
