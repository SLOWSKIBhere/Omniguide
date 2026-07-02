"""
OmniGuide v2.0.0 — Telemetry
Firestore logging with robust error handling.
Never blocks the main pipeline.
"""
import logging
from datetime import datetime, timezone

logger = logging.getLogger("omniguide.telemetry")

# Lazy init — don't crash if Firestore isn't available
_db = None

def _get_db():
    global _db
    if _db is None:
        try:
            from google.cloud import firestore
            _db = firestore.AsyncClient(project="omniguide-hackathon")
        except Exception as e:
            logger.warning("Firestore unavailable: %s — telemetry will be skipped", e)
    return _db


async def log_interaction(
    session_id: str,
    user_query: str,
    observer_output: str,
    guide_response: str,
    token_count: int,
    latency_ms: float
):
    """Logs interaction to Firestore. Never raises — failures are logged only."""
    try:
        db = _get_db()
        if db is None:
            return  # Firestore not available, skip silently

        await db.collection("agent_telemetry").document().set({
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_query": user_query,
            "observer_output": observer_output,
            "guide_response": guide_response,
            "token_count": token_count,
            "latency_ms": round(latency_ms, 2),
            "version": "2.0.0",
        })
        logger.info("[TELEMETRY] Logged | latency=%.0fms tokens=%d", latency_ms, token_count)

    except Exception as e:
        logger.warning("[TELEMETRY] Skipped (non-fatal): %s", e)
