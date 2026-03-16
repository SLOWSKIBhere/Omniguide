from google.cloud import firestore
from datetime import datetime, timezone

# FIX: Use AsyncClient so Firestore never blocks the event loop
db = firestore.AsyncClient(project="omniguide-hackathon")

async def log_interaction(
    session_id: str,
    user_query: str,
    observer_output: str,
    guide_response: str,
    token_count: int,
    latency_ms: float
):
    """
    Logs every agent interaction to Firestore agent_telemetry collection.
    Fully async - never blocks the WebSocket handler.
    """
    try:
        await db.collection("agent_telemetry").document().set({
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_query": user_query,
            "observer_output": observer_output,
            "guide_response": guide_response,
            "token_count": token_count,
            "latency_ms": round(latency_ms, 2),
        })
        print(f"[TELEMETRY] Logged | latency={latency_ms:.0f}ms | tokens={token_count}")
    except Exception as e:
        # Never let telemetry failures crash the main pipeline
        print(f"[TELEMETRY ERROR] {e}")
