import os
import json
import uuid
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent import run_agent_pipeline
from telemetry import log_interaction


app = FastAPI(title="OmniGuide API", version="1.1.0")

# Allow all origins in dev — tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Cloud Run uses this endpoint to verify the container is alive."""
    gemini_key_set = bool(os.environ.get("GEMINI_API_KEY"))
    return JSONResponse({
        "status": "OmniGuide is live",
        "version": "1.1.0",
        "gemini_key_configured": gemini_key_set,
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket handler.
    Expects JSON: { "image": "<base64 JPEG>", "query": "<text>" }
    Returns JSON: { "response": "...", "context": "...", "latency_ms": 220.4 }
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    print(f"[WS] Session connected: {session_id}")

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON payload"}))
                continue

            image_base64 = data.get("image", "").strip()
            user_query = data.get("query", "").strip()

            if not image_base64:
                await websocket.send_text(json.dumps({"error": "Missing or empty image field"}))
                continue
            if not user_query:
                await websocket.send_text(json.dumps({"error": "Missing or empty query field"}))
                continue

            print(f"[WS] Query: '{user_query[:60]}'")

            result = await run_agent_pipeline(image_base64, user_query)

            asyncio.create_task(log_interaction(
                session_id=session_id,
                user_query=user_query,
                observer_output=result["observer_context"],
                guide_response=result["guide_response"],
                token_count=result["total_tokens"],
                latency_ms=result["latency_ms"]
            ))

            await websocket.send_text(json.dumps({
                "response": result["guide_response"],
                "context": result["observer_context"],
                "latency_ms": result["latency_ms"],
                "tokens": result["total_tokens"]
            }))

    except WebSocketDisconnect:
        print(f"[WS] Session disconnected: {session_id}")
    except Exception as e:
        print(f"[WS ERROR] {e}")
