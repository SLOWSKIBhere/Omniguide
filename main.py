import os
import json
import uuid
import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent import run_agent_pipeline
from telemetry import log_interaction

logger = logging.getLogger("omniguide")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="OmniGuide API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    image: str  # base64 JPEG
    query: str


@app.get("/health")
async def health():
    gemini_key_set = bool(os.environ.get("GEMINI_API_KEY"))
    return JSONResponse({
        "status": "OmniGuide is live",
        "version": "1.2.0",
        "gemini_key_configured": gemini_key_set,
    })


@app.post("/ask")
async def ask_endpoint(req: AskRequest):
    """
    REST endpoint for screen + query processing.
    More reliable than WebSocket on Cloud Run.
    """
    session_id = str(uuid.uuid4())

    image_base64 = req.image.strip()
    user_query = req.query.strip()

    if not image_base64:
        return JSONResponse({"error": "Missing or empty image field"}, status_code=400)
    if not user_query:
        return JSONResponse({"error": "Missing or empty query field"}, status_code=400)

    result = await run_agent_pipeline(image_base64, user_query)

    asyncio.create_task(log_interaction(
        session_id=session_id,
        user_query=user_query,
        observer_output=result["observer_context"],
        guide_response=result["guide_response"],
        token_count=result["total_tokens"],
        latency_ms=result["latency_ms"]
    ))

    return JSONResponse({
        "response": result["guide_response"],
        "context": result["observer_context"],
        "latency_ms": result["latency_ms"],
        "tokens": result["total_tokens"]
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"[WS] Session: {session_id}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            image_base64 = data.get("image", "").strip()
            user_query = data.get("query", "").strip()

            if not image_base64 or not user_query:
                await websocket.send_text(json.dumps({"error": "Missing image or query"}))
                continue

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
        logger.info(f"[WS] Disconnected: {session_id}")
    except Exception as e:
        logger.error(f"[WS ERROR] {e}")
