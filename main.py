"""
OmniGuide v2.0.0 — FastAPI Server
Multi-agent pipeline: Vision + OCR → Context → Intent → Reasoning → Response
"""
import os
import json
import uuid
import asyncio
import logging
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from google import genai

from agents import (
    VisionAgent, OCRAgent, ContextBuilder,
    IntentRouter, ReasoningAgent, ResponseAgent
)
from models import ScreenContext, IntentClassification, AskRequest
from telemetry import log_interaction

logger = logging.getLogger("omniguide")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)

app = FastAPI(title="OmniGuide API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize Gemini client + agents ──
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

vision_agent = VisionAgent(gemini_client)
ocr_agent = OCRAgent(gemini_client)
context_builder = ContextBuilder(vision_agent, ocr_agent)
intent_router = IntentRouter(gemini_client)
reasoning_agent = ReasoningAgent(gemini_client)
response_agent = ResponseAgent()


async def run_pipeline(image_base64: str, user_query: str) -> dict:
    """
    Full multi-agent pipeline:
    1. Vision + OCR (parallel) → Context Builder
    2. Intent Router (uses context)
    3. Reasoning Agent (uses context + intent)
    4. Response Agent (formats output)
    """
    start = time.time()
    all_tokens = 0
    all_errors = []
    full_chain = []

    # ── Stage 1: Context Building (Vision + OCR in parallel) ──
    logger.info("Pipeline start: query='%s'", user_query[:60])
    context, ctx_tokens, ctx_errors, ctx_chain = await context_builder.build(image_base64)
    all_tokens += ctx_tokens
    all_errors.extend(ctx_errors)
    full_chain.extend(ctx_chain)

    # ── Stage 2: Intent Classification ──
    intent, intent_tokens, intent_err = await intent_router.classify(user_query, context)
    all_tokens += intent_tokens
    if intent_err:
        all_errors.append(f"intent: {intent_err}")
    else:
        full_chain.append("intent")

    # ── Stage 3: Reasoning ──
    response_text, reason_tokens, reason_err = await reasoning_agent.reason(user_query, context, intent)
    all_tokens += reason_tokens
    if reason_err:
        all_errors.append(f"reasoning: {reason_err}")
    else:
        full_chain.append("reasoning")

    # ── Stage 4: Response Formatting ──
    latency_ms = (time.time() - start) * 1000

    result = response_agent.build(
        response_text=response_text,
        context=context,
        intent=intent,
        latency_ms=latency_ms,
        tokens=all_tokens,
        errors=all_errors,
        agent_chain=full_chain
    )

    logger.info(
        "Pipeline complete: chain=%s tokens=%d latency=%.0fms errors=%d",
        full_chain, all_tokens, latency_ms, len(all_errors)
    )

    return result


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "OmniGuide is live",
        "version": "2.0.0",
        "gemini_key_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "agents": ["vision", "ocr", "context", "intent", "reasoning", "response"],
    })


@app.post("/ask")
async def ask_endpoint(req: AskRequest):
    """REST endpoint — preferred for Cloud Run (more reliable than WebSocket)."""
    session_id = str(uuid.uuid4())

    image_base64 = req.image.strip()
    user_query = req.query.strip()

    if not image_base64:
        return JSONResponse({"error": "Missing or empty image field"}, status_code=400)
    if not user_query:
        return JSONResponse({"error": "Missing or empty query field"}, status_code=400)

    result = await run_pipeline(image_base64, user_query)

    # Fire-and-forget telemetry
    asyncio.create_task(log_interaction(
        session_id=session_id,
        user_query=user_query,
        observer_output=result["context"],
        guide_response=result["response"],
        token_count=result["tokens"],
        latency_ms=result["latency_ms"]
    ))

    return JSONResponse(result)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint — kept for backward compatibility."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info("[WS] Session: %s", session_id)

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

            result = await run_pipeline(image_base64, user_query)

            asyncio.create_task(log_interaction(
                session_id=session_id,
                user_query=user_query,
                observer_output=result["context"],
                guide_response=result["response"],
                token_count=result["tokens"],
                latency_ms=result["latency_ms"]
            ))

            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        logger.info("[WS] Disconnected: %s", session_id)
    except Exception as e:
        logger.error("[WS ERROR] %s", e)


@app.get("/r")
async def redirect_endpoint(id: str, url: str):
    """
    Click-tracking redirect for financial scraper articles.
    Logs article_hash + url + timestamp to Firestore article_clicks collection.
    Returns HTTP 302 redirect to the real article URL.
    Reuses telemetry._get_db() — no new Firestore client, no new imports beyond stdlib.
    """
    from urllib.parse import unquote
    from fastapi.responses import RedirectResponse
    import telemetry

    # Decode the URL (percent-encoded before being passed as query param)
    real_url = unquote(url)

    # Fire-and-forget Firestore log — never blocks the redirect
    try:
        db = telemetry._get_db()
        if db is not None:
            from datetime import datetime, timezone
            asyncio.create_task(
                db.collection("article_clicks").document().set({
                    "article_hash": id,
                    "url": real_url,
                    "clicked_at": datetime.now(timezone.utc).isoformat(),
                    "source": "whatsapp",
                })
            )
    except Exception as e:
        logger.warning("[REDIRECT] Firestore log failed (non-fatal): %s", e)

    return RedirectResponse(url=real_url, status_code=302)
