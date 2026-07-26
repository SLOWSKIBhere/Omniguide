"""OmniGuide v2.1 — provider-independent, screen-grounded agent pipeline."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from urllib.parse import unquote, urlparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from agents import ContextBuilder, IntentRouter, OCRAgent, ReasoningAgent, ResponseAgent, VisionAgent
from models import AskRequest
from providers import ProviderRouter
from telemetry import log_interaction

logger = logging.getLogger("omniguide")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

VERSION = "2.1.0"
model_router = ProviderRouter()
vision_agent = VisionAgent(model_router)
ocr_agent = OCRAgent(model_router)
context_builder = ContextBuilder(vision_agent, ocr_agent)
intent_router = IntentRouter(model_router)
reasoning_agent = ReasoningAgent(model_router)
response_agent = ResponseAgent()

app = FastAPI(title="OmniGuide API", version=VERSION)
_default_origins = "https://slowskibhere.github.io,http://localhost:8000,http://127.0.0.1:8000"
allowed_origins = [item.strip() for item in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


async def run_pipeline(image_base64: str, user_query: str) -> dict:
    run_id = str(uuid.uuid4())
    started = time.perf_counter()
    tokens = 0
    errors: list[str] = []
    chain: list[str] = []
    traces: list[dict] = []

    context, ctx_tokens, ctx_errors, ctx_chain, ctx_traces = await context_builder.build(image_base64)
    tokens += ctx_tokens
    errors.extend(ctx_errors)
    chain.extend(ctx_chain)
    traces.extend(ctx_traces)

    intent, intent_tokens, intent_error, intent_trace = await intent_router.classify(user_query, context)
    tokens += intent_tokens
    traces.append(intent_trace)
    if intent_trace.get("status") in {"ok", "fallback"}:
        chain.append("intent")
    if intent_error:
        errors.append(f"intent: {intent_error}")

    response_text = ""
    if context.grounded:
        response_text, reason_tokens, reason_error, reason_trace = await reasoning_agent.reason(
            user_query, context, intent
        )
        tokens += reason_tokens
        traces.append(reason_trace)
        if reason_error:
            errors.append(f"reasoning: {reason_error}")
        else:
            chain.append("reasoning")
    else:
        traces.append({
            "stage": "reasoning",
            "status": "blocked",
            "tokens": 0,
            "error": "Skipped because screen grounding failed",
        })
        errors.append("reasoning: skipped because screen grounding failed")

    latency_ms = (time.perf_counter() - started) * 1000
    return response_agent.build(
        run_id=run_id,
        response_text=response_text,
        context=context,
        intent=intent,
        latency_ms=latency_ms,
        tokens=tokens,
        errors=errors,
        agent_chain=chain,
        traces=traces,
    )


@app.get("/health")
async def health():
    vision_available = model_router.has_available_provider(vision=True)
    text_available = model_router.has_available_provider(vision=False)
    providers = model_router.describe()
    provider_ready = vision_available and text_available
    gemini_configured = any(
        item["name"] == "gemini" and item.get("configured", False)
        for item in providers
    )
    return JSONResponse({
        "status": "OmniGuide is live",
        "version": VERSION,
        "model_available": provider_ready,
        "provider_ready": provider_ready,
        "gemini_key_configured": gemini_configured,
        "vision_available": vision_available,
        "text_available": text_available,
        "providers": providers,
        "provider_order": [item["name"] for item in providers],
        "execution_contract": "screen_grounded_only",
        "agents": ["vision", "ocr", "context", "intent", "reasoning", "response"],
    })


async def _record(session_id: str, user_query: str, result: dict) -> None:
    await log_interaction(
        session_id=session_id,
        user_query=user_query,
        observer_output=result["context"],
        guide_response=result["response"],
        token_count=result["tokens"],
        latency_ms=result["latency_ms"],
        metadata={
            "run_id": result["run_id"],
            "status": result["status"],
            "grounded": result["grounded"],
            "verified": result["verified"],
            "provider": result.get("provider"),
            "model": result.get("model"),
            "agent_chain": result["agent_chain"],
            "errors": result["errors"],
        },
    )


@app.post("/ask")
async def ask_endpoint(req: AskRequest):
    session_id = str(uuid.uuid4())
    image_base64 = req.image.strip()
    user_query = req.query.strip()
    if not image_base64:
        return JSONResponse({"error": "Missing or empty image field"}, status_code=400)
    if not user_query:
        return JSONResponse({"error": "Missing or empty query field"}, status_code=400)

    result = await run_pipeline(image_base64, user_query)
    asyncio.create_task(_record(session_id, user_query, result))
    return JSONResponse(result, status_code=200 if result["verified"] else 503)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue
            if not isinstance(data, dict):
                await websocket.send_text(json.dumps({"error": "JSON payload must be an object"}))
                continue
            image_base64 = str(data.get("image") or "").strip()
            user_query = str(data.get("query") or "").strip()
            if not image_base64 or not user_query:
                await websocket.send_text(json.dumps({"error": "Missing image or query"}))
                continue
            result = await run_pipeline(image_base64, user_query)
            asyncio.create_task(_record(session_id, user_query, result))
            await websocket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", session_id)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        try:
            await websocket.send_text(json.dumps({"error": "Pipeline request failed"}))
        except Exception:
            pass


@app.get("/r")
async def redirect_endpoint(id: str, url: str):
    real_url = unquote(url)
    parsed = urlparse(real_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return JSONResponse({"error": "Invalid redirect URL"}, status_code=400)
    try:
        import telemetry
        db = telemetry._get_db()
        if db is not None:
            from datetime import datetime, timezone
            asyncio.create_task(db.collection("article_clicks").document().set({
                "article_hash": id,
                "url": real_url,
                "clicked_at": datetime.now(timezone.utc).isoformat(),
                "source": "whatsapp",
            }))
    except Exception as exc:
        logger.warning("Redirect telemetry failed: %s", exc)
    return RedirectResponse(url=real_url, status_code=302)
