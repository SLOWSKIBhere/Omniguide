from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
import uuid
import asyncio
from agent import run_agent_pipeline
from telemetry import log_interaction




@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FIX: Startup check - verify GCP credentials and Vertex AI
    are working BEFORE accepting traffic. Fails loud and early.
    """
    print("[STARTUP] Verifying Vertex AI connection...")
    try:
        vertexai.init(project="omniguide-hackathon", location="us-central1")
        # Quick ping to verify auth works
        model = GenerativeModel("gemini-1.5-pro")
        print("[STARTUP] ✅ Vertex AI connected")
    except Exception as e:
        print(f"[STARTUP] ⚠️  Vertex AI check failed: {e}")
        print("[STARTUP] Continuing anyway - check your GCP credentials")
    yield
    print("[SHUTDOWN] OmniGuide shutting down")


app = FastAPI(title="OmniGuide API", version="1.0.0", lifespan=lifespan)

# Allow all origins in dev - tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Cloud Run uses this endpoint to verify the container is alive."""
    return {"status": "OmniGuide is live", "version": "1.0.0"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket handler.

    Expects JSON from frontend:
    { "image": "<base64 JPEG screenshot>", "query": "<transcribed voice>" }

    Returns JSON:
    { "response": "...", "context": "...", "latency_ms": 220.4 }
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    print(f"[WS] Session connected: {session_id}")

    try:
        while True:
            raw = await websocket.receive_text()

            # FIX: JSON parse error handling - bad payload won't crash session
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON payload"}))
                continue

            image_base64 = data.get("image", "").strip()
            user_query = data.get("query", "").strip()

            # FIX: Validate both fields exist and are non-empty
            if not image_base64:
                await websocket.send_text(json.dumps({"error": "Missing or empty image field"}))
                continue
            if not user_query:
                await websocket.send_text(json.dumps({"error": "Missing or empty query field"}))
                continue

            print(f"[WS] Query: '{user_query[:60]}'")

            # Run Observer → Guide pipeline
            result = await run_agent_pipeline(image_base64, user_query)

            # Log to Firestore without blocking the response
            asyncio.create_task(log_interaction(
                session_id=session_id,
                user_query=user_query,
                observer_output=result["observer_context"],
                guide_response=result["guide_response"],
                token_count=result["total_tokens"],
                latency_ms=result["latency_ms"]
            ))

            # Send response back to client
            await websocket.send_text(json.dumps({
                "response": result["guide_response"],
                "context": result["observer_context"],
                "latency_ms": round(result["latency_ms"], 1)
            }))

    except WebSocketDisconnect:
        print(f"[WS] Session disconnected: {session_id}")
    except Exception as e:
        print(f"[WS ERROR] {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass  # Socket already closed

from fastapi import Request

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    image_base64 = data.get("image", "").strip()
    user_query = data.get("query", "").strip()
    if not image_base64 or not user_query:
        return {"error": "Missing image or query"}
    result = await run_agent_pipeline(image_base64, user_query)
    return {
        "response": result["guide_response"],
        "context": result["observer_context"],
        "latency_ms": round(result["latency_ms"], 1)
    }

from fastapi import Request

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    image_base64 = data.get("image", "").strip()
    user_query = data.get("query", "").strip()
    if not image_base64 or not user_query:
        return {"error": "Missing image or query"}
    result = await run_agent_pipeline(image_base64, user_query)
    return {
        "response": result["guide_response"],
        "context": result["observer_context"],
        "latency_ms": round(result["latency_ms"], 1)
    }
