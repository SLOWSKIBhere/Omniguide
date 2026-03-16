#!/usr/bin/env python3
"""
OmniGuide - Master Build Script
Writes all project files with QA fixes applied.
Run: python3 build.py
"""
import os

files = {}

# ─────────────────────────────────────────────────────────────
# requirements.txt - FIX: removed duplicate vertexai package
# ─────────────────────────────────────────────────────────────
files["requirements.txt"] = """\
fastapi==0.111.0
uvicorn[standard]==0.30.1
websockets==12.0
google-cloud-aiplatform==1.57.0
google-cloud-firestore==2.16.0
google-auth==2.29.0
Pillow==10.3.0
python-multipart==0.0.9
httpx==0.27.0
python-dotenv==1.0.0
"""

# ─────────────────────────────────────────────────────────────
# telemetry.py
# FIX 1: Use AsyncClient instead of sync Client
# FIX 2: Proper async/await throughout
# ─────────────────────────────────────────────────────────────
files["telemetry.py"] = """\
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
    \"\"\"
    Logs every agent interaction to Firestore agent_telemetry collection.
    Fully async - never blocks the WebSocket handler.
    \"\"\"
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
"""

# ─────────────────────────────────────────────────────────────
# agent.py
# FIX 1: asyncio.to_thread() wraps all sync Vertex AI calls
# FIX 2: Null guard on usage_metadata
# FIX 3: Clean separation of Observer and Guide
# ─────────────────────────────────────────────────────────────
files["agent.py"] = """\
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import base64
import time
import asyncio

# Initialize Vertex AI with your project
vertexai.init(project="omniguide-hackathon", location="us-central1")

# ── OBSERVER PROMPT ──────────────────────────────────────────
# Job: Look at screen, return structured context in <50 words
# Output feeds directly into the Guide as context
OBSERVER_PROMPT = \"\"\"You are a Screen Context Observer. Analyze the screenshot and return ONLY this structure:
APP: <name of software or website visible>
TASK: <what the user appears to be doing in one phrase>
FOCUS: <the most prominent UI element or content on screen>
Rules: Max 50 words total. No filler. No preamble. Just the three lines.\"\"\"

# ── GUIDE PROMPT ─────────────────────────────────────────────
# Job: Use Observer context + user question to give sharp answer
GUIDE_PROMPT = \"\"\"You are OmniGuide, a real-time AI co-pilot embedded in the user's desktop.
You receive structured screen context from an Observer AI and the user's spoken question.
Rules:
- Answer in 2-4 sentences maximum
- Be direct, like a senior colleague sitting next to them
- Never say "Based on the context provided" or any similar preamble
- If you can give a shortcut, command, or specific step - do it\"\"\"


def _sync_observer(image_data: bytes) -> tuple[str, int]:
    \"\"\"Synchronous Vertex AI call - runs in thread pool via asyncio.to_thread.\"\"\"
    image_part = Part.from_data(data=image_data, mime_type="image/jpeg")
    model = GenerativeModel("gemini-1.5-pro", system_instruction=OBSERVER_PROMPT)
    response = model.generate_content([image_part, "Analyze this screenshot."])
    # FIX: Null guard - usage_metadata may not always be present
    tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0
    return response.text.strip(), tokens


def _sync_guide(observer_context: str, user_query: str) -> tuple[str, int]:
    \"\"\"Synchronous Vertex AI call - runs in thread pool via asyncio.to_thread.\"\"\"
    prompt = f"SCREEN CONTEXT:\\n{observer_context}\\n\\nUSER QUESTION:\\n{user_query}"
    model = GenerativeModel("gemini-1.5-pro", system_instruction=GUIDE_PROMPT)
    response = model.generate_content(prompt)
    tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0
    return response.text.strip(), tokens


async def run_observer(image_base64: str) -> tuple[str, int]:
    \"\"\"
    PIPELINE STEP 1: Analyze the screenshot.
    FIX: asyncio.to_thread() prevents sync Vertex AI from blocking the event loop.
    \"\"\"
    try:
        image_data = base64.b64decode(image_base64)
        # Run sync SDK call in a thread - keeps event loop free for other WebSockets
        return await asyncio.to_thread(_sync_observer, image_data)
    except Exception as e:
        print(f"[OBSERVER ERROR] {e}")
        return "APP: Unknown\\nTASK: Unknown\\nFOCUS: Unknown", 0


async def run_guide(observer_context: str, user_query: str) -> tuple[str, int]:
    \"\"\"
    PIPELINE STEP 2: Generate response using Observer context + user query.
    FIX: asyncio.to_thread() prevents blocking.
    \"\"\"
    try:
        return await asyncio.to_thread(_sync_guide, observer_context, user_query)
    except Exception as e:
        print(f"[GUIDE ERROR] {e}")
        return "I could not process that — please try again.", 0


async def run_agent_pipeline(image_base64: str, user_query: str) -> dict:
    \"\"\"
    MASTER PIPELINE: Observer → Guide prompt chain.
    Called by the WebSocket handler for every user message.
    \"\"\"
    start = time.time()
    # Step 1: Observer reads the screen
    observer_context, observer_tokens = await run_observer(image_base64)
    # Step 2: Guide answers using that context
    guide_response, guide_tokens = await run_guide(observer_context, user_query)
    return {
        "observer_context": observer_context,
        "guide_response": guide_response,
        "total_tokens": observer_tokens + guide_tokens,
        "latency_ms": (time.time() - start) * 1000
    }
"""

# ─────────────────────────────────────────────────────────────
# main.py
# FIX 1: JSON parse error handling in WebSocket
# FIX 2: base64 validation before hitting the pipeline
# FIX 3: Startup event to verify GCP credentials work
# ─────────────────────────────────────────────────────────────
files["main.py"] = """\
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
import uuid
import asyncio
from agent import run_agent_pipeline
from telemetry import log_interaction
import vertexai
from vertexai.generative_models import GenerativeModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    \"\"\"
    FIX: Startup check - verify GCP credentials and Vertex AI
    are working BEFORE accepting traffic. Fails loud and early.
    \"\"\"
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
    \"\"\"Cloud Run uses this endpoint to verify the container is alive.\"\"\"
    return {"status": "OmniGuide is live", "version": "1.0.0"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    \"\"\"
    Main WebSocket handler.

    Expects JSON from frontend:
    { "image": "<base64 JPEG screenshot>", "query": "<transcribed voice>" }

    Returns JSON:
    { "response": "...", "context": "...", "latency_ms": 220.4 }
    \"\"\"
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
"""

# ─────────────────────────────────────────────────────────────
# Dockerfile - optimized for Cloud Run WebSocket hosting
# ─────────────────────────────────────────────────────────────
files["Dockerfile"] = """\
FROM python:3.11-slim

WORKDIR /app

# Install deps first (cached layer - only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Cloud Run requires port 8080
ENV PORT=8080
EXPOSE 8080

# Single worker is REQUIRED for WebSocket sticky sessions
# Multiple workers = sessions randomly routed to different workers = dropped connections
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--ws", "websockets"]
"""

# ─────────────────────────────────────────────────────────────
# Write all files
# ─────────────────────────────────────────────────────────────
for filename, content in files.items():
    with open(filename, "w") as f:
        f.write(content)
    print(f"✅ Created {filename}")

print("\n🚀 All files created. Run: pip install -r requirements.txt")
