# OmniGuide — Real-Time Multimodal AI Screen Co-Pilot

> An AI agent that sees your screen, hears your question, and tells you exactly what to do — in real time.

## 🔗 Live Demo
**Frontend:** https://slowskibhere.github.io/Omniguide/omniguide.html
**Backend Health Check:** https://omniguide-backend-973581476293.us-central1.run.app/health

## 🧠 Architecture (v2.0.0 — Multi-Agent Pipeline)

```text
Browser (Screen Capture + Voice/Text Input)
    ↓ POST /ask (base64 JPEG + query string)
Cloud Run — FastAPI Backend
    ↓
    ├─ Vision Agent (Gemini 2.0 Flash)
    │   → Structured JSON: { app, task, focus, confidence }
    ├─ OCR Agent (Gemini 2.0 Flash, parallel)
    │   → Extracts visible text from screenshot
    ↓
Context Builder
    → Merges Vision + OCR into unified ScreenContext
    ↓
Intent Router (Gemini structured JSON)
    → Classifies: debug_help | how_to | what_is | navigation | code_review | general
    → Extracts entities + reasoning hint
    ↓
Reasoning Agent (Gemini 2.0 Flash)
    → Intent-specific prompt strategy
    → Retry logic (2 retries with backoff)
    → Graceful fallback with visible text if all retries fail
    ↓
Response Agent
    → Formats output with metadata (intent, confidence, agent chain, errors)
    ↓
Firestore — agent_telemetry (async, non-blocking)
    ↓
Browser — displays response with agent chain badges
```

## 🛠 Stack
- AI: Google GenAI SDK + Gemini 2.0 Flash
- Backend: Python FastAPI on Google Cloud Run
- Database: Google Firestore (async, non-blocking telemetry)
- Frontend: Single HTML file (no frameworks, native Web APIs)
- Communication: REST POST /ask (preferred) + WebSocket /ws (backward compat)

## 📁 Project Structure

```
main.py          — FastAPI server, pipeline orchestration
models.py        — Pydantic data models (ScreenContext, IntentClassification, AgentResponse)
agents/
  __init__.py    — Package exports
  vision.py      — Vision Agent (screenshot → structured JSON context)
  ocr.py         — OCR Agent (screenshot → extracted text)
  context.py     — Context Builder (merges vision + OCR in parallel)
  intent.py      — Intent Router (query → classified intent with retries)
  reasoning.py   — Reasoning Agent (intent-specific response generation)
  response.py    — Response Agent (output formatting + metadata)
telemetry.py     — Firestore logging (async, never blocks pipeline)
omniguide.html   — Frontend (screen capture, voice, text, agent chain display)
Dockerfile       — Cloud Run container config
requirements.txt — Python dependencies
```

## 🔑 Key Improvements (v1.x → v2.0.0)

1. **Modular multi-agent architecture** — 6 agents with clear interfaces
2. **Structured JSON output** — Replaces brittle "APP: X / TASK: Y / FOCUS: Z" text parser
3. **Graceful fallbacks** — Never returns all-"Unknown"; uses "unidentified" + confidence scores
4. **Parallel execution** — Vision + OCR run concurrently via asyncio.gather
5. **Retry logic** — Intent and Reasoning agents retry with backoff before falling back
6. **Intent-aware prompting** — 6 intent types with specialized system prompts
7. **REST + WebSocket** — REST /ask preferred for Cloud Run reliability
8. **Rich telemetry** — Agent chain, error list, confidence scores in every response
9. **Non-blocking telemetry** — Firestore failures never crash the pipeline

## 💻 Run Locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
uvicorn main:app --host 0.0.0.0 --port 8080
# Serve frontend separately:
python -m http.server 8000
# Open: http://localhost:8000/omniguide.html
```

## ☁️ Deploy to Cloud Run

```bash
gcloud services enable aiplatform.googleapis.com run.googleapis.com firestore.googleapis.com

gcloud run deploy omniguide-backend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --session-affinity \
  --set-env-vars GEMINI_API_KEY=your_key_here
```

## 📡 API Endpoints

### GET /health
Returns backend status, version, and configured agents.

### POST /ask
```json
Request: { "image": "<base64 JPEG>", "query": "How do I fix this error?" }
Response: {
  "response": "The TypeError on line 24 means...",
  "context": "APP: VS Code / TASK: debugging / FOCUS: error on line 24",
  "intent": "debug_help",
  "confidence": 0.85,
  "latency_ms": 1840.5,
  "tokens": 1240,
  "errors": [],
  "agent_chain": ["vision", "ocr", "intent", "reasoning"],
  "version": "2.0.0"
}
```

### WS /ws
WebSocket endpoint (backward compatible). Same response format.

## 🎯 Intent Types

| Type | Description | Example Query |
|------|-------------|---------------|
| debug_help | Fixing errors/bugs | "Why is this crashing?" |
| how_to | Learning to do something | "How do I add a navbar?" |
| what_is | Understanding concepts | "What does this error mean?" |
| navigation | Finding UI elements | "Where is the settings page?" |
| code_review | Improving code | "Review this function" |
| general | Anything else | "What should I do next?" |
