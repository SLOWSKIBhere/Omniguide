# OmniGuide — Provider-Independent Screen Co-Pilot

> A real-time agent that observes a shared screen, extracts evidence, classifies intent, and returns a verified answer.

## Live endpoints

- Frontend: `https://slowskibhere.github.io/Omniguide/omniguide.html`
- Backend health: `https://omniguide-backend-973581476293.us-central1.run.app/health`

## Architecture — v2.1

```text
Browser screen capture + question
              ↓
        FastAPI /ask
              ↓
      Provider Router
  ┌───────────┴────────────┐
  │ OpenAI-compatible      │ primary/default seam
  │ Gemini                 │ optional fallback
  └───────────┬────────────┘
              ↓
   Vision + OCR in parallel
              ↓
 Grounding/Evidence Contract
      no evidence → HTTP 503
              ↓
 Deterministic Intent Router
              ↓
   Grounded Reasoning Agent
              ↓
 Verified Response + Traces
              ↓
 Optional Firestore telemetry
```

### Core invariant

OmniGuide does not return a successful screen-aware answer unless at least one observation stage produces usable evidence and the reasoning stage completes. When providers are unavailable or exhausted, `/ask` returns HTTP 503 with `verified: false` instead of generating a plausible but ungrounded response.

## Provider independence

The agents no longer import or instantiate Gemini directly. They depend on `ProviderRouter`, which supports ordered failover.

Default order:

```env
MODEL_PROVIDER_ORDER=openai_compatible,gemini
```

The OpenAI-compatible adapter works with hosted or local endpoints that implement the Chat Completions multimodal contract, including OpenRouter, OpenAI-compatible gateways, LM Studio, and vLLM deployments. The selected vision model must support image input.

Gemini remains available as an optional fallback and is not required by the application architecture.

## Agent pipeline

1. **Vision Agent** — produces structured application, task, focus, visible-text, and confidence evidence.
2. **OCR Agent** — independently extracts literal screen text.
3. **Context Builder** — merges successful evidence and sets the `grounded` contract.
4. **Intent Router** — deterministic by default, avoiding a model call for simple classification.
5. **Reasoning Agent** — runs only when screen grounding succeeded.
6. **Response Agent** — marks output as verified only when both grounding and reasoning completed.

Every result includes `run_id`, `status`, `grounded`, `verified`, provider/model provenance, agent chain, errors, and stage traces.

## Local setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8080
```

Configure at least one complete multimodal provider:

```env
OPENAI_COMPAT_BASE_URL=https://openrouter.ai/api/v1
OPENAI_COMPAT_API_KEY=your_key
OPENAI_COMPAT_VISION_MODEL=openai/gpt-4o-mini
OPENAI_COMPAT_TEXT_MODEL=openai/gpt-4o-mini
```

Or use Gemini only:

```env
MODEL_PROVIDER_ORDER=gemini
GEMINI_API_KEY=your_key
GEMINI_VISION_MODEL=gemini-2.0-flash
GEMINI_TEXT_MODEL=gemini-2.0-flash
```

No real keys belong in Git. `.env`, `.env.yaml`, and key files are ignored.

## Tests

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

The regression suite verifies:

- provider failover
- deterministic intent routing without a model call
- failed observation stages do not count as grounding
- reasoning cannot run without screen evidence
- unverified model-like text is discarded
- `/ask` returns HTTP 503 rather than a fake answer when no provider is configured

## Health response

`GET /health` reports provider readiness without exposing keys:

```json
{
  "status": "OmniGuide is live",
  "version": "2.1.0",
  "model_available": true,
  "vision_available": true,
  "text_available": true,
  "provider_order": ["openai_compatible", "gemini"],
  "execution_contract": "screen_grounded_only"
}
```

## API

### `POST /ask`

Request:

```json
{"image": "<base64 JPEG>", "query": "How do I fix this error?"}
```

Verified success returns HTTP 200. Missing grounding or a failed reasoning provider returns HTTP 503 with `verified: false` and a safe error message.

### `WS /ws`

Backward-compatible WebSocket transport using the same pipeline and response contract.

### `GET /r`

Validated HTTP/HTTPS click-tracking redirect with optional non-blocking Firestore logging.

## Deployment

Cloud Run needs the provider variables from `.env.example`. For an OpenAI-compatible primary with Gemini fallback, set both sets of keys and keep:

```env
MODEL_PROVIDER_ORDER=openai_compatible,gemini
```

For a local provider, point `OPENAI_COMPAT_BASE_URL` at the reachable gateway. A Cloud Run container cannot reach an LM Studio server bound only to a developer laptop’s `127.0.0.1`.
