import os
import base64
import time
import asyncio
import io

from google import genai
import PIL.Image

# Initialize client from environment variable — never hardcode keys
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

OBSERVER_PROMPT = (
    "You are a Screen Context Observer. Analyze the screenshot and return ONLY:\n"
    "APP: <software visible>\n"
    "TASK: <what user is doing>\n"
    "FOCUS: <main UI element>\n"
    "Max 50 words."
)

GUIDE_PROMPT = (
    "You are OmniGuide, a real-time AI co-pilot. "
    "Answer in 2-4 sentences. Be direct. No preamble."
)


def _sync_observer(image_data: bytes) -> tuple:
    img = PIL.Image.open(io.BytesIO(image_data))
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[OBSERVER_PROMPT, img]
    )
    tokens = getattr(response.usage_metadata, "total_token_count", 0) if response.usage_metadata else 0
    return response.text.strip(), tokens


def _sync_guide(context: str, query: str) -> tuple:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"SCREEN CONTEXT:\n{context}\n\nUSER QUESTION:\n{query}\n\n{GUIDE_PROMPT}"
    )
    tokens = getattr(response.usage_metadata, "total_token_count", 0) if response.usage_metadata else 0
    return response.text.strip(), tokens


async def run_observer(image_base64: str) -> tuple:
    try:
        return await asyncio.to_thread(_sync_observer, base64.b64decode(image_base64))
    except Exception as e:
        print(f"[OBSERVER ERROR] {e}")
        return "APP: Unknown\nTASK: Unknown\nFOCUS: Unknown", 0


async def run_guide(context: str, query: str) -> tuple:
    try:
        return await asyncio.to_thread(_sync_guide, context, query)
    except Exception as e:
        print(f"[GUIDE ERROR] {e}")
        return "I could not process that, please try again.", 0


async def run_agent_pipeline(image_base64: str, user_query: str) -> dict:
    start = time.time()
    context, t1 = await run_observer(image_base64)
    response, t2 = await run_guide(context, user_query)
    return {
        "observer_context": context,
        "guide_response": response,
        "total_tokens": t1 + t2,
        "latency_ms": round((time.time() - start) * 1000, 2)
    }
