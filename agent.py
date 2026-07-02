import os
import base64
import time
import asyncio
import io
import logging

from google import genai
import PIL.Image

logger = logging.getLogger("omniguide.agent")

# Initialize client from environment variable
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
    "Answer in 2-4 sentences. Be direct. No preamble. "
    "If the context is unknown, still try to help based on the question."
)


def _sync_observer(image_data: bytes) -> tuple:
    """Analyze screenshot with Gemini Vision."""
    img = PIL.Image.open(io.BytesIO(image_data))

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[OBSERVER_PROMPT, img]
    )
    tokens = getattr(response.usage_metadata, "total_token_count", 0) if response.usage_metadata else 0
    return response.text.strip(), tokens


def _sync_guide(context: str, query: str) -> tuple:
    """Generate guidance based on screen context + user question."""
    prompt = f"SCREEN CONTEXT:\n{context}\n\nUSER QUESTION:\n{query}\n\n{GUIDE_PROMPT}"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    tokens = getattr(response.usage_metadata, "total_token_count", 0) if response.usage_metadata else 0
    return response.text.strip(), tokens


async def run_observer(image_base64: str) -> tuple:
    try:
        image_bytes = base64.b64decode(image_base64)
        if len(image_bytes) < 100:
            return "APP: Unknown\nTASK: Unknown\nFOCUS: Image too small or empty", 0
        return await asyncio.to_thread(_sync_observer, image_bytes)
    except Exception as e:
        error_msg = f"OBSERVER_ERROR: {type(e).__name__}: {str(e)[:200]}"
        logger.error(error_msg)
        return f"APP: Unknown\nTASK: Unknown\nFOCUS: {error_msg}", 0


async def run_guide(context: str, query: str) -> tuple:
    try:
        return await asyncio.to_thread(_sync_guide, context, query)
    except Exception as e:
        error_msg = f"GUIDE_ERROR: {type(e).__name__}: {str(e)[:200]}"
        logger.error(error_msg)
        return f"I could not process that. Error: {str(e)[:150]}", 0


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
