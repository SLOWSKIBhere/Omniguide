from google import genai
from google.genai import types
import base64, time, asyncio, io
import PIL.Image

client = genai.Client(api_key="AIzaSyAxnpECKh2GHdeWtJa6JTicIzrBSG3xzUA")

OBSERVER_PROMPT = "You are a Screen Context Observer. Analyze the screenshot and return ONLY:\nAPP: <software visible>\nTASK: <what user is doing>\nFOCUS: <main UI element>\nMax 50 words."

GUIDE_PROMPT = "You are OmniGuide, a real-time AI co-pilot. Answer in 2-4 sentences. Be direct. No preamble."

def _sync_observer(image_data):
    img = PIL.Image.open(io.BytesIO(image_data))
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[OBSERVER_PROMPT, img]
    )
    return response.text.strip(), 0

def _sync_guide(context, query):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"SCREEN CONTEXT:\n{context}\n\nUSER QUESTION:\n{query}\n\n{GUIDE_PROMPT}"
    )
    return response.text.strip(), 0

async def run_observer(image_base64):
    try:
        return await asyncio.to_thread(_sync_observer, base64.b64decode(image_base64))
    except Exception as e:
        print(f"[OBSERVER ERROR] {e}")
        return "APP: Unknown\nTASK: Unknown\nFOCUS: Unknown", 0

async def run_guide(context, query):
    try:
        return await asyncio.to_thread(_sync_guide, context, query)
    except Exception as e:
        print(f"[GUIDE ERROR] {e}")
        return "I could not process that, please try again.", 0

async def run_agent_pipeline(image_base64, user_query):
    start = time.time()
    context, t1 = await run_observer(image_base64)
    response, t2 = await run_guide(context, user_query)
    return {
        "observer_context": context,
        "guide_response": response,
        "total_tokens": t1+t2,
        "latency_ms": (time.time()-start)*1000
    }
