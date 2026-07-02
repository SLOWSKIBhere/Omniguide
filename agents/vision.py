"""
Vision Agent — Analyzes screenshots using Gemini 2.0 Flash vision.
Returns structured JSON, never returns all-Unknown.
"""
import base64
import io
import asyncio
import logging

from google import genai
import PIL.Image

from models import ScreenContext

logger = logging.getLogger("omniguide.agents.vision")

GEMINI_MODEL = "gemini-2.0-flash"

VISION_PROMPT = """Analyze this screenshot and return a JSON object with these fields:
- "app": Name of the application or website visible (e.g. "VS Code", "Chrome", "Terminal")
- "task": What the user appears to be doing (e.g. "debugging Python", "browsing documentation")
- "focus": The main UI element or area in focus (e.g. "error popup", "code editor line 24")
- "visible_text": Key text visible on screen, max 200 chars (error messages, headings, labels)
- "confidence": Your confidence score from 0.0 to 1.0

If you cannot identify something, use "unidentified" for that field — NOT "Unknown".
Return ONLY valid JSON, no markdown."""


class VisionAgent:
    def __init__(self, client: genai.Client):
        self.client = client

    def _sync_analyze(self, image_bytes: bytes) -> ScreenContext:
        img = PIL.Image.open(io.BytesIO(image_bytes))

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[VISION_PROMPT, img],
            config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
            }
        )

        import json
        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        data = json.loads(raw)
        tokens = getattr(response.usage_metadata, "total_token_count", 0) if response.usage_metadata else 0

        return ScreenContext(
            app=data.get("app", "unidentified"),
            task=data.get("task", "unidentified"),
            focus=data.get("focus", "unidentified"),
            visible_text=data.get("visible_text", ""),
            confidence=float(data.get("confidence", 0.5)),
            source="vision"
        ), tokens

    async def analyze(self, image_base64: str) -> tuple:
        """
        Returns (ScreenContext, token_count, error_or_None)
        Never raises — always returns a valid ScreenContext.
        """
        try:
            image_bytes = base64.b64decode(image_base64)
            if len(image_bytes) < 200:
                logger.warning("Image too small: %d bytes", len(image_bytes))
                return ScreenContext(
                    app="unidentified", task="unidentified",
                    focus="unidentified", confidence=0.0,
                    source="vision_fallback"
                ), 0, "Image payload too small"

            result, tokens = await asyncio.to_thread(self._sync_analyze, image_bytes)
            logger.info("Vision OK: app=%s confidence=%.2f tokens=%d", result.app, result.confidence, tokens)
            return result, tokens, None

        except Exception as e:
            logger.error("Vision agent failed: %s: %s", type(e).__name__, str(e)[:200])
            # Graceful fallback — partial context, not all "Unknown"
            return ScreenContext(
                app="unidentified",
                task="unidentified",
                focus="unidentified",
                confidence=0.0,
                source="vision_error"
            ), 0, f"{type(e).__name__}: {str(e)[:150]}"
