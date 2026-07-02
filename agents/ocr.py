"""
OCR Agent — Extracts text content from screenshots using Gemini.
Augments the Vision agent with raw text extraction for better context.
"""
import base64
import io
import asyncio
import logging

from google import genai
import PIL.Image

logger = logging.getLogger("omniguide.agents.ocr")

GEMINI_MODEL = "gemini-2.0-flash"

OCR_PROMPT = """Extract all visible text from this screenshot. Return a JSON object:
{"text": "<all visible text, max 500 chars, preserve order>"}
Focus on: error messages, code, headings, button labels, URLs, file names.
Return ONLY valid JSON."""


class OCRAgent:
    def __init__(self, client: genai.Client):
        self.client = client

    def _sync_extract(self, image_bytes: bytes) -> tuple:
        img = PIL.Image.open(io.BytesIO(image_bytes))

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[OCR_PROMPT, img],
            config={
                "response_mime_type": "application/json",
                "temperature": 0.0,
            }
        )

        import json
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]

        data = json.loads(raw)
        tokens = getattr(response.usage_metadata, "total_token_count", 0) if response.usage_metadata else 0
        return data.get("text", ""), tokens

    async def extract(self, image_base64: str) -> tuple:
        """
        Returns (extracted_text, token_count, error_or_None)
        Never raises — returns empty string on failure.
        """
        try:
            image_bytes = base64.b64decode(image_base64)
            if len(image_bytes) < 200:
                return "", 0, "Image too small"

            text, tokens = await asyncio.to_thread(self._sync_extract, image_bytes)
            logger.info("OCR OK: %d chars extracted, tokens=%d", len(text), tokens)
            return text, tokens, None

        except Exception as e:
            logger.error("OCR agent failed: %s: %s", type(e).__name__, str(e)[:200])
            return "", 0, f"{type(e).__name__}: {str(e)[:150]}"
