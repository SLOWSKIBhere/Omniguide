"""OCR stage backed by the provider router."""
from __future__ import annotations

import base64
import logging

from providers import ProviderRouter, sanitize_provider_error

logger = logging.getLogger("omniguide.agents.ocr")

OCR_PROMPT = """Extract visible text from this screenshot. Return ONLY JSON:
{"text": "<visible text in reading order, maximum 800 characters>"}
Prioritize errors, code, headings, labels, URLs, and filenames. Do not invent text."""


class OCRAgent:
    def __init__(self, router: ProviderRouter):
        self.router = router

    async def extract(self, image_base64: str) -> tuple:
        trace = {"stage": "ocr", "status": "error", "tokens": 0}
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
            if len(image_bytes) < 200:
                raise ValueError("Image payload too small")
            result = await self.router.generate_json(
                system_prompt="You are OmniGuide's literal OCR evidence extractor.",
                user_prompt=OCR_PROMPT,
                image_base64=image_base64,
            )
            text = str((result.data or {}).get("text") or "").strip()[:800]
            if not text:
                raise ValueError("OCR provider returned no visible text")
            trace.update({
                "status": "ok", "provider": result.provider, "model": result.model,
                "tokens": result.tokens, "attempts": result.attempts,
            })
            return text, result.tokens, None, trace
        except Exception as exc:
            error = sanitize_provider_error(exc)
            logger.error("OCR failed: %s", error)
            trace["error"] = error
            return "", 0, error, trace
