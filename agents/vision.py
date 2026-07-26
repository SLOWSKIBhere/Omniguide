"""Vision stage backed by the provider router."""
from __future__ import annotations

import base64
import logging

from models import ScreenContext
from providers import ProviderRouter, sanitize_provider_error

logger = logging.getLogger("omniguide.agents.vision")

VISION_PROMPT = """Analyze the screenshot and return ONLY a JSON object with:
app, task, focus, visible_text, confidence.
Use concise factual descriptions. visible_text must be at most 500 characters.
Use \"unidentified\" when uncertain. confidence must be 0.0 to 1.0.
Do not infer screen content that is not visibly supported."""


class VisionAgent:
    def __init__(self, router: ProviderRouter):
        self.router = router

    async def analyze(self, image_base64: str) -> tuple:
        trace = {"stage": "vision", "status": "error", "tokens": 0}
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
            if len(image_bytes) < 200:
                raise ValueError("Image payload too small")
            result = await self.router.generate_json(
                system_prompt="You are OmniGuide's screen-evidence observer.",
                user_prompt=VISION_PROMPT,
                image_base64=image_base64,
            )
            data = result.data or {}
            app = str(data.get("app") or "unidentified")[:120]
            task = str(data.get("task") or "unidentified")[:200]
            focus = str(data.get("focus") or "unidentified")[:200]
            visible_text = str(data.get("visible_text") or "")[:500]
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            meaningful = any(
                value and value.lower() not in {"unknown", "unidentified", "n/a"}
                for value in (app, task, focus, visible_text)
            )
            if not meaningful:
                raise ValueError("Vision provider returned no usable screen evidence")
            ctx = ScreenContext(
                app=app,
                task=task,
                focus=focus,
                visible_text=visible_text,
                confidence=confidence,
                source=f"vision:{result.provider}",
                grounded=True,
                evidence=["vision"],
            )
            trace.update({
                "status": "ok", "provider": result.provider, "model": result.model,
                "tokens": result.tokens, "attempts": result.attempts,
            })
            return ctx, result.tokens, None, trace
        except Exception as exc:
            error = sanitize_provider_error(exc)
            logger.error("Vision failed: %s", error)
            trace["error"] = error
            return ScreenContext(source="vision_error"), 0, error, trace
