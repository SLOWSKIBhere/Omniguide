"""Parallel context builder with explicit grounding evidence."""
from __future__ import annotations

import asyncio
import logging

from models import ScreenContext

logger = logging.getLogger("omniguide.agents.context")


class ContextBuilder:
    def __init__(self, vision, ocr):
        self.vision = vision
        self.ocr = ocr

    async def build(self, image_base64: str) -> tuple:
        vision_result, ocr_result = await asyncio.gather(
            self.vision.analyze(image_base64),
            self.ocr.extract(image_base64),
            return_exceptions=True,
        )
        errors: list[str] = []
        chain: list[str] = []
        traces: list[dict] = []
        tokens = 0
        ctx = ScreenContext()

        if isinstance(vision_result, Exception):
            errors.append(f"vision: {type(vision_result).__name__}")
            traces.append({"stage": "vision", "status": "error", "error": type(vision_result).__name__})
        else:
            v_ctx, v_tokens, v_error, v_trace = vision_result
            traces.append(v_trace)
            tokens += v_tokens
            if v_error:
                errors.append(f"vision: {v_error}")
            else:
                ctx = v_ctx
                chain.append("vision")

        if isinstance(ocr_result, Exception):
            errors.append(f"ocr: {type(ocr_result).__name__}")
            traces.append({"stage": "ocr", "status": "error", "error": type(ocr_result).__name__})
        else:
            text, o_tokens, o_error, o_trace = ocr_result
            traces.append(o_trace)
            tokens += o_tokens
            if o_error:
                errors.append(f"ocr: {o_error}")
            else:
                chain.append("ocr")
                if text and len(text) > len(ctx.visible_text):
                    ctx.visible_text = text
                if "ocr" not in ctx.evidence:
                    ctx.evidence.append("ocr")

        ctx.grounded = bool(chain)
        if ctx.grounded:
            ctx.source = "+".join(chain)
            if len(chain) == 2:
                ctx.confidence = min(1.0, ctx.confidence + 0.1)
        else:
            ctx.source = "context_unavailable"
            ctx.confidence = 0.0
            ctx.evidence = []

        logger.info("Context grounded=%s chain=%s errors=%d", ctx.grounded, chain, len(errors))
        return ctx, tokens, errors, chain, traces
