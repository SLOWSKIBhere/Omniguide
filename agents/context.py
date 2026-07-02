"""
Context Builder — Merges Vision + OCR outputs into a unified ScreenContext.
Runs both agents in parallel, picks the best result, fills gaps.
"""
import asyncio
import logging

from agents.vision import VisionAgent
from agents.ocr import OCRAgent
from models import ScreenContext

logger = logging.getLogger("omniguide.agents.context")


class ContextBuilder:
    def __init__(self, vision: VisionAgent, ocr: OCRAgent):
        self.vision = vision
        self.ocr = ocr

    async def build(self, image_base64: str) -> tuple:
        """
        Runs Vision + OCR in parallel, merges results.
        Returns (ScreenContext, total_tokens, errors_list, agent_chain_list)
        """
        errors = []
        agent_chain = []

        # Run both agents concurrently
        vision_task = asyncio.create_task(self.vision.analyze(image_base64))
        ocr_task = asyncio.create_task(self.ocr.extract(image_base64))

        vision_result, ocr_result = await asyncio.gather(
            vision_task, ocr_task, return_exceptions=True
        )

        # Process vision result
        total_tokens = 0
        ctx = ScreenContext(source="context_builder")

        if isinstance(vision_result, Exception):
            errors.append(f"vision: {type(vision_result).__name__}")
            logger.error("Vision agent exception: %s", vision_result)
        else:
            v_ctx, v_tokens, v_err = vision_result
            total_tokens += v_tokens
            ctx = v_ctx  # Start with vision result
            if v_err:
                errors.append(f"vision: {v_err}")
            else:
                agent_chain.append("vision")

        # Merge OCR result into context
        if isinstance(ocr_result, Exception):
            errors.append(f"ocr: {type(ocr_result).__name__}")
        else:
            ocr_text, ocr_tokens, ocr_err = ocr_result
            total_tokens += ocr_tokens
            if ocr_err:
                errors.append(f"ocr: {ocr_err}")
            else:
                agent_chain.append("ocr")
                # Enrich context with OCR text
                if ocr_text and not ctx.visible_text:
                    ctx.visible_text = ocr_text
                elif ocr_text and ctx.visible_text and len(ocr_text) > len(ctx.visible_text):
                    ctx.visible_text = ocr_text  # Use longer extraction

        # Confidence adjustment — if both agents succeeded, boost confidence
        if len(agent_chain) == 2:
            ctx.confidence = min(1.0, ctx.confidence + 0.1)

        logger.info(
            "Context built: app=%s task=%s confidence=%.2f errors=%d chain=%s",
            ctx.app, ctx.task, ctx.confidence, len(errors), agent_chain
        )

        return ctx, total_tokens, errors, agent_chain
