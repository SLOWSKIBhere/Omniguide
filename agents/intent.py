"""
Intent Router — Classifies user intent using Gemini structured JSON.
Determines what kind of help the user needs and routes accordingly.
Never blocks the pipeline — defaults to GENERAL on any failure.
"""
import asyncio
import json
import logging

from google import genai
from models import ScreenContext, IntentClassification, IntentType

logger = logging.getLogger("omniguide.agents.intent")

GEMINI_MODEL = "gemini-2.0-flash"

INTENT_PROMPT = """You are an intent classifier. Given a user's question and their screen context, classify the intent.

Return JSON:
{
  "intent_type": one of ["debug_help", "how_to", "what_is", "navigation", "code_review", "general"],
  "confidence": 0.0 to 1.0,
  "entities": ["key terms from the question"],
  "reasoning_hint": "one sentence guiding the reasoning agent on approach"
}

Definitions:
- debug_help: User has an error/bug and needs help fixing it
- how_to: User wants to know how to do something
- what_is: User is asking what something is or means
- navigation: User wants to find or navigate to something
- code_review: User wants code reviewed or improved
- general: Anything else

Return ONLY valid JSON."""

MAX_RETRIES = 2


class IntentRouter:
    def __init__(self, client: genai.Client):
        self.client = client

    def _sync_classify(self, query: str, context: ScreenContext) -> tuple:
        prompt = f"""{INTENT_PROMPT}

USER QUESTION: "{query}"
SCREEN CONTEXT: app={context.app}, task={context.task}, focus={context.focus}
VISIBLE TEXT: {context.visible_text[:300]}"""

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.0,
            }
        )

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]

        data = json.loads(raw)
        tokens = getattr(response.usage_metadata, "total_token_count", 0) if response.usage_metadata else 0

        # Parse intent type with fallback
        try:
            intent = IntentType(data.get("intent_type", "general"))
        except ValueError:
            intent = IntentType.GENERAL

        return IntentClassification(
            intent_type=intent,
            confidence=float(data.get("confidence", 0.5)),
            entities=data.get("entities", []),
            reasoning_hint=data.get("reasoning_hint", "")
        ), tokens

    async def classify(self, query: str, context: ScreenContext) -> tuple:
        """
        Returns (IntentClassification, token_count, error_or_None)
        Retries up to MAX_RETRIES on failure, then falls back to GENERAL.
        """
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                result, tokens = await asyncio.to_thread(self._sync_classify, query, context)
                logger.info(
                    "Intent: type=%s confidence=%.2f entities=%s attempt=%d",
                    result.intent_type, result.confidence, result.entities, attempt
                )
                return result, tokens, None

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)[:150]}"
                logger.warning("Intent classify attempt %d failed: %s", attempt + 1, last_error)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Brief backoff

        # Graceful fallback — never block the pipeline
        logger.error("Intent router exhausted retries, falling back to GENERAL")
        return IntentClassification(
            intent_type=IntentType.GENERAL,
            confidence=0.0,
            entities=[],
            reasoning_hint="Classification failed — provide best-effort general help"
        ), 0, last_error
