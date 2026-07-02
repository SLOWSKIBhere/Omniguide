"""
Reasoning Agent — Generates the actual AI response.
Uses intent classification to pick the right prompt strategy.
Includes retry logic and intent-specific system prompts.
"""
import asyncio
import logging

from google import genai
from models import ScreenContext, IntentClassification, IntentType

logger = logging.getLogger("omniguide.agents.reasoning")

GEMINI_MODEL = "gemini-2.0-flash"
MAX_RETRIES = 2

# Intent-specific prompt strategies
INTENT_PROMPTS = {
    IntentType.DEBUG_HELP: (
        "You are a debugging expert. The user has an error or bug. "
        "Identify the likely cause from the screen context and visible text. "
        "Give a specific fix in 2-4 sentences. If you see an error message, address it directly. "
        "Include a code snippet if helpful."
    ),
    IntentType.HOW_TO: (
        "You are a helpful guide. The user wants to know how to do something. "
        "Give clear, step-by-step instructions in 2-4 sentences. "
        "Reference the app they're using for context-specific shortcuts."
    ),
    IntentType.WHAT_IS: (
        "You are a knowledgeable assistant. The user is asking what something means. "
        "Explain concisely in 2-3 sentences. Relate it to what's on their screen if relevant."
    ),
    IntentType.NAVIGATION: (
        "You are a navigation helper. The user wants to find or go to something. "
        "Give specific steps to navigate there in 2-3 sentences. "
        "Reference menu names, keyboard shortcuts, or UI elements visible on screen."
    ),
    IntentType.CODE_REVIEW: (
        "You are a code reviewer. The user wants code reviewed or improved. "
        "Identify issues and suggest improvements in 3-4 sentences. "
        "Reference specific lines or patterns visible in the screen context."
    ),
    IntentType.GENERAL: (
        "You are OmniGuide, a real-time AI co-pilot. "
        "Answer in 2-4 sentences. Be direct and helpful. No preamble. "
        "Use the screen context to make your answer specific to what the user is doing."
    ),
}


class ReasoningAgent:
    def __init__(self, client: genai.Client):
        self.client = client

    def _sync_reason(self, query: str, context: ScreenContext, intent: IntentClassification) -> tuple:
        system_prompt = INTENT_PROMPTS.get(intent.intent_type, INTENT_PROMPTS[IntentType.GENERAL])

        # Build context block — handle unidentified fields gracefully
        context_parts = []
        if context.app and context.app != "unidentified":
            context_parts.append(f"App: {context.app}")
        if context.task and context.task != "unidentified":
            context_parts.append(f"Task: {context.task}")
        if context.focus and context.focus != "unidentified":
            context_parts.append(f"Focus: {context.focus}")
        if context.visible_text:
            context_parts.append(f"Visible text: {context.visible_text[:400]}")

        context_block = "\n".join(context_parts) if context_parts else "Screen context unavailable — answer based on the question alone."

        prompt = f"""{system_prompt}

SCREEN CONTEXT:
{context_block}

INTENT HINT: {intent.reasoning_hint}

USER QUESTION: {query}

Respond directly and helpfully:"""

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": 0.3}
        )

        tokens = getattr(response.usage_metadata, "total_token_count", 0) if response.usage_metadata else 0
        return response.text.strip(), tokens

    async def reason(self, query: str, context: ScreenContext, intent: IntentClassification) -> tuple:
        """
        Returns (response_text, token_count, error_or_None)
        Retries on failure, then returns a graceful fallback message.
        """
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                result, tokens = await asyncio.to_thread(self._sync_reason, query, context, intent)
                if result and len(result) > 5:
                    logger.info("Reasoning OK: intent=%s tokens=%d attempt=%d", intent.intent_type, tokens, attempt)
                    return result, tokens, None
                else:
                    raise ValueError("Response too short or empty")

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)[:150]}"
                logger.warning("Reasoning attempt %d failed: %s", attempt + 1, last_error)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(0.5 * (attempt + 1))

        # Graceful fallback — still try to be useful
        logger.error("Reasoning agent exhausted retries")
        fallback = f"I can see you're working in {context.app} but I'm having trouble generating a detailed response right now. "

        if context.visible_text:
            fallback += f"I can see text including: \"{context.visible_text[:100]}\". "

        if intent.intent_type == IntentType.DEBUG_HELP and context.visible_text:
            # Try to extract error-like text as a last resort
            fallback += "If you're seeing an error, try searching for the exact error message or pasting it here again."
        else:
            fallback += "Try rephrasing your question or asking again — the AI service may have had a temporary issue."

        return fallback, 0, last_error
