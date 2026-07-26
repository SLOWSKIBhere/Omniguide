"""Grounded reasoning stage using the provider router."""
from __future__ import annotations

from models import IntentClassification, IntentType, ScreenContext
from providers import ProviderRouter, sanitize_provider_error

INTENT_PROMPTS = {
    IntentType.DEBUG_HELP: "Diagnose the visible error. Give the smallest safe fix and exact next action.",
    IntentType.HOW_TO: "Give concise steps tailored to the visible application.",
    IntentType.WHAT_IS: "Explain the visible concept or message in plain language.",
    IntentType.NAVIGATION: "Give exact navigation steps using visible UI labels or shortcuts.",
    IntentType.CODE_REVIEW: "Review only the visible code and name concrete improvements.",
    IntentType.GENERAL: "Answer directly and stay grounded in the supplied screen evidence.",
}


class ReasoningAgent:
    def __init__(self, router: ProviderRouter):
        self.router = router

    async def reason(self, query: str, context: ScreenContext, intent: IntentClassification) -> tuple:
        trace = {"stage": "reasoning", "status": "error", "tokens": 0}
        if not context.grounded or not context.evidence:
            error = "Grounding contract failed: no successful vision or OCR evidence"
            trace["error"] = error
            return "", 0, error, trace

        system_prompt = (
            "You are OmniGuide, a real-time screen co-pilot. "
            "Use only the supplied screen context and user question. "
            "Do not claim to see details absent from the context. "
            "Answer in 2-5 sentences, with commands or exact steps when useful. "
            + INTENT_PROMPTS.get(intent.intent_type, INTENT_PROMPTS[IntentType.GENERAL])
        )
        user_prompt = f"""SCREEN EVIDENCE:
App: {context.app}
Task: {context.task}
Focus: {context.focus}
Visible text: {context.visible_text[:800] or '(none extracted)'}
Evidence stages: {', '.join(context.evidence)}

INTENT: {intent.intent_type.value}
INTENT HINT: {intent.reasoning_hint}
USER QUESTION: {query}"""
        try:
            result = await self.router.generate_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            text = result.text.strip()
            if len(text) < 8:
                raise ValueError("Reasoning provider returned an empty response")
            trace.update({
                "status": "ok", "provider": result.provider, "model": result.model,
                "tokens": result.tokens, "attempts": result.attempts,
            })
            return text, result.tokens, None, trace
        except Exception as exc:
            error = sanitize_provider_error(exc)
            trace["error"] = error
            return "", 0, error, trace
