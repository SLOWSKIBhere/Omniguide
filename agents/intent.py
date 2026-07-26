"""Deterministic intent router with optional provider-backed mode."""
from __future__ import annotations

import os
import re

from models import IntentClassification, IntentType, ScreenContext
from providers import ProviderRouter, sanitize_provider_error

MODEL_PROMPT = """Classify the request as one of: debug_help, how_to, what_is,
navigation, code_review, general. Return JSON with intent_type, confidence,
entities, reasoning_hint. Return only JSON."""

_HINTS = {
    IntentType.DEBUG_HELP: "Diagnose the visible failure and give the smallest concrete fix.",
    IntentType.HOW_TO: "Give exact steps using the visible application and controls.",
    IntentType.WHAT_IS: "Explain the visible concept or message concisely.",
    IntentType.NAVIGATION: "Point to the exact visible menu, control, or shortcut.",
    IntentType.CODE_REVIEW: "Review only code or evidence visible on screen.",
    IntentType.GENERAL: "Answer directly while staying grounded in visible evidence.",
}

_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "what", "where", "when",
    "how", "does", "doing", "please", "could", "would", "should", "help",
}


class IntentRouter:
    def __init__(self, router: ProviderRouter):
        self.router = router
        self.mode = os.getenv("INTENT_ROUTER_MODE", "deterministic").strip().lower()

    @staticmethod
    def _entities(query: str) -> list[str]:
        words = re.findall(r"[A-Za-z0-9_.-]{3,}", query)
        output: list[str] = []
        for word in words:
            if word.lower() in _STOPWORDS or word.lower() in {w.lower() for w in output}:
                continue
            output.append(word)
            if len(output) == 8:
                break
        return output

    @staticmethod
    def _deterministic(query: str) -> IntentClassification:
        q = query.lower().strip()
        patterns = [
            (IntentType.CODE_REVIEW, r"\b(review|refactor|optimi[sz]e|code quality|improve this code)\b"),
            (IntentType.DEBUG_HELP, r"\b(error|bug|crash|failed|failing|doesn.?t work|not working|fix|traceback|exception|404|500)\b"),
            (IntentType.NAVIGATION, r"\b(where|find|navigate|which menu|which button|click|open settings)\b"),
            (IntentType.WHAT_IS, r"^(what is|what does|what are|why is|meaning of)\b"),
            (IntentType.HOW_TO, r"^(how do|how can|how to|steps to)\b"),
        ]
        selected = IntentType.GENERAL
        confidence = 0.62
        for intent, pattern in patterns:
            if re.search(pattern, q):
                selected = intent
                confidence = 0.9
                break
        return IntentClassification(
            intent_type=selected,
            confidence=confidence,
            entities=IntentRouter._entities(query),
            reasoning_hint=_HINTS[selected],
        )

    async def classify(self, query: str, context: ScreenContext) -> tuple:
        if self.mode != "model":
            result = self._deterministic(query)
            trace = {
                "stage": "intent", "status": "ok", "provider": "deterministic",
                "model": "semantic_rules_v1", "tokens": 0,
            }
            return result, 0, None, trace

        try:
            provider_result = await self.router.generate_json(
                system_prompt="You are OmniGuide's intent router.",
                user_prompt=(
                    f"{MODEL_PROMPT}\nQUERY: {query}\nAPP: {context.app}\n"
                    f"TASK: {context.task}\nFOCUS: {context.focus}"
                ),
            )
            data = provider_result.data or {}
            try:
                intent = IntentType(str(data.get("intent_type", "general")))
            except ValueError:
                intent = IntentType.GENERAL
            result = IntentClassification(
                intent_type=intent,
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
                entities=[str(item)[:80] for item in (data.get("entities") or [])[:8]],
                reasoning_hint=str(data.get("reasoning_hint") or _HINTS[intent])[:240],
            )
            trace = {
                "stage": "intent", "status": "ok", "provider": provider_result.provider,
                "model": provider_result.model, "tokens": provider_result.tokens,
                "attempts": provider_result.attempts,
            }
            return result, provider_result.tokens, None, trace
        except Exception as exc:
            error = sanitize_provider_error(exc)
            result = self._deterministic(query)
            trace = {
                "stage": "intent", "status": "fallback", "provider": "deterministic",
                "model": "semantic_rules_v1", "tokens": 0, "error": error,
            }
            return result, 0, error, trace
