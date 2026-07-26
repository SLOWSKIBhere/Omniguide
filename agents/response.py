"""Verified response assembly."""
from __future__ import annotations

from typing import List

from models import IntentClassification, ScreenContext


class ResponseAgent:
    @staticmethod
    def format_context(ctx: ScreenContext) -> str:
        parts = []
        if ctx.app not in {"", "unidentified", "Unknown"}:
            parts.append(f"APP: {ctx.app}")
        if ctx.task not in {"", "unidentified", "Unknown"}:
            parts.append(f"TASK: {ctx.task}")
        if ctx.focus not in {"", "unidentified", "Unknown"}:
            parts.append(f"FOCUS: {ctx.focus}")
        if not parts:
            return "CONTEXT: unavailable — no screen-grounded answer generated"
        return " / ".join(parts)

    @staticmethod
    def build(
        *,
        run_id: str,
        response_text: str,
        context: ScreenContext,
        intent: IntentClassification,
        latency_ms: float,
        tokens: int,
        errors: List[str],
        agent_chain: List[str],
        traces: List[dict],
    ) -> dict:
        reasoning_trace = next((t for t in traces if t.get("stage") == "reasoning"), None)
        reasoning_ok = bool(reasoning_trace and reasoning_trace.get("status") == "ok")
        verified = bool(context.grounded and response_text.strip() and reasoning_ok)

        if verified:
            status = "degraded" if errors else "ok"
            response = response_text.strip()
        elif not context.grounded:
            status = "unavailable"
            response = (
                "No answer was generated because OmniGuide did not receive verified screen evidence. "
                "Check the configured multimodal provider, then capture the screen and retry."
            )
        else:
            status = "unavailable"
            response = (
                "The screen was captured, but no reasoning provider completed successfully. "
                "Configure an available OpenAI-compatible or Gemini provider and retry."
            )

        provider = reasoning_trace.get("provider") if reasoning_trace else None
        model = reasoning_trace.get("model") if reasoning_trace else None
        return {
            "response": response,
            "context": ResponseAgent.format_context(context),
            "intent": intent.intent_type.value,
            "confidence": round(max(context.confidence, intent.confidence), 2),
            "latency_ms": round(latency_ms, 2),
            "tokens": tokens,
            "status": status,
            "grounded": context.grounded,
            "verified": verified,
            "run_id": run_id,
            "provider": provider or None,
            "model": model or None,
            "errors": errors,
            "agent_chain": agent_chain,
            "traces": traces,
            "error": None if verified else response,
            "version": "2.1.0",
        }
