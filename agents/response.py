"""
Response Agent — Formats the final pipeline output.
Assembles context display string, computes metadata, ensures consistent structure.
"""
import logging
from typing import List

from models import ScreenContext, IntentClassification, AgentResponse

logger = logging.getLogger("omniguide.agents.response")


class ResponseAgent:
    @staticmethod
    def format_context(ctx: ScreenContext) -> str:
        """Human-readable context string for the frontend."""
        parts = []
        if ctx.app and ctx.app != "unidentified":
            parts.append(f"APP: {ctx.app}")
        if ctx.task and ctx.task != "unidentified":
            parts.append(f"TASK: {ctx.task}")
        if ctx.focus and ctx.focus != "unidentified":
            parts.append(f"FOCUS: {ctx.focus}")
        if not parts:
            return "CONTEXT: limited — answered from question alone"
        return " / ".join(parts)

    @staticmethod
    def build(
        response_text: str,
        context: ScreenContext,
        intent: IntentClassification,
        latency_ms: float,
        tokens: int,
        errors: List[str],
        agent_chain: List[str]
    ) -> dict:
        """
        Assembles the final JSON response for the API.
        Always returns a valid dict — never raises.
        """
        context_str = ResponseAgent.format_context(context)

        # Determine if the response contains an error signal
        has_errors = bool(errors)
        if has_errors and "exhausted retries" in " ".join(errors):
            logger.warning("Pipeline completed with agent failures: %s", errors)

        return {
            "response": response_text,
            "context": context_str,
            "intent": intent.intent_type.value,
            "confidence": round(max(context.confidence, intent.confidence), 2),
            "latency_ms": round(latency_ms, 2),
            "tokens": tokens,
            "errors": errors,
            "agent_chain": agent_chain,
            "version": "2.0.0"
        }
