"""
OmniGuide v2.0.0 — Data Models
Structured models for the multi-agent pipeline.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class IntentType(str, Enum):
    DEBUG_HELP = "debug_help"
    HOW_TO = "how_to"
    WHAT_IS = "what_is"
    NAVIGATION = "navigation"
    CODE_REVIEW = "code_review"
    GENERAL = "general"


class ScreenContext(BaseModel):
    """Structured screen analysis from Vision + OCR agents."""
    app: str = Field(default="Unknown", description="Application visible on screen")
    task: str = Field(default="Unknown", description="What the user appears to be doing")
    focus: str = Field(default="Unknown", description="Main UI element in focus")
    visible_text: str = Field(default="", description="Key text extracted from screen")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score 0-1")
    source: str = Field(default="gemini", description="Which agent produced this context")


class IntentClassification(BaseModel):
    """User intent classification from the Intent Router."""
    intent_type: IntentType = Field(default=IntentType.GENERAL)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    entities: List[str] = Field(default_factory=list, description="Key entities extracted from query")
    reasoning_hint: str = Field(default="", description="Hint for the reasoning agent")


class AgentResponse(BaseModel):
    """Final response from the pipeline."""
    response: str
    context: str
    intent: str
    confidence: float
    latency_ms: float
    tokens: int
    errors: List[str] = Field(default_factory=list)
    agent_chain: List[str] = Field(default_factory=list, description="Which agents ran successfully")


class AskRequest(BaseModel):
    """REST API request model."""
    image: str
    query: str
