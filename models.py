"""OmniGuide v2.1 data contracts."""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    DEBUG_HELP = "debug_help"
    HOW_TO = "how_to"
    WHAT_IS = "what_is"
    NAVIGATION = "navigation"
    CODE_REVIEW = "code_review"
    GENERAL = "general"


class StageTrace(BaseModel):
    stage: str
    status: str
    provider: str = ""
    model: str = ""
    tokens: int = 0
    error: Optional[str] = None
    attempts: List[str] = Field(default_factory=list)


class ScreenContext(BaseModel):
    app: str = "unidentified"
    task: str = "unidentified"
    focus: str = "unidentified"
    visible_text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "context_builder"
    grounded: bool = False
    evidence: List[str] = Field(default_factory=list)


class IntentClassification(BaseModel):
    intent_type: IntentType = IntentType.GENERAL
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    entities: List[str] = Field(default_factory=list)
    reasoning_hint: str = ""


class AgentResponse(BaseModel):
    response: str
    context: str
    intent: str
    confidence: float
    latency_ms: float
    tokens: int
    status: str
    grounded: bool
    verified: bool
    run_id: str
    provider: Optional[str] = None
    model: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    agent_chain: List[str] = Field(default_factory=list)
    traces: List[StageTrace] = Field(default_factory=list)
    version: str = "2.1.0"


class AskRequest(BaseModel):
    image: str = Field(min_length=1, max_length=10_000_000)
    query: str = Field(min_length=1, max_length=4_000)
