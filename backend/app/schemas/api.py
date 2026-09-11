"""Pydantic schemas for API request/response validation."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Health ---
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Session ---
class SessionCreateRequest(BaseModel):
    """Request to create a new voice session."""

    system_prompt: str = "You are a helpful voice assistant."
    stt_provider: str | None = None
    stt_model: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    tts_provider: str | None = None
    tts_model: str | None = None
    tts_voice: str | None = None


class SessionResponse(BaseModel):
    """Session information response."""

    id: str
    status: str
    system_prompt: str | None = None
    stt_provider: str | None = None
    stt_model: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    tts_provider: str | None = None
    tts_model: str | None = None
    tts_voice: str | None = None
    duration_seconds: float | None = None
    total_cost: float | None = None
    message_count: int = 0
    created_at: datetime
    ended_at: datetime | None = None


# --- Chat ---
class ChatRequest(BaseModel):
    """Request to send a text message to the agent."""

    message: str = Field(..., min_length=1, max_length=10000)
    session_id: str | None = None
    provider: str | None = None
    model: str | None = None


class ChatResponse(BaseModel):
    """Agent response to a chat message."""

    response: str
    session_id: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    iterations: int = 0
    latency_ms: float | None = None


# --- Tools ---
class ToolInfoResponse(BaseModel):
    """Information about a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    enabled: bool


class ToolListResponse(BaseModel):
    """List of all registered tools."""

    tools: list[ToolInfoResponse]
    count: int


# --- Provider ---
class ProviderInfoResponse(BaseModel):
    """Information about a configured provider."""

    provider_type: str
    provider_name: str
    display_name: str
    is_active: bool
