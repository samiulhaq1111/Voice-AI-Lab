"""Shared types for provider interfaces."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ProviderType(StrEnum):
    """Types of AI providers."""

    STT = "stt"
    LLM = "llm"
    TTS = "tts"


@dataclass
class STTResult:
    """Result from a speech-to-text provider."""

    text: str
    confidence: float = 1.0
    language: str = "en"
    is_final: bool = True
    duration_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMMessage:
    """A message in an LLM conversation."""

    role: str  # system, user, assistant, tool
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    # prompt_tokens, completion_tokens, total_tokens
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TTSResult:
    """Result from a text-to-speech provider."""

    audio_data: bytes = b""
    content_type: str = "audio/wav"
    duration_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSchema:
    """Schema for a tool exposed to the LLM."""

    name: str
    description: str
    parameters: dict[str, Any]
