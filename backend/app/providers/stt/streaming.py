"""Streaming STT abstraction (Phase 6A).

Defines a provider-neutral streaming speech-to-text session interface.
The realtime voice gateway depends on this abstraction, never on
Deepgram-specific protocol details.

Unlike ``STTInterface.stream_transcribe`` (pull-based, buffer-friendly),
the streaming session supports a *bidirectional* realtime pattern:

    session.start()        # open provider connection
    session.send_audio()   # push PCM chunks (client -> provider)
    session.receive()      # pull transcript events (provider -> client)
    session.finish()       # signal end of audio
    session.close()        # release resources
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.core.logging import logger

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

SUPPORTED_ENCODINGS = {"linear16"}


@dataclass
class StreamConfig:
    """Configuration for a realtime streaming STT session."""

    model: str | None = None
    language: str = "en"
    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "linear16"
    interim_results: bool = True
    endpointing_ms: int = 300
    utterance_end_ms: int = 1000

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = settings.default_stt_model or "nova-3"

    def validate(self) -> str | None:
        """Return an error message if the configuration is invalid, else None."""
        if self.encoding not in SUPPORTED_ENCODINGS:
            return (
                f"Unsupported encoding: '{self.encoding}'. "
                f"Supported: {sorted(SUPPORTED_ENCODINGS)}"
            )
        if not (8000 <= self.sample_rate <= 48000):
            return f"sample_rate out of range (8000-48000): {self.sample_rate}"
        if self.channels != 1:
            return f"Only mono audio is supported, got channels={self.channels}"
        if not (100 <= self.endpointing_ms <= 3000):
            return f"endpointing_ms out of range (100-3000): {self.endpointing_ms}"
        return None


@dataclass
class StreamEvent:
    """A single event produced by a streaming STT session."""

    # One of: "partial" | "final" | "utterance_end" | "error"
    type: str
    text: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class StreamingSTTSession(ABC):
    """Abstract realtime streaming STT session.

    One session corresponds to one provider streaming connection.
    Implementations must be safe to ``close()`` from a ``finally`` block
    even if ``start()`` failed.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique name of the STT provider (e.g. 'deepgram')."""

    @abstractmethod
    async def start(self) -> None:
        """Open the provider streaming connection."""

    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None:
        """Forward one binary audio chunk to the provider."""

    @abstractmethod
    async def receive(self) -> StreamEvent | None:
        """Return the next meaningful event.

        Contract: None MUST mean only "provider stream ended". Ignorable
        provider messages (metadata, silence ticks, stray frames) must be
        consumed internally — returning None for them would prematurely
        terminate the gateway's event pump.
        """

    @abstractmethod
    async def finish(self) -> None:
        """Signal end-of-audio so the provider can flush final results."""

    @abstractmethod
    async def close(self) -> None:
        """Release all resources. Must be idempotent."""


class StreamingSTTError(Exception):
    """Raised when a streaming STT session cannot be established or fails."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def open_streaming_session(config: StreamConfig) -> StreamingSTTSession:
    """Create a streaming STT session for the configured default provider.

    The provider is resolved from settings (default_stt_provider), keeping
    the gateway free of provider-specific imports.
    """
    provider = settings.default_stt_provider

    if provider == "deepgram":
        # Imported lazily to keep this module provider-agnostic
        from app.providers.stt.deepgram_streaming import DeepgramStreamingSession

        if not settings.is_provider_configured("deepgram"):
            raise StreamingSTTError(
                "Deepgram API key not configured. Set DEEPGRAM_API_KEY."
            )
        logger.info("[STT:STREAM] Creating streaming session provider=deepgram")
        return DeepgramStreamingSession(config)

    raise StreamingSTTError(
        f"Streaming STT not supported for provider: '{provider}'"
    )
