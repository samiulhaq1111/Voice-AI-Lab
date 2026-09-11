"""Abstract base interface for Speech-to-Text providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.providers.types import STTResult


class STTInterface(ABC):
    """Abstract interface that all STT providers must implement.

    The Agent Runtime depends on this interface, NOT on concrete providers.
    Adding a new STT provider requires implementing this interface.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique name of this STT provider (e.g. 'deepgram')."""

    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        *,
        model: str | None = None,
        language: str = "en",
    ) -> STTResult:
        """Transcribe a complete audio buffer to text.

        Args:
            audio_data: Raw audio bytes.
            model: Model identifier override (provider-specific).
            language: Expected language code.

        Returns:
            STTResult with the transcribed text.
        """

    @abstractmethod
    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[STTResult]:
        """Stream audio chunks and yield incremental transcription results.

        Args:
            audio_stream: Async iterator of audio byte chunks.

        Yields:
            STTResult instances (is_final=False for interim, True for final).
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by this provider."""
