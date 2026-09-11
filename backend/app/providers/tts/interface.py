"""Abstract base interface for Text-to-Speech providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.providers.types import TTSResult


class TTSInterface(ABC):
    """Abstract interface that all TTS providers must implement.

    The Agent Runtime depends on this interface, NOT on concrete providers.
    Adding a new TTS provider requires implementing this interface.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique name of this TTS provider (e.g. 'elevenlabs')."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Convert text to speech audio.

        Args:
            text: Text to synthesize.
            voice: Voice identifier override.
            model: Model identifier override.
            speed: Speech speed multiplier.

        Returns:
            TTSResult containing audio data.
        """

    @abstractmethod
    async def stream_synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        speed: float = 1.0,
    ) -> AsyncIterator[bytes]:
        """Stream synthesized audio in chunks.

        Args:
            text: Text to synthesize.
            voice: Voice identifier override.
            model: Model identifier override.
            speed: Speech speed multiplier.

        Yields:
            Audio byte chunks.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by this provider."""
