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
        output_format: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream synthesized audio in chunks.

        Chunks are raw encoded-audio bytes in the requested output format.
        Providers that expose a telephony-native format (e.g. ElevenLabs
        'ulaw_8000') emit chunks that are directly playable without local
        transcoding.

        Args:
            text: Text to synthesize.
            voice: Voice identifier override.
            model: Model identifier override.
            speed: Speech speed multiplier.
            output_format: Provider-specific audio format identifier
                (e.g. 'ulaw_8000', 'mp3_44100_128'). None uses the
                provider default.

        Yields:
            Audio byte chunks.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by this provider."""
