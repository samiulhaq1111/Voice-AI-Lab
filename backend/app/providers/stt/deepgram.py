"""Deepgram STT adapter implementing STTInterface."""

from collections.abc import AsyncIterator

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.providers.stt.interface import STTInterface
from app.providers.types import STTResult

_DEEPGRAM_TRANSCRIBE_URL = "https://api.deepgram.com/v1/listen"


class DeepgramAdapter(STTInterface):
    """Deepgram adapter for speech-to-text.

    Uses the Deepgram REST API for batch transcription.
    Requires DEEPGRAM_API_KEY in environment configuration.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        resolved_key = api_key or settings.deepgram_api_key
        if not resolved_key:
            raise ValueError("Deepgram API key not configured. Set DEEPGRAM_API_KEY.")

        self._api_key = resolved_key
        self._default_model = default_model or settings.default_stt_model or "nova-3"
        self._timeout = timeout or settings.stt_timeout
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            headers={
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "audio/wav",
                "Accept": "application/json",
            },
        )
        logger.info(
            "Deepgram adapter initialized (model=%s)",
            self._default_model,
        )

    @property
    def provider_name(self) -> str:
        return "deepgram"

    async def transcribe(
        self,
        audio_data: bytes,
        *,
        model: str | None = None,
        language: str = "en",
    ) -> STTResult:
        """Transcribe audio using the Deepgram REST API."""
        resolved_model = model or self._default_model
        logger.info(
            "Deepgram transcribe request (model=%s, bytes=%d)",
            resolved_model,
            len(audio_data),
        )

        try:
            response = await self._client.post(
                _DEEPGRAM_TRANSCRIBE_URL,
                content=audio_data,
                params={
                    "model": resolved_model,
                    "language": language,
                    "smart_format": "true",
                },
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                logger.error("Deepgram authentication failed")
                raise RuntimeError("Deepgram authentication failed") from e
            logger.error("Deepgram API error: HTTP %d", status)
            raise RuntimeError(f"Deepgram API error: HTTP {status}") from e

        except httpx.TimeoutException:
            logger.error("Deepgram request timed out after %.1fs", self._timeout)
            raise RuntimeError("Deepgram request timed out")

        except httpx.HTTPError as e:
            logger.error("Deepgram HTTP error: %s", e)
            raise RuntimeError(f"Deepgram HTTP error: {e}") from e

        data = response.json()
        return self._parse_response(data, resolved_model)

    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[STTResult]:
        """Streaming transcription is deferred to Phase 3 (WebRTC).

        For now, this collects all chunks and performs batch transcription.
        """
        chunks: list[bytes] = []
        async for chunk in audio_stream:
            chunks.append(chunk)

        if chunks:
            combined = b"".join(chunks)
            result = await self.transcribe(combined)
            yield result

    async def close(self) -> None:
        await self._client.aclose()

    def _parse_response(self, data: dict, model: str) -> STTResult:
        """Parse Deepgram API response into generic STTResult."""
        results = data.get("results", {})
        channels = results.get("channels", [])

        if not channels:
            return STTResult(
                text="",
                confidence=0.0,
                is_final=True,
                metadata={"model": model, "raw": {}},
            )

        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return STTResult(
                text="",
                confidence=0.0,
                is_final=True,
                metadata={"model": model},
            )

        best = alternatives[0]
        text = best.get("transcript", "")
        confidence = best.get("confidence", 0.0)
        language = results.get("language", "en")

        # Extract duration from metadata if available
        metadata_block = results.get("metadata", {})
        duration = metadata_block.get("duration")

        logger.info(
            "Deepgram transcription complete (text_len=%d, confidence=%.2f)",
            len(text),
            confidence,
        )

        return STTResult(
            text=text,
            confidence=confidence,
            language=language,
            is_final=True,
            duration_seconds=duration,
            metadata={"model": model, "provider": "deepgram"},
        )
