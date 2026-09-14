"""Deepgram STT adapter implementing STTInterface."""

import time
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
            "Deepgram adapter initialized model=%s api_key_configured=%s auth_scheme=Token",
            self._default_model,
            bool(self._api_key),
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
        content_type: str = "audio/wav",
    ) -> STTResult:
        """Transcribe audio using the Deepgram REST API.

        Args:
            audio_data: Raw audio bytes.
            model: Model identifier override.
            language: Expected language code.
            content_type: MIME type of the audio data
                (e.g. 'audio/wav', 'audio/webm', 'audio/mp4').
                Deepgram supports many formats; defaults to wav.
        """
        resolved_model = model or self._default_model
        logger.info(
            "[DEEPGRAM] Request started model=%s content_type=%s "
            "audio_bytes=%d api_key_configured=%s",
            resolved_model,
            content_type,
            len(audio_data),
            bool(self._api_key),
        )

        request_start = time.monotonic()
        try:
            response = await self._client.post(
                _DEEPGRAM_TRANSCRIBE_URL,
                content=audio_data,
                headers={"Content-Type": content_type},
                params={
                    "model": resolved_model,
                    "language": language,
                    "smart_format": "true",
                },
            )
            duration_ms = (time.monotonic() - request_start) * 1000
            logger.info(
                "[DEEPGRAM] Response received status=%d duration_ms=%.0f",
                response.status_code,
                duration_ms,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            duration_ms = (time.monotonic() - request_start) * 1000
            if status == 401:
                logger.error(
                    "[DEEPGRAM] Authentication failed status=401 endpoint=%s duration_ms=%.0f",
                    _DEEPGRAM_TRANSCRIBE_URL,
                    duration_ms,
                )
                raise RuntimeError("Deepgram authentication failed") from e
            # Try to extract safe error message from response body
            safe_detail = ""
            try:
                body = e.response.json()
                safe_detail = body.get("err_msg", "") or body.get("error", "")
            except Exception:
                pass
            logger.error(
                "[DEEPGRAM] Provider request failed status=%d error_type=%s endpoint=%s detail=%s",
                status,
                type(e).__name__,
                _DEEPGRAM_TRANSCRIBE_URL,
                safe_detail or "(no detail)",
            )
            raise RuntimeError(f"Deepgram API error: HTTP {status}") from e

        except httpx.TimeoutException:
            logger.error(
                "[DEEPGRAM] Request timed out after %.1fs endpoint=%s",
                self._timeout,
                _DEEPGRAM_TRANSCRIBE_URL,
            )
            raise RuntimeError("Deepgram request timed out")

        except httpx.HTTPError as e:
            logger.error(
                "[DEEPGRAM] HTTP error type=%s endpoint=%s",
                type(e).__name__,
                _DEEPGRAM_TRANSCRIBE_URL,
            )
            raise RuntimeError(f"Deepgram HTTP error: {e}") from e

        data = response.json()
        result = self._parse_response(data, resolved_model)
        if result.text:
            logger.info(
                "[DEEPGRAM] Transcription successful transcript_length=%d",
                len(result.text),
            )
        else:
            logger.warning(
                "[DEEPGRAM] Empty transcript returned results_present=%s channels_present=%s",
                bool(data.get("results")),
                bool(data.get("results", {}).get("channels")),
            )
        return result

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
            "[DEEPGRAM] Response parsed transcript_length=%d confidence=%.2f duration_seconds=%s",
            len(text),
            confidence,
            f"{duration:.2f}" if duration else "none",
        )

        return STTResult(
            text=text,
            confidence=confidence,
            language=language,
            is_final=True,
            duration_seconds=duration,
            metadata={"model": model, "provider": "deepgram"},
        )
