"""ElevenLabs TTS adapter implementing TTSInterface."""

import time
from collections.abc import AsyncIterator

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.providers.tts.interface import TTSInterface
from app.providers.types import TTSResult

_ELEVENLABS_TTS_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_ELEVENLABS_TTS_STREAM_URL_TEMPLATE = (
    "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
)
_DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"  # API-compatible Free tier voice
_DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"  # Free-tier compatible


class ElevenLabsAdapter(TTSInterface):
    """ElevenLabs adapter for text-to-speech.

    Uses the ElevenLabs REST API.
    Requires ELEVENLABS_API_KEY in environment configuration.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_voice: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        resolved_key = api_key or settings.elevenlabs_api_key
        if not resolved_key:
            raise ValueError("ElevenLabs API key not configured. Set ELEVENLABS_API_KEY.")

        self._api_key = resolved_key
        self._default_voice = default_voice or settings.default_tts_voice or _DEFAULT_VOICE
        self._default_model = default_model or settings.default_tts_model or "eleven_flash_v2_5"
        self._timeout = timeout or settings.tts_timeout
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            headers={
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
        )
        logger.info(
            "ElevenLabs adapter initialized (model=%s, voice=%s)",
            self._default_model,
            self._default_voice[:8] + "...",
        )

    @property
    def provider_name(self) -> str:
        return "elevenlabs"

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Convert text to speech using the ElevenLabs REST API."""
        resolved_voice = voice or self._default_voice
        resolved_model = model or self._default_model
        url = _ELEVENLABS_TTS_URL_TEMPLATE.format(voice_id=resolved_voice)

        # output_format must be a query parameter per ElevenLabs API spec
        params = {"output_format": _DEFAULT_OUTPUT_FORMAT}

        payload = {
            "text": text,
            "model_id": resolved_model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        logger.info(
            "[VOICE:TTS] synthesize start provider=elevenlabs "
            "model=%s voice=%s text_length=%d",
            resolved_model,
            resolved_voice[:8] + "...",
            len(text),
        )

        request_start = time.monotonic()
        try:
            response = await self._client.post(url, json=payload, params=params)
            duration_ms = (time.monotonic() - request_start) * 1000
            logger.info(
                "[VOICE:TTS] response status=%d content_type=%s "
                "response_bytes=%d duration_ms=%.0f",
                response.status_code,
                response.headers.get("content-type", "unknown"),
                len(response.content),
                duration_ms,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            duration_ms = (time.monotonic() - request_start) * 1000
            # Extract structured ElevenLabs error details for diagnostics.
            # ElevenLabs returns: {"detail": {"message": ..., "status": ...}}
            # or {"detail": "string"} or other shapes.
            error_detail = ""
            error_code = ""
            request_id = ""
            try:
                body = e.response.json()
                if isinstance(body, dict):
                    detail = body.get("detail", "")
                    if isinstance(detail, dict):
                        error_detail = detail.get("message", "")
                        error_code = detail.get("status", detail.get("code", ""))
                    else:
                        error_detail = str(detail)
                    # request_id may be at top level or in detail
                    request_id = (
                        body.get("request_id", "")
                        or (detail.get("request_id", "") if isinstance(detail, dict) else "")
                    )
                else:
                    error_detail = str(body)[:200]
            except Exception:
                error_detail = e.response.text[:200] if e.response.text else ""

            # Log full diagnostic breakdown for every non-2xx response
            logger.error(
                "[VOICE:TTS] failure status=%d code=%s request_id=%s "
                "duration_ms=%.0f detail=%s",
                status,
                error_code or "(none)",
                request_id or "(none)",
                duration_ms,
                error_detail[:200] if error_detail else "(empty)",
            )

            if status == 401:
                raise RuntimeError(
                    f"ElevenLabs authentication failed (401): "
                    f"{error_code or 'unknown'} — {error_detail or 'no detail'}"
                ) from e
            if status == 402:
                raise RuntimeError(
                    f"ElevenLabs billing/entitlement error (402): "
                    f"{error_code or 'unknown'} — {error_detail or 'no detail'}"
                ) from e
            if status == 403:
                raise RuntimeError(
                    f"ElevenLabs authorization failed (403): "
                    f"{error_code or 'unknown'} — {error_detail or 'no detail'}"
                ) from e
            if status == 422:
                raise RuntimeError(
                    f"ElevenLabs invalid input (422): "
                    f"{error_code or 'unknown'} — {error_detail or 'no detail'}"
                ) from e
            if status == 429:
                raise RuntimeError(
                    f"ElevenLabs rate limit exceeded (429): "
                    f"{error_code or 'unknown'} — {error_detail or 'no detail'}"
                ) from e
            if status >= 500:
                raise RuntimeError(
                    f"ElevenLabs server error ({status}): "
                    f"{error_code or 'unknown'} — {error_detail or 'no detail'}"
                ) from e
            raise RuntimeError(
                f"ElevenLabs API error ({status}): "
                f"{error_code or 'unknown'} — {error_detail or 'no detail'}"
            ) from e

        except httpx.TimeoutException:
            duration_ms = (time.monotonic() - request_start) * 1000
            logger.error(
                "[VOICE:TTS] provider timeout after %.1fs duration_ms=%.0f",
                self._timeout,
                duration_ms,
            )
            raise RuntimeError("ElevenLabs request timed out")

        except httpx.HTTPError as e:
            logger.error(
                "[VOICE:TTS] provider network error type=%s",
                type(e).__name__,
            )
            raise RuntimeError(f"ElevenLabs HTTP error: {e}") from e

        audio_data = response.content
        content_type = response.headers.get("content-type", "audio/mpeg")

        logger.info(
            "[VOICE:TTS] synthesize success bytes=%d duration_ms=%.0f",
            len(audio_data),
            (time.monotonic() - request_start) * 1000,
        )

        return TTSResult(
            audio_data=audio_data,
            content_type=content_type,
            metadata={
                "provider": "elevenlabs",
                "model": resolved_model,
                "voice": resolved_voice,
            },
        )

    async def stream_synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        speed: float = 1.0,
        output_format: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream synthesized audio chunks from ElevenLabs.

        Uses the dedicated ElevenLabs streaming endpoint, which returns
        chunked HTTP audio so playback can begin before generation ends.

        Args:
            text: Text to synthesize.
            voice: Voice identifier override.
            model: Model identifier override.
            speed: Speech speed multiplier (unused by ElevenLabs).
            output_format: Audio format identifier. Telephony callers pass
                'ulaw_8000' to receive raw mu-law 8kHz bytes that need no
                transcoding. Defaults to the adapter MP3 format.

        Yields:
            Raw audio byte chunks in the requested format.
        """
        resolved_voice = voice or self._default_voice
        resolved_model = model or self._default_model
        resolved_format = output_format or _DEFAULT_OUTPUT_FORMAT
        url = _ELEVENLABS_TTS_STREAM_URL_TEMPLATE.format(voice_id=resolved_voice)

        params = {"output_format": resolved_format}

        payload = {
            "text": text,
            "model_id": resolved_model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        stream_start = time.monotonic()
        first_chunk_ms: float | None = None
        total_bytes = 0
        chunks = 0

        try:
            async with self._client.stream(
                "POST",
                url,
                json=payload,
                params=params,
                headers={"Accept": "application/octet-stream"},
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    chunks += 1
                    total_bytes += len(chunk)
                    if first_chunk_ms is None:
                        first_chunk_ms = (time.monotonic() - stream_start) * 1000
                        logger.info(
                            "[VOICE:TTS] stream first_chunk provider=elevenlabs "
                            "format=%s bytes=%d ttfa_ms=%.0f",
                            resolved_format,
                            len(chunk),
                            first_chunk_ms,
                        )
                    yield chunk

            logger.info(
                "[VOICE:TTS] stream complete provider=elevenlabs format=%s "
                "chunks=%d bytes=%d ttfa_ms=%s total_ms=%.0f",
                resolved_format,
                chunks,
                total_bytes,
                f"{first_chunk_ms:.0f}" if first_chunk_ms is not None else "n/a",
                (time.monotonic() - stream_start) * 1000,
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                "ElevenLabs stream error: HTTP %d (format=%s)",
                e.response.status_code,
                resolved_format,
            )
            raise RuntimeError(
                f"ElevenLabs stream error: HTTP {e.response.status_code}"
            ) from e

        except httpx.HTTPError as e:
            logger.error("ElevenLabs stream HTTP error: %s", e)
            raise RuntimeError(f"ElevenLabs stream error: {e}") from e

    async def close(self) -> None:
        await self._client.aclose()
