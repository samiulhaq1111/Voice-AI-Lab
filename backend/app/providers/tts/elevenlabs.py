"""ElevenLabs TTS adapter implementing TTSInterface."""

import hashlib
import time
from collections.abc import AsyncIterator

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.providers.tts.interface import TTSInterface
from app.providers.types import TTSResult

_ELEVENLABS_TTS_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"  # API-compatible Free tier voice
_DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"  # Free-tier compatible


def _safe_key_fingerprint(key: str) -> str:
    """Return a short non-reversible hash fingerprint of an API key."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


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
            "[VOICE:TTS] provider=elevenlabs endpoint=%s voice_id=%s "
            "model_id=%s output_format=%s text_length=%d "
            "api_key_configured=%s api_key_fingerprint=%s",
            url,
            resolved_voice[:8] + "...",
            resolved_model,
            _DEFAULT_OUTPUT_FORMAT,
            len(text),
            bool(self._api_key),
            _safe_key_fingerprint(self._api_key),
        )
        logger.info("[VOICE:TTS] request_started")

        request_start = time.monotonic()
        try:
            response = await self._client.post(url, json=payload, params=params)
            duration_ms = (time.monotonic() - request_start) * 1000
            logger.info(
                "[VOICE:TTS] response_status=%d content_type=%s response_bytes=%d duration_ms=%.0f",
                response.status_code,
                response.headers.get("content-type", "unknown"),
                len(response.content),
                duration_ms,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            duration_ms = (time.monotonic() - request_start) * 1000
            # Extract safe error detail from response body for diagnostics
            error_detail = ""
            try:
                body = e.response.json()
                # ElevenLabs returns {"detail": {"message": "...", "status": "..."}}
                if isinstance(body, dict):
                    detail = body.get("detail", "")
                    if isinstance(detail, dict):
                        error_detail = detail.get("message", str(detail))
                    else:
                        error_detail = str(detail)
                else:
                    error_detail = str(body)[:200]
            except Exception:
                error_detail = e.response.text[:200] if e.response.text else ""

            if status == 400:
                logger.error(
                    "[VOICE:TTS] provider_error status=400 duration_ms=%.0f "
                    "model=%s voice=%s text_length=%d error_detail=%s",
                    duration_ms,
                    resolved_model,
                    resolved_voice[:8] + "...",
                    len(text),
                    error_detail,
                )
                raise RuntimeError(f"ElevenLabs API error: HTTP 400 — {error_detail}") from e
            if status == 401:
                logger.error(
                    "[ELEVENLABS] Authentication failed status=401 duration_ms=%.0f",
                    duration_ms,
                )
                raise RuntimeError("ElevenLabs authentication failed") from e
            if status == 402:
                logger.error(
                    "[VOICE:TTS] provider_error status=402 duration_ms=%.0f error=%s",
                    duration_ms,
                    error_detail,
                )
                raise RuntimeError(f"ElevenLabs billing/entitlement error: {error_detail}") from e
            if status == 403:
                logger.error(
                    "[ELEVENLABS] Authorization failed status=403 duration_ms=%.0f",
                    duration_ms,
                )
                raise RuntimeError("ElevenLabs authorization failed") from e
            if status == 422:
                logger.error(
                    "[ELEVENLABS] Invalid input status=422 duration_ms=%.0f error=%s",
                    duration_ms,
                    error_detail,
                )
                raise RuntimeError("ElevenLabs invalid input") from e
            if status == 429:
                logger.error(
                    "[ELEVENLABS] Rate limit exceeded status=429 duration_ms=%.0f",
                    duration_ms,
                )
                raise RuntimeError("ElevenLabs rate limit exceeded") from e
            logger.error(
                "[ELEVENLABS] Request failed status=%d duration_ms=%.0f error=%s",
                status,
                duration_ms,
                error_detail,
            )
            raise RuntimeError(f"ElevenLabs API error: HTTP {status}") from e

        except httpx.TimeoutException:
            duration_ms = (time.monotonic() - request_start) * 1000
            logger.error(
                "[ELEVENLABS] Request timed out after %.1fs duration_ms=%.0f",
                self._timeout,
                duration_ms,
            )
            raise RuntimeError("ElevenLabs request timed out")

        except httpx.HTTPError as e:
            logger.error("[ELEVENLABS] HTTP error type=%s", type(e).__name__)
            raise RuntimeError(f"ElevenLabs HTTP error: {e}") from e

        audio_data = response.content
        content_type = response.headers.get("content-type", "audio/mpeg")

        logger.info(
            "[ELEVENLABS] Synthesis complete audio_bytes=%d",
            len(audio_data),
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
    ) -> AsyncIterator[bytes]:
        """Stream synthesized audio chunks from ElevenLabs."""
        resolved_voice = voice or self._default_voice
        resolved_model = model or self._default_model
        url = _ELEVENLABS_TTS_URL_TEMPLATE.format(voice_id=resolved_voice)

        params = {"output_format": _DEFAULT_OUTPUT_FORMAT}

        payload = {
            "text": text,
            "model_id": resolved_model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        try:
            async with self._client.stream(
                "POST",
                url,
                json=payload,
                params=params,
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk

        except httpx.HTTPStatusError as e:
            logger.error(
                "ElevenLabs stream error: HTTP %d",
                e.response.status_code,
            )
            raise RuntimeError(f"ElevenLabs stream error: HTTP {e.response.status_code}") from e

        except httpx.HTTPError as e:
            logger.error("ElevenLabs stream HTTP error: %s", e)
            raise RuntimeError(f"ElevenLabs stream error: {e}") from e

    async def close(self) -> None:
        await self._client.aclose()
