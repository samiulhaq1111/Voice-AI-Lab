"""ElevenLabs TTS adapter implementing TTSInterface."""

from collections.abc import AsyncIterator

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.providers.tts.interface import TTSInterface
from app.providers.types import TTSResult

_ELEVENLABS_TTS_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"  # Rachel - ElevenLabs default


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
        self._default_model = default_model or settings.default_tts_model or "eleven_monolingual_v1"
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

        logger.info(
            "ElevenLabs synthesize request (text_len=%d, model=%s)",
            len(text),
            resolved_model,
        )

        payload = {
            "text": text,
            "model_id": resolved_model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                logger.error("ElevenLabs authentication failed")
                raise RuntimeError("ElevenLabs authentication failed") from e
            if status == 422:
                logger.error("ElevenLabs invalid input: %s", e.response.text[:200])
                raise RuntimeError("ElevenLabs invalid input") from e
            logger.error("ElevenLabs API error: HTTP %d", status)
            raise RuntimeError(f"ElevenLabs API error: HTTP {status}") from e

        except httpx.TimeoutException:
            logger.error("ElevenLabs request timed out after %.1fs", self._timeout)
            raise RuntimeError("ElevenLabs request timed out")

        except httpx.HTTPError as e:
            logger.error("ElevenLabs HTTP error: %s", e)
            raise RuntimeError(f"ElevenLabs HTTP error: {e}") from e

        audio_data = response.content
        content_type = response.headers.get("content-type", "audio/mpeg")

        logger.info(
            "ElevenLabs synthesis complete (audio_bytes=%d)",
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
