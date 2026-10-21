"""Azure AI Speech TTS adapter implementing TTSInterface.

Two synthesis paths are supported:

- ``synthesize`` uses the standard Speech SDK request/response API and
  returns one complete MP3 payload (used by the turn-based voice pipeline and
  the telephony buffered fallback).
- ``stream_synthesize`` uses the SDK's incremental text-input streaming API:
  ``SpeechSynthesisRequest(SpeechSynthesisRequestInputType.TextStream)`` over
  the WebSocket v2 endpoint. Audio is forwarded from the SDK ``synthesizing``
  event as it arrives; the request is never awaited to completion before the
  first chunk is yielded.

The realtime gateway's SentenceBuffer decides when an LLM phrase is complete,
so one adapter call corresponds to one completed phrase -- never one token.
``SpeechSynthesizer`` instances are pooled per path (bounded, one request per
instance at a time) so SDK objects are reused while the gateway pre-generates
the next sentence. The Speech SDK does not expose a persistent synthesis
connection, so transport setup remains per request.

Secrets are read from ``AZURE_TTS_KEY`` / ``AZURE_TTS_REGION`` and are never
logged. The voice defaults to ``AZURE_TTS_VOICE`` (``en-US-JennyNeural`` when
unset).
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import azure.cognitiveservices.speech as speechsdk

from app.core.config import settings
from app.core.logging import logger
from app.providers.tts.interface import TTSInterface
from app.providers.types import TTSResult

_DEFAULT_VOICE = "en-US-JennyNeural"
_STREAM_ENDPOINT_TEMPLATE = (
    "wss://{region}.tts.speech.microsoft.com/cognitiveservices/websocket/v2"
)

# The realtime gateway runs at most two per-sentence syntheses concurrently.
# A matching pool keeps one SDK request per synthesizer and reuses the
# synthesizer object across sequential sentences.
_MAX_CONCURRENT_SYNTHESIS = 2

# Friendly output-format aliases -> (SpeechSynthesisOutputFormat member,
# content type). None/empty selects the realtime-compatible MP3 format.
_OUTPUT_FORMATS: dict[str, tuple[str, str]] = {
    "": ("Audio24Khz96KBitRateMonoMp3", "audio/mpeg"),
    "mp3": ("Audio24Khz96KBitRateMonoMp3", "audio/mpeg"),
    "audio/mpeg": ("Audio24Khz96KBitRateMonoMp3", "audio/mpeg"),
    "audio-24khz-96kbitrate-mono-mp3": (
        "Audio24Khz96KBitRateMonoMp3",
        "audio/mpeg",
    ),
    # Telephony-native path: raw mu-law 8 kHz, forwarded without transcoding.
    "ulaw_8000": ("Raw8Khz8BitMonoMULaw", "audio/x-mulaw"),
    "raw8khz8bitmonomulaw": ("Raw8Khz8BitMonoMULaw", "audio/x-mulaw"),
}


class AzureTTSAdapter(TTSInterface):
    """Azure AI Speech adapter for text-to-speech.

    Requires ``AZURE_TTS_KEY`` and ``AZURE_TTS_REGION``. The voice comes from
    the per-call override, the constructor default, ``AZURE_TTS_VOICE``, or
    ``en-US-JennyNeural`` -- in that order.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        region: str | None = None,
        default_voice: str | None = None,
        timeout: float | None = None,
        sdk: Any | None = None,
    ) -> None:
        resolved_key = api_key or settings.azure_tts_key
        resolved_region = region or settings.azure_tts_region
        if not resolved_key or not resolved_region:
            raise ValueError(
                "Azure Speech credentials not configured. "
                "Set AZURE_TTS_KEY and AZURE_TTS_REGION."
            )

        # Injectable SDK module keeps unit tests free of live Azure calls.
        self._sdk = sdk or speechsdk
        self._api_key = resolved_key
        self._region = resolved_region
        self._default_voice = (
            default_voice or settings.azure_tts_voice or _DEFAULT_VOICE
        )
        self._timeout = timeout or settings.tts_timeout
        self._stream_endpoint = _STREAM_ENDPOINT_TEMPLATE.format(
            region=resolved_region
        )
        self._closed = False

        # Pools are keyed by path: "stream" uses the WebSocket v2 endpoint and
        # "buffered" uses the standard endpoint. Each synthesizer handles one
        # request at a time; the semaphore bounds concurrent SDK requests.
        self._semaphores = {
            "stream": asyncio.Semaphore(_MAX_CONCURRENT_SYNTHESIS),
            "buffered": asyncio.Semaphore(_MAX_CONCURRENT_SYNTHESIS),
        }
        self._idle: dict[str, list[tuple[Any, Any]]] = {
            "stream": [],
            "buffered": [],
        }

        logger.info(
            "Azure Speech adapter initialized (region=%s, voice=%s)",
            self._region,
            self._default_voice,
        )

    @property
    def provider_name(self) -> str:
        return "azure"

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Synthesize one complete MP3 payload with the standard SDK API."""
        self._ensure_open()
        if not text.strip():
            raise ValueError("Azure TTS text must not be empty")

        resolved_voice = voice or self._default_voice
        output_format, content_type = self._resolve_output_format(None)
        started_at = time.monotonic()
        synth: Any | None = None
        config: Any | None = None
        reusable = False

        logger.info(
            "[VOICE:TTS] synthesize start provider=azure voice=%s "
            "text_length=%d",
            resolved_voice,
            len(text),
        )
        try:
            synth, config = await self._acquire_synthesizer(
                resolved_voice, output_format, "buffered"
            )
            # The SDK call and its blocking Future are both moved off the
            # event loop. Speed is intentionally not applied: the standard
            # text API does not accept a rate value.
            future = await asyncio.to_thread(synth.speak_text_async, text)
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(future.get), timeout=self._timeout
                )
            except TimeoutError as e:
                self._stop_synthesizer(synth)
                raise RuntimeError(
                    f"Azure TTS request timed out after {self._timeout:.1f}s"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Azure TTS synthesis failed: {self._redact(str(e))}"
                ) from e

            self._raise_if_not_completed(result, "synthesis")
            audio_data = bytes(getattr(result, "audio_data", b"") or b"")
            if not audio_data:
                raise RuntimeError("Azure TTS returned no audio")

            reusable = True
            duration_ms = (time.monotonic() - started_at) * 1000
            logger.info(
                "[VOICE:TTS] synthesize success provider=azure bytes=%d "
                "duration_ms=%.0f",
                len(audio_data),
                duration_ms,
            )
            return TTSResult(
                audio_data=audio_data,
                content_type=content_type,
                metadata={
                    "provider": "azure",
                    "voice": resolved_voice,
                    "region": self._region,
                    "output_format": "mp3",
                },
            )
        except Exception as e:
            logger.error(
                "[VOICE:TTS] failure provider=azure duration_ms=%.0f "
                "error_type=%s error=%s",
                (time.monotonic() - started_at) * 1000,
                type(e).__name__,
                self._redact(str(e)),
            )
            raise
        finally:
            if synth is not None and config is not None:
                self._release_synthesizer(synth, config, "buffered", reusable)

    async def stream_synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        speed: float = 1.0,
        output_format: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream Azure audio chunks for one completed phrase.

        Uses ``SpeechSynthesisRequestInputType.TextStream`` (SDK >= 1.37) on
        the WebSocket v2 endpoint. ``synthesizing`` events are bridged from
        the SDK callback thread to the asyncio event loop, so the first chunk
        is yielded while synthesis is still running.
        """
        self._ensure_open()
        resolved_voice = voice or self._default_voice
        resolved_format, content_type = self._resolve_output_format(output_format)
        if not text.strip():
            return

        started_at = time.monotonic()
        text_submitted_at: float | None = None
        synth: Any | None = None
        config: Any | None = None
        reusable = False
        first_audio_at: float | None = None
        chunks = 0
        total_bytes = 0
        closed = False
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _put(item: tuple[str, Any]) -> None:
            if not closed:
                queue.put_nowait(item)

        def _emit(item: tuple[str, Any]) -> None:
            """Cross the SDK callback thread -> event loop boundary safely."""
            try:
                loop.call_soon_threadsafe(_put, item)
            except RuntimeError:
                # The session/event loop was already torn down.
                pass

        def _on_synthesizing(evt: Any) -> None:
            data = getattr(evt.result, "audio_data", b"") or b""
            if data:
                _emit(("audio", bytes(data)))

        def _on_completed(evt: Any) -> None:
            # Audio is delivered exclusively through ``synthesizing`` events;
            # the completion result is used only as the terminal marker.
            _emit(("completed", evt.result))

        def _on_canceled(evt: Any) -> None:
            _emit(("canceled", evt.result))

        logger.info(
            "[VOICE:TTS] stream start provider=azure voice=%s chars=%d "
            "format=%s content_type=%s",
            resolved_voice,
            len(text),
            output_format or "mp3",
            content_type,
        )

        try:
            synth, config = await self._acquire_synthesizer(
                resolved_voice, resolved_format, "stream"
            )
            synth.synthesizing.connect(_on_synthesizing)
            synth.synthesis_completed.connect(_on_completed)
            synth.synthesis_canceled.connect(_on_canceled)

            request = self._sdk.SpeechSynthesisRequest(
                input_type=self._sdk.SpeechSynthesisRequestInputType.TextStream
            )
            try:
                # Official SDK order: start the request first, then write the
                # text and close the input stream. Both calls run off-loop.
                text_submitted_at = time.monotonic()
                future = await asyncio.to_thread(
                    self._start_text_stream, synth, request, text
                )
            except Exception as e:
                raise RuntimeError(
                    f"Azure TTS stream start failed: {self._redact(str(e))}"
                ) from e

            async with asyncio.timeout(self._timeout):
                while True:
                    kind, payload = await queue.get()
                    if kind == "audio":
                        if first_audio_at is None:
                            first_audio_at = time.monotonic()
                            logger.info(
                                "[VOICE:TTS] stream first_chunk provider=azure "
                                "bytes=%d ttfa_ms=%.0f",
                                len(payload),
                                (first_audio_at - (text_submitted_at or started_at))
                                * 1000,
                            )
                        chunks += 1
                        total_bytes += len(payload)
                        yield payload
                    elif kind == "completed":
                        self._raise_if_not_completed(payload, "stream")
                        if chunks == 0:
                            raise RuntimeError(
                                "Azure TTS stream returned no audio"
                            )
                        reusable = True
                        # The completion event already arrived; get() returns
                        # immediately and releases the SDK future.
                        try:
                            await asyncio.to_thread(future.get)
                        except Exception as e:
                            logger.debug(
                                "[VOICE:TTS] stream future finalize failed "
                                "error_type=%s",
                                type(e).__name__,
                            )
                        break
                    else:  # canceled
                        self._raise_if_not_completed(payload, "stream")
        except TimeoutError as e:
            raise RuntimeError(
                f"Azure TTS stream timed out after {self._timeout:.1f}s"
            ) from e
        except asyncio.CancelledError:
            logger.info(
                "[VOICE:TTS] stream cancelled provider=azure chunks=%d "
                "bytes=%d",
                chunks,
                total_bytes,
            )
            raise
        except Exception as e:
            logger.error(
                "[VOICE:TTS] stream failure provider=azure chunks=%d "
                "bytes=%d error_type=%s error=%s",
                chunks,
                total_bytes,
                type(e).__name__,
                self._redact(str(e)),
            )
            raise
        finally:
            closed = True
            if synth is not None and config is not None:
                self._disconnect_handlers(synth)
                self._release_synthesizer(synth, config, "stream", reusable)
            logger.debug(
                "[VOICE:TTS] stream finished provider=azure ok=%s chunks=%d "
                "bytes=%d ttfa_ms=%s total_ms=%.0f",
                reusable,
                chunks,
                total_bytes,
                f"{(first_audio_at - (text_submitted_at or started_at)) * 1000:.0f}"
                if first_audio_at is not None
                else "n/a",
                (time.monotonic() - started_at) * 1000,
            )

    async def close(self) -> None:
        """Release idle synthesizers; in-flight requests close on release."""
        self._closed = True
        for mode in ("stream", "buffered"):
            idle, self._idle[mode] = self._idle[mode], []
            for synth, _config in idle:
                self._stop_synthesizer(synth)
        logger.debug("Azure Speech adapter closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Azure TTS adapter is closed")

    def _redact(self, message: str) -> str:
        """Remove the subscription key from any SDK error text."""
        if not message:
            return message
        return message.replace(self._api_key, "***")

    def _resolve_output_format(self, output_format: str | None) -> tuple[Any, str]:
        key = (output_format or "").strip().lower()
        try:
            enum_name, content_type = _OUTPUT_FORMATS[key]
        except KeyError as e:
            supported = ", ".join(sorted(name for name in _OUTPUT_FORMATS if name))
            raise ValueError(
                f"Unsupported Azure TTS output format: '{output_format}'. "
                f"Supported: {supported}"
            ) from e
        return getattr(self._sdk.SpeechSynthesisOutputFormat, enum_name), content_type

    async def _acquire_synthesizer(
        self, voice: str, output_format: Any, mode: str
    ) -> tuple[Any, Any]:
        """Check out an idle synthesizer or create one off-loop."""
        await self._semaphores[mode].acquire()
        try:
            if self._idle[mode]:
                synth, config = self._idle[mode].pop()
                config.speech_synthesis_voice_name = voice
                config.set_speech_synthesis_output_format(output_format)
                return synth, config
            return await asyncio.to_thread(
                self._create_synthesizer, voice, output_format, mode
            )
        except BaseException:
            self._semaphores[mode].release()
            raise

    def _create_synthesizer(
        self, voice: str, output_format: Any, mode: str
    ) -> tuple[Any, Any]:
        """Create one SDK synthesizer (called in a worker thread)."""
        if mode == "stream":
            config = self._sdk.SpeechConfig(
                endpoint=self._stream_endpoint,
                subscription=self._api_key,
            )
        else:
            config = self._sdk.SpeechConfig(
                subscription=self._api_key,
                region=self._region,
            )
        config.speech_synthesis_voice_name = voice
        config.set_speech_synthesis_output_format(output_format)
        # audio_config=None drops SDK playback on the server.
        synth = self._sdk.SpeechSynthesizer(
            speech_config=config, audio_config=None
        )
        return synth, config

    def _release_synthesizer(
        self, synth: Any, config: Any, mode: str, reusable: bool
    ) -> None:
        if reusable and not self._closed:
            self._idle[mode].append((synth, config))
        else:
            self._stop_synthesizer(synth)
        self._semaphores[mode].release()

    @staticmethod
    def _start_text_stream(synth: Any, request: Any, text: str) -> Any:
        """Start a TextStream request, feed one phrase, and close input.

        Runs in a worker thread. The official SDK sample starts the request
        before writing text; the returned future is kept alive by the caller
        until the completion event arrives.
        """
        future = synth.speak_async(request)
        request.input_stream.write(text)
        request.input_stream.close()
        return future

    @staticmethod
    def _stop_synthesizer(synth: Any) -> None:
        try:
            synth.stop_speaking_async()
        except Exception:
            pass

    @staticmethod
    def _disconnect_handlers(synth: Any) -> None:
        for signal in (
            synth.synthesizing,
            synth.synthesis_completed,
            synth.synthesis_canceled,
        ):
            disconnect = getattr(signal, "disconnect_all", None)
            if callable(disconnect):
                try:
                    disconnect()
                except Exception:
                    pass

    def _raise_if_not_completed(self, result: Any, context: str) -> None:
        reason_name = self._reason_name(result)
        if reason_name == "SynthesizingAudioCompleted":
            return
        details = getattr(result, "cancellation_details", None)
        error_code = getattr(details, "error_code", None)
        error_details = getattr(details, "error_details", None)
        raise RuntimeError(
            self._redact(
                f"Azure TTS {context} canceled: reason={reason_name} "
                f"code={error_code or 'unknown'} "
                f"details={error_details or 'none'}"
            )
        )

    @staticmethod
    def _reason_name(result: Any) -> str:
        reason = getattr(result, "reason", None)
        if reason is None:
            return "unknown"
        return getattr(reason, "name", str(reason))
