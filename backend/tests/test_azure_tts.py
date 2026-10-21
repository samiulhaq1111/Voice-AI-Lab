"""Targeted tests for the Azure AI Speech TTS adapter.

The Azure SDK is replaced with an in-memory double, so these tests never
make Azure network calls or paid requests. They cover configuration,
buffered synthesis, TextStream streaming, cancellation/lifecycle cleanup,
SDK cancellation errors, factory registration, and secret-free catalogues.
"""

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings, settings
from app.providers.factory import get_available_providers, get_tts_provider
from app.providers.tts.azure import AzureTTSAdapter


class _FakeSignal:
    """Minimal EventSignal double (connect / disconnect_all / emit)."""

    def __init__(self) -> None:
        self._callbacks: list[Any] = []

    def connect(self, callback: Any) -> None:
        self._callbacks.append(callback)

    def disconnect_all(self) -> None:
        self._callbacks.clear()

    def emit(self, event: Any) -> None:
        for callback in list(self._callbacks):
            callback(event)


class _FakeCancellationDetails:
    error_code = "ServiceTimeout"
    error_details = "failed with test-key"


class _FakeResult:
    def __init__(
        self,
        audio_data: bytes = b"",
        reason: str = "SynthesizingAudioCompleted",
        cancellation_details: Any | None = None,
    ) -> None:
        self.audio_data = audio_data
        self.reason = reason
        self.cancellation_details = cancellation_details


class _FakeFuture:
    def __init__(self, result: _FakeResult) -> None:
        self._result = result

    def get(self) -> _FakeResult:
        return self._result


class _FakeSpeechConfig:
    def __init__(
        self,
        subscription: str | None = None,
        region: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.subscription = subscription
        self.region = region
        self.endpoint = endpoint
        self.speech_synthesis_voice_name: str | None = None
        self.output_format: Any | None = None

    def set_speech_synthesis_output_format(self, output_format: Any) -> None:
        self.output_format = output_format


class _FakeInputStream:
    def __init__(self) -> None:
        self.written: list[str] = []
        self.closed = False

    def write(self, text: str) -> None:
        self.written.append(text)

    def close(self) -> None:
        self.closed = True


class _FakeSpeechSynthesisRequest:
    def __init__(self, input_type: str) -> None:
        self.input_type = input_type
        self.input_stream = _FakeInputStream()


class _FakeSpeechSynthesizer:
    """SDK synthesizer double that emits events from a worker thread."""

    instances: list["_FakeSpeechSynthesizer"] = []
    block_stream_completion = False
    emit_canceled = False
    emit_no_audio = False

    def __init__(self, speech_config: _FakeSpeechConfig, audio_config: Any = None) -> None:
        self.config = speech_config
        self.audio_config = audio_config
        self.synthesizing = _FakeSignal()
        self.synthesis_completed = _FakeSignal()
        self.synthesis_canceled = _FakeSignal()
        self.spoken_texts: list[str] = []
        self.stream_requests: list[_FakeSpeechSynthesisRequest] = []
        self.stop_calls = 0
        self._complete_event = threading.Event()
        type(self).instances.append(self)

    def speak_text_async(self, text: str) -> _FakeFuture:
        self.spoken_texts.append(text)
        return _FakeFuture(_FakeResult(audio_data=b"complete-mp3"))

    def speak_async(self, request: _FakeSpeechSynthesisRequest) -> _FakeFuture:
        self.stream_requests.append(request)

        def _run() -> None:
            chunks = (
                () if type(self).emit_no_audio else (b"chunk-1", b"chunk-2")
            )
            for chunk in chunks:
                self.synthesizing.emit(
                    SimpleNamespace(result=_FakeResult(audio_data=chunk))
                )
            if type(self).block_stream_completion:
                self._complete_event.wait(5)
            if type(self).emit_canceled:
                self.synthesis_canceled.emit(
                    SimpleNamespace(
                        result=_FakeResult(
                            reason="Canceled",
                            cancellation_details=_FakeCancellationDetails(),
                        )
                    )
                )
            else:
                self.synthesis_completed.emit(
                    SimpleNamespace(result=_FakeResult())
                )

        threading.Thread(target=_run, daemon=True).start()
        return _FakeFuture(_FakeResult())

    def stop_speaking_async(self) -> None:
        self.stop_calls += 1
        self._complete_event.set()


class _FakeSDK:
    SpeechConfig = _FakeSpeechConfig
    SpeechSynthesizer = _FakeSpeechSynthesizer
    SpeechSynthesisRequest = _FakeSpeechSynthesisRequest

    class SpeechSynthesisRequestInputType:
        TextStream = "TextStream"

    class SpeechSynthesisOutputFormat:
        Audio24Khz96KBitRateMonoMp3 = "mp3"
        Raw8Khz8BitMonoMULaw = "ulaw"


@pytest.fixture(autouse=True)
def _reset_fake_sdk() -> None:
    _FakeSpeechSynthesizer.instances.clear()
    _FakeSpeechSynthesizer.block_stream_completion = False
    _FakeSpeechSynthesizer.emit_canceled = False
    _FakeSpeechSynthesizer.emit_no_audio = False
    yield
    _FakeSpeechSynthesizer.instances.clear()
    _FakeSpeechSynthesizer.block_stream_completion = False
    _FakeSpeechSynthesizer.emit_canceled = False
    _FakeSpeechSynthesizer.emit_no_audio = False


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> AzureTTSAdapter:
    monkeypatch.setattr(settings, "azure_tts_key", "test-key")
    monkeypatch.setattr(settings, "azure_tts_region", "eastus")
    monkeypatch.setattr(settings, "azure_tts_voice", "en-US-JennyNeural")
    return AzureTTSAdapter(sdk=_FakeSDK())


def test_missing_credentials_raise_without_leaking_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "azure_tts_key", "")
    monkeypatch.setattr(settings, "azure_tts_region", "")

    with pytest.raises(ValueError) as exc_info:
        AzureTTSAdapter(api_key="", region="", sdk=_FakeSDK())

    message = str(exc_info.value)
    assert "AZURE_TTS_KEY" in message
    assert "AZURE_TTS_REGION" in message
    assert "test-key" not in message


def test_azure_configuration_requires_key_and_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZURE_TTS_KEY", raising=False)
    monkeypatch.delenv("AZURE_TTS_REGION", raising=False)
    monkeypatch.delenv("AZURE_TTS_VOICE", raising=False)

    configured = Settings(
        _env_file=None,
        azure_tts_key="key",
        azure_tts_region="eastus",
    )
    assert configured.azure_tts_voice == "en-US-JennyNeural"
    assert configured.is_provider_configured("azure") is True

    missing_region = Settings(
        _env_file=None,
        azure_tts_key="key",
        azure_tts_region="",
    )
    assert missing_region.is_provider_configured("azure") is False


@pytest.mark.asyncio
async def test_synthesize_returns_complete_mp3(adapter: AzureTTSAdapter) -> None:
    result = await adapter.synthesize("Hello world", voice="en-US-GuyNeural")

    assert result.audio_data == b"complete-mp3"
    assert result.content_type == "audio/mpeg"
    assert result.metadata["provider"] == "azure"
    assert result.metadata["voice"] == "en-US-GuyNeural"

    synth = _FakeSpeechSynthesizer.instances[0]
    assert synth.audio_config is None
    assert synth.config.subscription == "test-key"
    assert synth.config.region == "eastus"
    assert synth.config.endpoint is None
    assert synth.config.speech_synthesis_voice_name == "en-US-GuyNeural"
    assert synth.config.output_format == "mp3"
    assert synth.spoken_texts == ["Hello world"]


@pytest.mark.asyncio
async def test_stream_uses_text_stream_and_reuses_synthesizer(
    adapter: AzureTTSAdapter,
) -> None:
    chunks = [
        chunk
        async for chunk in adapter.stream_synthesize("Hello", output_format="mp3")
    ]
    assert chunks == [b"chunk-1", b"chunk-2"]

    synth = _FakeSpeechSynthesizer.instances[0]
    request = synth.stream_requests[0]
    assert request.input_type == "TextStream"
    assert request.input_stream.written == ["Hello"]
    assert request.input_stream.closed is True
    assert synth.config.subscription == "test-key"
    assert synth.config.region is None
    assert synth.config.endpoint == (
        "wss://eastus.tts.speech.microsoft.com/cognitiveservices/websocket/v2"
    )
    assert synth.config.speech_synthesis_voice_name == "en-US-JennyNeural"
    assert synth.config.output_format == "mp3"

    # A sequential sentence reuses the pooled synthesizer/connection.
    second = [chunk async for chunk in adapter.stream_synthesize("Again")]
    assert second == [b"chunk-1", b"chunk-2"]
    assert len(_FakeSpeechSynthesizer.instances) == 1


@pytest.mark.asyncio
async def test_stream_telephony_ulaw_format(adapter: AzureTTSAdapter) -> None:
    chunks = [
        chunk
        async for chunk in adapter.stream_synthesize(
            "Hello", output_format="ulaw_8000"
        )
    ]
    assert chunks == [b"chunk-1", b"chunk-2"]
    assert _FakeSpeechSynthesizer.instances[0].config.output_format == "ulaw"


@pytest.mark.asyncio
async def test_stream_cancellation_stops_and_discards_synthesizer(
    adapter: AzureTTSAdapter,
) -> None:
    _FakeSpeechSynthesizer.block_stream_completion = True
    stream = adapter.stream_synthesize("Hello")

    assert await anext(stream) == b"chunk-1"
    synth = _FakeSpeechSynthesizer.instances[0]
    await stream.aclose()
    await asyncio.sleep(0.05)

    assert synth.stop_calls == 1
    assert adapter._idle["stream"] == []

    # A cancelled request is not returned to the pool: the next call creates
    # a fresh synthesizer instead of racing the still-closing SDK request.
    _FakeSpeechSynthesizer.block_stream_completion = False
    second = [chunk async for chunk in adapter.stream_synthesize("Again")]
    assert second == [b"chunk-1", b"chunk-2"]
    assert len(_FakeSpeechSynthesizer.instances) == 2


@pytest.mark.asyncio
async def test_stream_completion_without_audio_raises(
    adapter: AzureTTSAdapter,
) -> None:
    _FakeSpeechSynthesizer.emit_no_audio = True

    with pytest.raises(RuntimeError, match="no audio"):
        async for _ in adapter.stream_synthesize("Hello"):
            pass

    assert adapter._idle["stream"] == []


@pytest.mark.asyncio
async def test_stream_canceled_event_raises_and_redacts_key(
    adapter: AzureTTSAdapter,
) -> None:
    _FakeSpeechSynthesizer.emit_canceled = True

    with pytest.raises(RuntimeError) as exc_info:
        async for _ in adapter.stream_synthesize("Hello"):
            pass

    message = str(exc_info.value)
    assert "canceled" in message
    assert "test-key" not in message
    assert "***" in message
    assert _FakeSpeechSynthesizer.instances[0].stop_calls == 1


def test_factory_registers_azure_without_changing_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "azure_tts_key", "secret-azure-key")
    monkeypatch.setattr(settings, "azure_tts_region", "eastus")
    monkeypatch.setattr(settings, "azure_tts_voice", "en-US-JennyNeural")
    monkeypatch.setattr(settings, "default_tts_provider", "elevenlabs")

    adapter = get_tts_provider("azure")
    assert isinstance(adapter, AzureTTSAdapter)
    assert settings.default_tts_provider == "elevenlabs"
    asyncio.run(adapter.close())

    catalogue = get_available_providers()["tts"]
    assert catalogue[0]["provider"] == "elevenlabs"
    azure = next(entry for entry in catalogue if entry["provider"] == "azure")
    assert azure["configured"] is True
    assert azure["default_voice"] == "en-US-JennyNeural"
    assert "secret-azure-key" not in json.dumps(catalogue)
