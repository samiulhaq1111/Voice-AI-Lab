"""Tests for the provider factory."""

import pytest

from app.providers.factory import (
    ProviderError,
    get_available_providers,
    get_llm_provider,
    get_stt_provider,
    get_tts_provider,
)
from app.providers.llm.interface import LLMInterface
from app.providers.stt.interface import STTInterface
from app.providers.tts.interface import TTSInterface


class TestGetSTTProvider:
    """Tests for STT provider factory."""

    def test_get_deepgram(self) -> None:
        adapter = get_stt_provider("deepgram", api_key="test-key")
        assert isinstance(adapter, STTInterface)
        assert adapter.provider_name == "deepgram"

    def test_get_default_stt(self) -> None:
        adapter = get_stt_provider(api_key="test-key")
        assert isinstance(adapter, STTInterface)

    def test_unsupported_stt_provider(self) -> None:
        with pytest.raises(ProviderError, match="Unsupported STT provider"):
            get_stt_provider("nonexistent", api_key="test-key")


class TestGetLLMProvider:
    """Tests for LLM provider factory."""

    def test_get_openrouter(self) -> None:
        adapter = get_llm_provider("openrouter", api_key="test-key")
        assert isinstance(adapter, LLMInterface)
        assert adapter.provider_name == "openrouter"

    def test_get_default_llm(self) -> None:
        adapter = get_llm_provider(api_key="test-key")
        assert isinstance(adapter, LLMInterface)

    def test_unsupported_llm_provider(self) -> None:
        with pytest.raises(ProviderError, match="Unsupported LLM provider"):
            get_llm_provider("nonexistent", api_key="test-key")


class TestGetTTSProvider:
    """Tests for TTS provider factory."""

    def test_get_elevenlabs(self) -> None:
        adapter = get_tts_provider("elevenlabs", api_key="test-key")
        assert isinstance(adapter, TTSInterface)
        assert adapter.provider_name == "elevenlabs"

    def test_get_default_tts(self) -> None:
        adapter = get_tts_provider(api_key="test-key")
        assert isinstance(adapter, TTSInterface)

    def test_unsupported_tts_provider(self) -> None:
        with pytest.raises(ProviderError, match="Unsupported TTS provider"):
            get_tts_provider("nonexistent", api_key="test-key")


class TestMissingAPIKeys:
    """Tests for missing API key handling."""

    def test_deepgram_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        from app.core.config import settings

        settings.deepgram_api_key = ""
        with pytest.raises(ValueError, match="API key not configured"):
            get_stt_provider("deepgram", api_key="")

    def test_openrouter_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from app.core.config import settings

        settings.openrouter_api_key = ""
        with pytest.raises(ValueError, match="API key not configured"):
            get_llm_provider("openrouter", api_key="")

    def test_elevenlabs_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        from app.core.config import settings

        settings.elevenlabs_api_key = ""
        with pytest.raises(ValueError, match="API key not configured"):
            get_tts_provider("elevenlabs", api_key="")


class TestGetAvailableProviders:
    """Tests for the available providers metadata."""

    def test_returns_all_types(self) -> None:
        result = get_available_providers()
        assert "stt" in result
        assert "llm" in result
        assert "tts" in result

    def test_stt_providers(self) -> None:
        result = get_available_providers()
        stt = result["stt"]
        assert len(stt) >= 1
        assert stt[0]["provider"] == "deepgram"
        assert "models" in stt[0]
        assert "configured" in stt[0]

    def test_no_api_keys_exposed(self) -> None:
        result = get_available_providers()
        result_str = str(result)
        assert "test-deepgram-key" not in result_str
        assert "test-openrouter-key" not in result_str
        assert "test-elevenlabs-key" not in result_str
