"""Tests for application configuration."""

from app.core.config import Settings


def test_default_provider_settings() -> None:
    """Test that default provider settings are populated."""
    s = Settings(
        _env_file=None,
        deepgram_api_key="test",
        openrouter_api_key="test",
        elevenlabs_api_key="test",
    )
    assert s.default_stt_provider == "deepgram"
    assert s.default_stt_model == "nova-3"
    assert s.default_llm_provider == "openrouter"
    assert s.default_tts_provider == "elevenlabs"


def test_default_timeouts() -> None:
    """Test default timeout values."""
    s = Settings(_env_file=None)
    assert s.stt_timeout == 30.0
    assert s.llm_timeout == 60.0
    assert s.tts_timeout == 30.0


def test_is_provider_configured_with_key() -> None:
    """Test is_provider_configured returns True when key is set."""
    s = Settings(
        _env_file=None,
        deepgram_api_key="sk-test",
        openrouter_api_key="",
        elevenlabs_api_key="",
    )
    assert s.is_provider_configured("deepgram") is True
    assert s.is_provider_configured("openrouter") is False
    assert s.is_provider_configured("elevenlabs") is False


def test_is_provider_configured_unknown() -> None:
    """Test is_provider_configured returns False for unknown provider."""
    s = Settings(_env_file=None)
    assert s.is_provider_configured("nonexistent") is False


def test_cors_origin_list() -> None:
    """Test CORS origin parsing."""
    s = Settings(
        _env_file=None,
        cors_origins="http://localhost:5173,http://localhost:3000",
    )
    assert s.cors_origin_list == [
        "http://localhost:5173",
        "http://localhost:3000",
    ]


def test_cors_origin_list_empty() -> None:
    """Test CORS origin list with empty string."""
    s = Settings(_env_file=None, cors_origins="")
    assert s.cors_origin_list == []
