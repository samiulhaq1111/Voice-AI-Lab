"""Shared test fixtures and configuration."""

import os

import pytest


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests run with known environment variables.

    Sets fake API keys so adapters can be instantiated without real credentials.
    Also overrides settings directly to avoid .env file interference.
    """
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-deepgram-key-xxxxx")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-xxxxx")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-elevenlabs-key-xxxxx")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./test_voice_ai_lab.db")
    os.environ.setdefault("TESTING", "true")

    # Override settings directly — .env file values take priority over env vars
    from app.core.config import settings

    settings.deepgram_api_key = "test-deepgram-key-xxxxx"
    settings.openrouter_api_key = "test-openrouter-key-xxxxx"
    settings.elevenlabs_api_key = "test-elevenlabs-key-xxxxx"


@pytest.fixture(autouse=True)
def _ensure_test_tables() -> None:
    """Create all database tables for tests that use the DB.

    Drops and recreates tables to handle schema changes between runs.
    """
    from app.core.database import Base, engine

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
