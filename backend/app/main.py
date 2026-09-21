"""Voice AI Lab - FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.api.benchmarks import router as benchmarks_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.providers import router as providers_router
from app.api.tools import router as tools_router
from app.api.voice import router as voice_router
from app.api.voice_realtime import router as voice_realtime_router
from app.api.voice_telephony import router as voice_telephony_router
from app.core.config import settings
from app.core.database import Base, engine
from app.core.logging import logger, setup_logging
from app.services.tool_service import get_tool_registry


def _migrate_benchmark_columns(eng) -> None:
    """Add Phase 5E comparison columns to benchmark_results if missing.

    SQLite does not support ALTER TABLE ADD COLUMN via SQLAlchemy ORM,
    so we use raw SQL with existence checks.
    """
    inspector = inspect(eng)
    existing_columns = {col["name"] for col in inspector.get_columns("benchmark_results")}

    columns_to_add = {
        "comparison_id": "VARCHAR(36)",
        "configuration_id": "VARCHAR(100)",
    }

    for col_name, col_type in columns_to_add.items():
        if col_name not in existing_columns:
            with eng.begin() as conn:
                sql = f"ALTER TABLE benchmark_results ADD COLUMN {col_name} {col_type}"
                conn.execute(text(sql))
            logger.info("[MIGRATION] Added column benchmark_results.%s", col_name)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown."""
    # Startup
    setup_logging()
    logger.info("Starting Voice AI Lab backend...")

    # Safe configuration diagnostics (no secrets)
    logger.info(
        "[CONFIG] Deepgram API key configured: %s",
        settings.is_provider_configured("deepgram"),
    )
    logger.info(
        "[CONFIG] OpenRouter API key configured: %s",
        settings.is_provider_configured("openrouter"),
    )
    logger.info(
        "[CONFIG] ElevenLabs API key configured: %s",
        settings.is_provider_configured("elevenlabs"),
    )
    logger.info(
        "[CONFIG] Default STT: %s / %s",
        settings.default_stt_provider,
        settings.default_stt_model,
    )
    logger.info(
        "[CONFIG] Default LLM: %s / %s",
        settings.default_llm_provider,
        settings.default_llm_model,
    )
    logger.info(
        "[CONFIG] Default TTS: %s / %s",
        settings.default_tts_provider,
        settings.default_tts_model or "(not set)",
    )
    logger.info(
        "[CONFIG] Default TTS voice configured: %s",
        bool(settings.default_tts_voice),
    )

    # Create database tables
    Base.metadata.create_all(bind=engine)

    # Phase 5E migration: add comparison columns if missing
    _migrate_benchmark_columns(engine)

    logger.info("Database tables created")

    # Initialize tool registry
    registry = get_tool_registry()
    logger.info("Tool registry initialized with %d tools", registry.count)

    yield

    # Shutdown
    logger.info("Shutting down Voice AI Lab backend...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Voice AI Lab",
        description="Provider-independent voice AI platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health_router)
    app.include_router(tools_router, prefix="/api/v1")
    app.include_router(providers_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(benchmarks_router, prefix="/api/v1")
    app.include_router(voice_router)
    app.include_router(voice_realtime_router)
    app.include_router(voice_telephony_router)

    return app


app = create_app()
