"""Voice AI Lab - FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.providers import router as providers_router
from app.api.tools import router as tools_router
from app.api.voice import router as voice_router
from app.core.config import settings
from app.core.database import Base, engine
from app.core.logging import logger, setup_logging
from app.services.tool_service import get_tool_registry


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
    app.include_router(voice_router)

    return app


app = create_app()
