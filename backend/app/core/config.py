"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Application ---
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # --- Database ---
    database_url: str = "sqlite:///./voice_ai_lab.db"

    # --- CORS ---
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # --- AI Provider Keys ---
    openrouter_api_key: str = ""
    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""
    telnyx_api_key: str = ""
    telnyx_speak_voice: str = "female"
    telnyx_media_ws_url: str = ""

    # --- Default Provider Selections ---
    default_stt_provider: str = "deepgram"
    default_stt_model: str = "nova-3"
    default_llm_provider: str = "openrouter"
    default_llm_model: str = ""
    default_tts_provider: str = "elevenlabs"
    default_tts_model: str = ""
    default_tts_voice: str = ""

    # --- Provider Timeouts (seconds) ---
    stt_timeout: float = 30.0
    llm_timeout: float = 60.0
    tts_timeout: float = 30.0

    # --- Agent Runtime ---
    max_agent_iterations: int = 5
    max_conversation_messages: int = 50

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def is_provider_configured(self, provider: str) -> bool:
        """Check whether a provider has its API key set."""
        key_map = {
            "deepgram": self.deepgram_api_key,
            "openrouter": self.openrouter_api_key,
            "elevenlabs": self.elevenlabs_api_key,
            "telnyx": self.telnyx_api_key,
        }
        return bool(key_map.get(provider, ""))


settings = Settings()
