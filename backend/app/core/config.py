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
    # Microsoft Azure AI Speech TTS (optional provider).
    azure_tts_key: str = ""
    azure_tts_region: str = ""
    azure_tts_voice: str = "en-US-JennyNeural"
    telnyx_api_key: str = ""
    telnyx_speak_voice: str = "female"
    telnyx_media_ws_url: str = ""
    qwen_ws_url: str = ""
    qwen_ws_api_key: str = ""
    qwen_asr_ws_url: str = ""
    qwen_asr_api_key: str = ""
    # Microsoft MAI-Transcribe-2-Streaming (Azure AI) realtime STT.
    # Resource root, e.g. "https://<resource>.services.ai.azure.com".
    mai_stt_api_key: str = ""
    mai_stt_resource_url: str = ""
    mai_stt_deployment: str = "MAI-Transcribe-2-Streaming"

    # --- Default Provider Selections ---
    # Realtime streaming STT selection. Empty = use default_stt_provider.
    # Supported: "deepgram" (default) | "qwen" (Qwen ASR Kaggle WebSocket)
    # | "mai" (Microsoft MAI-Transcribe-2-Streaming Azure realtime API).
    stt_provider: str = ""
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

    # --- MCP Tools (optional; disabled by default) ---
    # When enabled, the backend connects at startup to ONE local stdio MCP
    # server and registers ONLY the explicitly allowlisted tools as native
    # tools (named mcp_<label>_<tool>) through the existing tool registry.
    # Startup is fail-soft: an unavailable server never blocks the app.
    mcp_enabled: bool = False
    mcp_server_label: str = "local"
    mcp_server_command: str = ""
    # Comma-separated stdio server arguments (empty = none).
    mcp_server_args: str = ""
    # Comma-separated MCP tool names to bridge. Explicit allowlist only;
    # tools are never selected implicitly from the server's listing.
    mcp_allowed_tools: str = ""
    mcp_tool_timeout: float = 10.0
    mcp_connect_timeout: float = 10.0
    # Shared secret for the runtime enable/disable control endpoints.
    # Empty (the default) restricts control to loopback clients; when set,
    # non-local callers must send it as the X-MCP-Control-Token header.
    mcp_control_token: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def mcp_server_arg_list(self) -> list[str]:
        """Parse comma-separated MCP stdio server arguments into a list."""
        return [arg.strip() for arg in self.mcp_server_args.split(",") if arg.strip()]

    @property
    def mcp_allowed_tool_list(self) -> list[str]:
        """Parse comma-separated allowlisted MCP tool names into a list."""
        return [name.strip() for name in self.mcp_allowed_tools.split(",") if name.strip()]

    def is_provider_configured(self, provider: str) -> bool:
        """Check whether a provider has its API key set."""
        if provider == "azure":
            # Azure needs both a key and a region to reach the resource.
            return bool(self.azure_tts_key and self.azure_tts_region)
        key_map = {
            "deepgram": self.deepgram_api_key,
            "openrouter": self.openrouter_api_key,
            "elevenlabs": self.elevenlabs_api_key,
            "telnyx": self.telnyx_api_key,
        }
        return bool(key_map.get(provider, ""))


settings = Settings()
