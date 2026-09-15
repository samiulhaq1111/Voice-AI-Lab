"""Pydantic schemas for API request/response validation."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Health ---
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Session ---
class SessionCreateRequest(BaseModel):
    """Request to create a new voice session."""

    system_prompt: str = "You are a helpful voice assistant."
    stt_provider: str | None = None
    stt_model: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    tts_provider: str | None = None
    tts_model: str | None = None
    tts_voice: str | None = None


class SessionResponse(BaseModel):
    """Session information response."""

    id: str
    status: str
    system_prompt: str | None = None
    stt_provider: str | None = None
    stt_model: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    tts_provider: str | None = None
    tts_model: str | None = None
    tts_voice: str | None = None
    duration_seconds: float | None = None
    total_cost: float | None = None
    message_count: int = 0
    created_at: datetime
    ended_at: datetime | None = None


# --- Chat ---
class ChatRequest(BaseModel):
    """Request to send a text message to the agent."""

    message: str = Field(..., min_length=1, max_length=10000)
    session_id: str | None = None
    provider: str | None = None
    model: str | None = None


class ChatResponse(BaseModel):
    """Agent response to a chat message."""

    response: str
    session_id: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    iterations: int = 0
    latency_ms: float | None = None


# --- Tools ---
class ToolInfoResponse(BaseModel):
    """Information about a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    enabled: bool


class ToolListResponse(BaseModel):
    """List of all registered tools."""

    tools: list[ToolInfoResponse]
    count: int


# --- Provider ---
class ProviderInfoResponse(BaseModel):
    """Information about a configured provider."""

    provider_type: str
    provider_name: str
    display_name: str
    is_active: bool


# --- Benchmark (Phase 5B) ---
class BenchmarkScenarioResponse(BaseModel):
    """Available benchmark scenario."""

    scenario_id: str
    name: str
    description: str
    category: str
    expected_tool_calls: int
    include_tts: bool


class BenchmarkRunRequest(BaseModel):
    """Request to execute a benchmark scenario."""

    scenario_id: str = Field(..., min_length=1, max_length=100)
    repetitions: int = Field(default=1, ge=1, le=50)


class BenchmarkRunResponse(BaseModel):
    """Result of a single benchmark run."""

    run_id: str
    scenario_id: str
    success: bool
    benchmark_mode: str
    response_text: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    iterations: int = 0
    llm_latency_ms: float | None = None
    tts_latency_ms: float | None = None
    total_processing_ms: float | None = None
    tool_execution_ms: float | None = None
    tts_audio_bytes: int | None = None
    tts_characters: int | None = None
    expected_tool_calls: int = 0
    actual_tool_calls: int = 0
    tool_call_match: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    llm_provider: str | None = None
    llm_model: str | None = None
    tts_provider: str | None = None
    tts_model: str | None = None
    benchmark_result_id: str | None = None


class BenchmarkBatchResponse(BaseModel):
    """Result of running a scenario N times with aggregation."""

    runs: list[BenchmarkRunResponse]
    aggregation: dict[str, Any] | None = None


# --- Benchmark Analytics (Phase 5C) ---


class LatencyStatsResponse(BaseModel):
    """Aggregated latency statistics."""

    avg_ms: float | None = None
    median_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None


class BenchmarkOverallSummary(BaseModel):
    """Overall summary across all benchmark runs."""

    total_runs: int
    successful_runs: int
    failed_runs: int
    success_rate: float
    latency: LatencyStatsResponse
    stt_latency: LatencyStatsResponse
    llm_latency: LatencyStatsResponse
    tts_latency: LatencyStatsResponse
    tool_execution: LatencyStatsResponse
    avg_prompt_tokens: float | None = None
    avg_completion_tokens: float | None = None
    avg_total_tokens: float | None = None
    avg_tts_characters: float | None = None
    avg_tts_audio_bytes: float | None = None


class BenchmarkScenarioSummary(BaseModel):
    """Summary for a single benchmark scenario."""

    scenario_id: str
    run_count: int
    successful_runs: int
    failed_runs: int
    success_rate: float
    latency: LatencyStatsResponse
    stt_latency: LatencyStatsResponse
    llm_latency: LatencyStatsResponse
    tts_latency: LatencyStatsResponse
    tool_execution: LatencyStatsResponse
    avg_prompt_tokens: float | None = None
    avg_completion_tokens: float | None = None
    avg_total_tokens: float | None = None
    avg_tts_characters: float | None = None


class BenchmarkProviderSummary(BaseModel):
    """Summary grouped by provider + model for a given stage."""

    provider: str | None
    model: str | None
    stage: str
    run_count: int
    successful_runs: int
    success_rate: float
    latency: LatencyStatsResponse
    avg_prompt_tokens: float | None = None
    avg_completion_tokens: float | None = None
    avg_total_tokens: float | None = None
    avg_tts_characters: float | None = None
    avg_tts_audio_bytes: float | None = None


class BenchmarkRecentResult(BaseModel):
    """Lightweight representation of a recent benchmark run."""

    id: str
    run_id: str | None
    scenario_id: str | None
    benchmark_mode: str | None
    success: bool
    created_at: str
    total_processing_ms: float | None
    stt_latency_ms: float | None
    llm_latency_ms: float | None
    tts_latency_ms: float | None
    tool_execution_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    token_usage: int | None
    tts_characters: int | None
    tts_audio_bytes: int | None
    llm_provider: str | None
    llm_model: str | None
    tts_provider: str | None
    tts_model: str | None
    total_cost: float | None = None


# --- Benchmark Cost (Phase 5D) ---


class BenchmarkCostBreakdown(BaseModel):
    """Cost breakdown for a single benchmark run."""

    run_id: str | None
    scenario_id: str | None
    benchmark_mode: str | None
    stt_cost: float | None = None
    llm_input_cost: float | None = None
    llm_output_cost: float | None = None
    llm_total_cost: float | None = None
    tts_cost: float | None = None
    total_cost: float | None = None
    currency: str
    pricing_version: str
    pricing_available: bool
    stt_pricing_source: str | None = None
    llm_pricing_source: str | None = None
    tts_pricing_source: str | None = None
    stt_audio_duration_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tts_characters: int | None = None


class BenchmarkCostSummary(BaseModel):
    """Aggregate cost summary across multiple benchmark runs."""

    total_runs: int
    runs_with_cost: int
    total_cost: float | None = None
    avg_cost: float | None = None
    min_cost: float | None = None
    max_cost: float | None = None
    currency: str
    pricing_version: str
