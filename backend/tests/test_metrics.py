"""Tests for Phase 5A voice turn metrics collection and persistence."""

import time

import pytest
from sqlalchemy.orm import Session

from app.models.benchmark_result import BenchmarkResult
from app.models.voice_session import VoiceSession
from app.services.metrics_service import (
    ERROR_STAGE_LLM,
    ERROR_STAGE_STT,
    ERROR_STAGE_TTS,
    VoiceTurnMetrics,
    new_turn_id,
    persist_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(db: Session) -> VoiceSession:
    session = VoiceSession(
        status="active",
        stt_provider="deepgram",
        stt_model="nova-3",
        llm_provider="openrouter",
        llm_model="test-model",
        tts_provider="elevenlabs",
        tts_model="eleven_flash_v2_5",
        tts_voice="JBFqnCBsd6RMkjVDRZzb",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _full_success_metrics(session_id: str) -> VoiceTurnMetrics:
    m = VoiceTurnMetrics(session_id=session_id)
    m.start_stt(
        provider="deepgram", model="nova-3", audio_bytes=57290, audio_format="audio/webm"
    )
    time.sleep(0.001)
    m.finish_stt(transcript_length=43, audio_duration_seconds=3.5)
    m.start_llm(provider="openrouter", model="test-model")
    time.sleep(0.001)
    m.finish_llm(
        usage={"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
        iterations=1,
    )
    m.start_tts(
        provider="elevenlabs",
        model="eleven_flash_v2_5",
        voice="JBFqnCBsd6RMkjVDRZzb",
        characters=53,
        output_format="mp3_44100_128",
    )
    time.sleep(0.001)
    m.finish_tts(audio_bytes=30973)
    m.complete()
    return m


# ---------------------------------------------------------------------------
# Turn ID
# ---------------------------------------------------------------------------


class TestTurnId:
    """A session may contain many turns — each needs a unique turn_id."""

    def test_new_turn_id_is_unique(self) -> None:
        ids = {new_turn_id() for _ in range(50)}
        assert len(ids) == 50

    def test_metrics_auto_generates_turn_id(self) -> None:
        m1 = VoiceTurnMetrics(session_id="abc123")
        m2 = VoiceTurnMetrics(session_id="abc123")
        assert m1.turn_id != m2.turn_id
        assert m1.session_id == m2.session_id == "abc123"


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class TestVoiceTurnTiming:
    """Latencies must be derived from a monotonic clock."""

    def test_latencies_are_none_before_stages_run(self) -> None:
        m = VoiceTurnMetrics()
        assert m.stt_latency_ms is None
        assert m.llm_latency_ms is None
        assert m.tts_latency_ms is None

    def test_stt_timing_capture(self) -> None:
        m = VoiceTurnMetrics()
        m.start_stt(
            provider="deepgram", model="nova-3", audio_bytes=1000, audio_format="audio/webm"
        )
        time.sleep(0.01)
        m.finish_stt(transcript_length=12, audio_duration_seconds=2.0)

        assert m.stt_latency_ms is not None
        assert m.stt_latency_ms >= 5
        assert m.stt_audio_bytes == 1000
        assert m.stt_audio_duration_seconds == 2.0
        assert m.transcript_length == 12

    def test_llm_timing_capture(self) -> None:
        m = VoiceTurnMetrics()
        m.start_llm(provider="openrouter", model="test-model")
        time.sleep(0.01)
        m.finish_llm(
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            iterations=2,
        )

        assert m.llm_latency_ms is not None
        assert m.llm_latency_ms >= 5
        assert m.prompt_tokens == 10
        assert m.completion_tokens == 5
        assert m.total_tokens == 15
        assert m.llm_iterations == 2

    def test_llm_usage_absent_stores_none(self) -> None:
        """Missing usage must store None, never invented values."""
        m = VoiceTurnMetrics()
        m.start_llm(provider="openrouter", model="test-model")
        m.finish_llm(usage=None, iterations=1)

        assert m.prompt_tokens is None
        assert m.completion_tokens is None
        assert m.total_tokens is None

    def test_tts_timing_capture(self) -> None:
        m = VoiceTurnMetrics()
        m.start_tts(
            provider="elevenlabs",
            model="eleven_flash_v2_5",
            voice="test-voice",
            characters=53,
            output_format="mp3_44100_128",
        )
        time.sleep(0.01)
        m.finish_tts(audio_bytes=30973)

        assert m.tts_latency_ms is not None
        assert m.tts_latency_ms >= 5
        assert m.tts_audio_bytes == 30973
        assert m.tts_characters == 53
        assert m.tts_output_format == "mp3_44100_128"

    def test_total_processing_covers_all_stages(self) -> None:
        m = _full_success_metrics("sess-1")
        assert m.total_processing_ms is not None
        assert m.total_processing_ms >= (m.stt_latency_ms or 0)
        assert m.total_processing_ms >= (m.llm_latency_ms or 0)
        assert m.total_processing_ms >= (m.tts_latency_ms or 0)

    def test_tool_timing_accumulates(self) -> None:
        m = VoiceTurnMetrics()
        m.record_tool_call(success=True, duration_ms=120.5)
        m.record_tool_call(success=True, duration_ms=80.0)
        m.record_tool_call(success=False, duration_ms=10.0)

        assert m.tool_count == 3
        assert m.tool_success_count == 2
        assert m.tool_execution_ms == pytest.approx(210.5)


# ---------------------------------------------------------------------------
# Event serialization
# ---------------------------------------------------------------------------


class TestMetricsEventSerialization:
    """The metrics event payload must be safe and JSON-serializable."""

    def test_payload_contains_expected_keys(self) -> None:
        payload = _full_success_metrics("sess-1").to_event_payload()

        for key in (
            "turn_id",
            "session_id",
            "stt_latency_ms",
            "llm_latency_ms",
            "tts_latency_ms",
            "total_processing_ms",
            "tool_count",
            "tts_audio_bytes",
            "success",
        ):
            assert key in payload, f"Missing key: {key}"

    def test_payload_is_json_serializable(self) -> None:
        import json

        payload = _full_success_metrics("sess-1").to_event_payload()
        encoded = json.dumps(payload)
        assert json.loads(encoded)["success"] is True

    def test_payload_excludes_monotonic_timestamps(self) -> None:
        """Raw perf_counter values must never leak to the client."""
        payload = _full_success_metrics("sess-1").to_event_payload()
        for key in payload:
            assert not key.endswith("_at"), f"Internal timestamp leaked: {key}"

    def test_payload_contains_no_secrets(self) -> None:
        """No API keys or credential-like fields may appear in the payload."""
        import json

        payload = _full_success_metrics("sess-1").to_event_payload()
        blob = json.dumps(payload).lower()

        for forbidden in ("api_key", "apikey", "xi-api-key", "authorization", "secret", "token="):
            assert forbidden not in blob, f"Secret-like content found: {forbidden}"
        # "sk_" / "sk-or-" prefixes used by the providers must not appear
        assert "sk_" not in blob
        assert "sk-or-" not in blob


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestMetricsPersistence:
    """Metrics are persisted via the existing BenchmarkResult model."""

    def test_successful_turn_persists_benchmark(self, db_session: Session) -> None:
        session = _make_session(db_session)
        metrics = _full_success_metrics(session.id)

        record = persist_metrics(db_session, metrics)

        assert record is not None
        assert record.turn_id == metrics.turn_id
        assert record.session_id == session.id
        assert record.conversation_success == "true"
        assert record.stt_provider == "deepgram"
        assert record.llm_provider == "openrouter"
        assert record.tts_provider == "elevenlabs"
        assert record.tts_voice == "JBFqnCBsd6RMkjVDRZzb"
        assert record.stt_latency_ms is not None
        assert record.llm_latency_ms is not None
        assert record.tts_latency_ms is not None
        assert record.total_processing_ms is not None
        assert record.prompt_tokens == 100
        assert record.completion_tokens == 25
        assert record.token_usage == 125
        assert record.tts_audio_bytes == 30973
        assert record.tts_characters == 53
        assert record.error_stage is None

    @pytest.mark.parametrize(
        "stage",
        [ERROR_STAGE_STT, ERROR_STAGE_LLM, ERROR_STAGE_TTS],
    )
    def test_failed_turn_persists_error_stage(self, db_session: Session, stage: str) -> None:
        session = _make_session(db_session)
        metrics = VoiceTurnMetrics(session_id=session.id)
        metrics.start_stt(
            provider="deepgram", model="nova-3", audio_bytes=500, audio_format="audio/webm"
        )
        metrics.fail(stage, "something went wrong")

        record = persist_metrics(db_session, metrics)

        assert record is not None
        assert record.conversation_success == "false"
        assert record.error_stage == stage
        assert record.error_message == "something went wrong"

    def test_error_message_is_truncated(self, db_session: Session) -> None:
        session = _make_session(db_session)
        metrics = VoiceTurnMetrics(session_id=session.id)
        metrics.fail(ERROR_STAGE_TTS, "x" * 2000)

        record = persist_metrics(db_session, metrics)

        assert record is not None
        assert record.error_message is not None
        assert len(record.error_message) <= 500

    def test_multiple_turns_same_session_distinct_turn_ids(self, db_session: Session) -> None:
        session = _make_session(db_session)

        r1 = persist_metrics(db_session, _full_success_metrics(session.id))
        r2 = persist_metrics(db_session, _full_success_metrics(session.id))

        assert r1 is not None and r2 is not None
        assert r1.session_id == r2.session_id == session.id
        assert r1.turn_id != r2.turn_id

        rows = (
            db_session.query(BenchmarkResult)
            .filter(BenchmarkResult.session_id == session.id)
            .all()
        )
        assert len(rows) == 2

    def test_persistence_failure_is_swallowed(self, db_session: Session) -> None:
        """A database failure must never propagate out of persist_metrics."""
        metrics = _full_success_metrics("nonexistent-session")

        # session_id violates the FK — commit will fail on some backends;
        # regardless, persist_metrics must not raise.
        result = persist_metrics(db_session, metrics)
        assert result is None or result.turn_id == metrics.turn_id
