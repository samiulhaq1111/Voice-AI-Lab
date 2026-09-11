"""Chat service: orchestrates the agent loop with session/message persistence."""

import json
import time
from typing import Any

from sqlalchemy.orm import Session

from app.agents.runtime import AgentConfig, AgentResult, AgentRuntime
from app.core.config import settings
from app.core.logging import logger
from app.models.message import Message
from app.models.tool_call import ToolCall
from app.models.usage_record import UsageRecord
from app.models.voice_session import VoiceSession
from app.providers.factory import ProviderError, get_llm_provider
from app.providers.llm.interface import LLMInterface
from app.services.tool_service import get_tool_registry
from app.tools.executor import ToolExecutor


class ChatError(Exception):
    """Raised when a chat request cannot be processed."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _create_session(db: Session, llm_provider: str, llm_model: str | None) -> VoiceSession:
    """Create a new voice session."""
    session = VoiceSession(
        status="active",
        llm_provider=llm_provider,
        llm_model=llm_model or None,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.info("Created session %s (provider=%s)", session.id, llm_provider)
    return session


def _save_message(
    db: Session,
    session_id: str,
    role: str,
    content: str | None,
    sequence: int,
    token_count: int | None = None,
) -> Message:
    """Persist a message to the database."""
    msg = Message(
        session_id=session_id,
        role=role,
        content=content,
        sequence=sequence,
        token_count=token_count,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def _save_tool_call(
    db: Session,
    session_id: str,
    tool_name: str,
    tool_input: str | None,
    tool_output: str | None,
    status: str,
    error_message: str | None = None,
    duration_ms: float | None = None,
) -> ToolCall:
    """Persist a tool call record to the database."""
    tc = ToolCall(
        session_id=session_id,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        status=status,
        error_message=error_message,
        duration_ms=duration_ms,
    )
    db.add(tc)
    db.commit()
    db.refresh(tc)
    return tc


def _save_usage(
    db: Session,
    session_id: str,
    provider_name: str,
    model: str | None,
    usage: dict[str, int],
) -> UsageRecord:
    """Persist usage record to the database."""
    record = UsageRecord(
        session_id=session_id,
        provider_type="llm",
        provider_name=provider_name,
        model=model,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        metadata_json=json.dumps({"total_tokens": usage.get("total_tokens", 0)}),
    )
    db.add(record)
    db.commit()
    return record


async def process_chat(
    db: Session,
    message: str,
    session_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    llm: LLMInterface | None = None,
) -> dict[str, Any]:
    """Process a chat message through the agent loop.

    Args:
        db: Database session.
        message: User message text.
        session_id: Existing session ID, or None to create a new one.
        provider: LLM provider name override.
        model: LLM model override.
        llm: Optional pre-built LLM interface (for testing with mocks).

    Returns:
        Dict with response, session_id, tool_calls, usage, iterations.

    Raises:
        ChatError: If the request cannot be processed.
    """
    start_time = time.monotonic()
    resolved_provider = provider or settings.default_llm_provider
    resolved_model = model or settings.default_llm_model or None

    # Get or create LLM
    if llm is None:
        try:
            llm = get_llm_provider(resolved_provider, model=resolved_model)
        except ProviderError as e:
            raise ChatError(str(e), status_code=400) from e
        except ValueError as e:
            raise ChatError(str(e), status_code=400) from e

    # Get or create session
    voice_session: VoiceSession | None = None
    if session_id:
        voice_session = db.query(VoiceSession).filter(VoiceSession.id == session_id).first()
        if voice_session is None:
            raise ChatError(f"Session '{session_id}' not found", status_code=404)
    else:
        voice_session = _create_session(db, resolved_provider, resolved_model)

    assert voice_session is not None
    sid = voice_session.id

    # Build agent
    tool_registry = get_tool_registry()
    tool_executor = ToolExecutor(tool_registry)
    config = AgentConfig(
        llm_model=resolved_model,
        max_tool_rounds=settings.max_agent_iterations,
    )
    runtime = AgentRuntime(
        llm=llm,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        config=config,
    )

    # Save user message
    seq = voice_session.message_count
    _save_message(db, sid, "user", message, seq)
    voice_session.message_count = seq + 1

    # Run agent loop
    logger.info("Chat start (session=%s, provider=%s)", sid, resolved_provider)
    try:
        result: AgentResult = await runtime.run(message)
    except Exception as e:
        logger.error("Agent loop failed: %s", e)
        raise ChatError(f"LLM provider error: {e}", status_code=502) from e

    # Save assistant message
    total_tokens = result.usage.get("total_tokens")
    _save_message(db, sid, "assistant", result.response, seq + 1, token_count=total_tokens)
    voice_session.message_count = seq + 2

    # Save tool calls
    for tc_dict in result.tool_calls:
        func_data = tc_dict.get("function", {})
        tool_name = func_data.get("name", "unknown")
        tool_input = func_data.get("arguments", "{}")
        _save_tool_call(db, sid, tool_name, tool_input, None, "success")

    # Save usage
    if result.usage:
        _save_usage(db, sid, resolved_provider, resolved_model, result.usage)

    # Update session
    voice_session.status = "active"
    db.commit()

    latency_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        "Chat completion (session=%s, iterations=%d, latency=%.0fms)",
        sid,
        result.iterations,
        latency_ms,
    )

    await runtime.close()

    return {
        "response": result.response,
        "session_id": sid,
        "tool_calls": result.tool_calls,
        "usage": result.usage,
        "iterations": result.iterations,
        "latency_ms": round(latency_ms, 2),
    }
