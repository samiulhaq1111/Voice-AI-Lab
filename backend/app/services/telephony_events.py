"""Telephony semantic event bus (Phase 8A — observability).

Broadcasts human-readable telephony pipeline events to connected
frontend observers via a separate WebSocket.  The telephone call
pipeline is NEVER blocked by observer delivery — broadcast is
fire-and-forget.

Architecture:
    Pipeline component  →  await broadcast_telephony_event(...)
                                ↓  (fire-and-forget, try/except)
                           _observers: set[WebSocket]
                                ↓
                           Frontend WS (/api/v1/voice/telephony/observe)
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import WebSocket

from app.core.logging import logger

# Connected frontend observer WebSockets
_observers: set[WebSocket] = set()


def register_observer(ws: WebSocket) -> None:
    """Add a frontend WebSocket to the observer set."""
    _observers.add(ws)
    logger.debug(
        "[TELNYX:OBSERVE] observer connected total=%d", len(_observers)
    )


def unregister_observer(ws: WebSocket) -> None:
    """Remove a frontend WebSocket from the observer set."""
    _observers.discard(ws)
    logger.debug(
        "[TELNYX:OBSERVE] observer disconnected total=%d", len(_observers)
    )


async def broadcast_telephony_event(
    event_type: str,
    call_id: str,
    message: str,
    *,
    turn: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Broadcast a semantic telephony event to all connected observers.

    This is fire-and-forget: the broadcast is spawned as a background task
    so the telephone pipeline never blocks on observer delivery. A stalled
    or disconnected frontend observer cannot delay STT/agent processing.

    Args:
        event_type: Semantic event type (e.g. "agent_processing").
        call_id: Telnyx call_control_id for this call.
        message: Human-readable description.
        turn: Optional turn number.
        metadata: Optional metadata dict (durations, sizes, etc.).
    """
    if not _observers:
        return

    event: dict[str, Any] = {
        "type": event_type,
        "call_id": call_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "message": message,
    }
    if turn is not None:
        event["turn"] = turn
    if metadata:
        event["metadata"] = metadata

    payload = json.dumps(event)
    # Spawn a task so the caller returns immediately. The task handles
    # timeouts and stale observer cleanup.
    asyncio.create_task(_broadcast_payload(payload))


async def _broadcast_payload(payload: str) -> None:
    """Deliver payload to all observers with per-send timeout.

    Observers that fail to receive within 100ms are marked stale and
    removed from the set. This guarantees the telephone pipeline never
    blocks on a slow frontend.
    """
    stale: list[WebSocket] = []

    for ws in list(_observers):
        try:
            await asyncio.wait_for(ws.send_text(payload), timeout=0.1)
        except (TimeoutError, Exception):
            stale.append(ws)

    # Clean up disconnected observers
    for ws in stale:
        _observers.discard(ws)
