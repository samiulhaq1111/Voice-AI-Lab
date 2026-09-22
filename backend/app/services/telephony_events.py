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

    This is fire-and-forget: if no observers are connected or a send
    fails, the error is silently logged and the telephone pipeline
    continues unaffected.

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
    stale: list[WebSocket] = []

    for ws in list(_observers):
        try:
            await ws.send_text(payload)
        except Exception:
            stale.append(ws)

    # Clean up disconnected observers
    for ws in stale:
        _observers.discard(ws)
