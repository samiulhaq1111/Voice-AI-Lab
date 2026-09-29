/** Phase 6H: sequential playback queue for sentence audio segments.
 *
 * The backend delivers one ordered `audio` segment per completed LLM
 * sentence (segment 1, 2, 3 …) for a single turn. The browser must play
 * them strictly in sentence order — segment N+1 never starts before
 * segment N ends. These state transitions are kept pure and DOM-free so
 * they can be unit-tested; the component owns the <audio> element, the
 * object URLs and the WebSocket playback report.
 */

export interface AudioSegment {
  /** Object URL of the decoded audio blob. */
  url: string;
  /** Turn this segment belongs to (null when the server did not label it). */
  turn: number | null;
  /** Date.now() − sent_epoch_ms at WS receipt (null when not measurable). */
  wsTransitMs: number | null;
  /** performance.now() when the segment message was received. */
  receivedAt: number;
}

export interface AudioQueueState {
  /** Segments waiting to play, in arrival (sentence) order. */
  pending: AudioSegment[];
  /** The segment currently attached to the <audio> element. */
  current: AudioSegment | null;
  /** Turns whose first-playback report was already sent. */
  reportedTurns: number[];
}

export function createAudioQueueState(): AudioQueueState {
  return { pending: [], current: null, reportedTurns: [] };
}

/** Append one received segment — arrival order is sentence order. */
export function enqueueSegment(
  state: AudioQueueState,
  segment: AudioSegment,
): AudioQueueState {
  return { ...state, pending: [...state.pending, segment] };
}

/**
 * Promote the next pending segment to `current` when the element is idle.
 * Returns the segment the caller must play, or null when a segment is
 * already playing or nothing is pending (queue order is preserved either
 * way — the caller re-invokes this after each segment ends).
 */
export function beginNextSegment(state: AudioQueueState): {
  state: AudioQueueState;
  next: AudioSegment | null;
} {
  if (state.current !== null || state.pending.length === 0) {
    return { state, next: null };
  }
  const [next, ...rest] = state.pending;
  return { state: { ...state, pending: rest, current: next }, next };
}

/**
 * Mark the current segment finished (ended or failed). Returns the object
 * URL the caller must revoke, if any.
 */
export function endCurrentSegment(state: AudioQueueState): {
  state: AudioQueueState;
  revokeUrl: string | null;
} {
  if (state.current === null) {
    return { state, revokeUrl: null };
  }
  return {
    state: { ...state, current: null },
    revokeUrl: state.current.url,
  };
}

/**
 * Claim the once-per-turn first-playback report. `first` is true exactly
 * once per turn — for the first segment of that turn that actually starts
 * playing; later sentences of the same turn do not re-report.
 */
export function markTurnReported(
  state: AudioQueueState,
  turn: number,
): { state: AudioQueueState; first: boolean } {
  if (state.reportedTurns.includes(turn)) {
    return { state, first: false };
  }
  return {
    state: { ...state, reportedTurns: [...state.reportedTurns, turn] },
    first: true,
  };
}

/**
 * Reset the queue (session teardown / STOP / new session) and return every
 * object URL still owned by the queue so the caller can revoke them.
 */
export function clearAudioQueue(state: AudioQueueState): {
  state: AudioQueueState;
  revokeUrls: string[];
} {
  const revokeUrls = state.pending.map((s) => s.url);
  if (state.current !== null) {
    revokeUrls.unshift(state.current.url);
  }
  return { state: createAudioQueueState(), revokeUrls };
}
