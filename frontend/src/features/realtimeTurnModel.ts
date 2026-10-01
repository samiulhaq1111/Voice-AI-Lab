/** Phase 6E conversation turn model for the realtime STT UI.
 *
 * Pure and dependency-free: the component stores no conversational state of
 * its own, so this exact module is unit-tested directly (see
 * tests/realtimeTurnModel.test.mjs).
 *
 * Turn lifecycle (server events in order):
 *   transcript_partial*  interim text — REPLACES in place (one evolving turn)
 *   transcript_final*    endpointed segments — ACCUMULATE into the current turn
 *   agent_processing     the turn was released/dispatched → commit ONE user
 *                        message built from the finalized segments (the same
 *                        text the agent received)
 *   agent_delta          progressive assistant text — appends the raw LLM
 *                        delta to the evolving assistant message
 *                        (assistantPartial)
 *   tool_progress        temporary status ack while a tool executes
 *                        (toolProgress) — cleared by agent_response
 *   agent_response       commit the assistant message (authoritative full
 *                        text replaces the accumulated draft)
 *
 * Interim partials never become conversation entries; only the dispatch
 * commits the user's turn.
 */

export interface ConversationEntry {
  role: 'user' | 'assistant';
  text: string;
  /** Tool calls made by the agent for this response (assistant entries). */
  toolCalls?: number;
  /** Agent iterations for this response (assistant entries). */
  iterations?: number;
}

export interface ConversationState {
  /** Committed conversation, oldest first. */
  entries: ConversationEntry[];
  /** Finalized segments of the in-progress user turn. */
  pendingFinals: string[];
  /** Latest interim (partial) text of the in-progress turn. */
  partial: string;
  /** True between dispatch (agent_processing) and the outcome. */
  agentProcessing: boolean;
  /** Accumulated agent_delta text of the in-flight assistant response. */
  assistantPartial: string;
  /** Temporary tool-progress status message (cleared by agent_response). */
  toolProgress: string;
}

export type ConversationAction =
  | { type: 'reset' }
  | { type: 'partial'; text: string }
  | { type: 'final'; text: string }
  | { type: 'agent_processing' }
  | { type: 'agent_delta'; text: string }
  | { type: 'tool_progress'; message: string }
  | {
      type: 'agent_response';
      text: string;
      toolCalls?: number;
      iterations?: number;
    }
  | { type: 'agent_failed' };

export function createConversationState(): ConversationState {
  return {
    entries: [],
    pendingFinals: [],
    partial: '',
    agentProcessing: false,
    assistantPartial: '',
    toolProgress: '',
  };
}

function joinSegments(segments: string[]): string {
  return segments
    .map((segment) => segment.trim())
    .filter(Boolean)
    .join(' ');
}

/** Text of the current user turn being spoken (finals + live partial). */
export function currentTurnText(state: ConversationState): string {
  return joinSegments([...state.pendingFinals, state.partial]);
}

export function conversationReducer(
  state: ConversationState,
  action: ConversationAction,
): ConversationState {
  switch (action.type) {
    case 'reset':
      return createConversationState();
    case 'partial':
      // One evolving message: the newest interim text replaces the previous.
      return { ...state, partial: action.text };
    case 'final':
      return {
        ...state,
        pendingFinals: action.text
          ? [...state.pendingFinals, action.text]
          : state.pendingFinals,
        partial: '',
      };
    case 'agent_processing': {
      // The backend released the turn: commit exactly ONE user message with
      // the finalized segments (the same text the agent receives).
      const text = joinSegments(state.pendingFinals);
      return {
        entries: text
          ? [...state.entries, { role: 'user', text }]
          : state.entries,
        pendingFinals: [],
        partial: '',
        agentProcessing: true,
        assistantPartial: '',
        toolProgress: '',
      };
    }
    case 'agent_delta':
      // Progressive assistant text: append the raw LLM delta to the
      // evolving message. agent_response later commits the authoritative
      // full text (replacing, not appending — no duplication).
      return {
        ...state,
        assistantPartial: state.assistantPartial + action.text,
      };
    case 'tool_progress':
      // Temporary status ack while a tool executes (spoken via the TTS
      // pipeline independently). Never enters the assistant accumulator;
      // agent_response clears it when the real answer arrives.
      return { ...state, toolProgress: action.message };
    case 'agent_response':
      return {
        ...state,
        entries: [
          ...state.entries,
          {
            role: 'assistant',
            text: action.text,
            toolCalls: action.toolCalls,
            iterations: action.iterations,
          },
        ],
        agentProcessing: false,
        // The event text is authoritative — drop the accumulated draft so
        // the final message is never duplicated.
        assistantPartial: '',
        toolProgress: '',
      };
    case 'agent_failed':
      // Error before a response: keep the committed history, stop the
      // in-flight indicator, discard the partial draft and status.
      return {
        ...state,
        agentProcessing: false,
        assistantPartial: '',
        toolProgress: '',
      };
  }
}
