/** Telephone Observability Panel (Phase 8A).
 *
 * Connects to the backend telephony observation WebSocket and displays
 * a live timeline of semantic events during a telephone call.
 *
 * Completely independent from the Telnyx media WebSocket — closing
 * this panel has no effect on the telephone call.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  TelephonyCallState,
  TelephonyEvent,
  TelephonyTurnInfo,
} from '../types';

const WS_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(
  'http',
  'ws',
);
const OBSERVE_WS_URL = `${WS_BASE}/api/v1/voice/telephony/observe`;

const MAX_EVENTS = 100;

const STATE_LABELS: Record<TelephonyCallState, string> = {
  idle: 'IDLE',
  connected: 'CALL CONNECTED',
  listening: 'LISTENING',
  processing: 'PROCESSING',
  speaking: 'SPEAKING',
  error: 'ERROR',
  completed: 'COMPLETED',
};

const STATE_COLORS: Record<TelephonyCallState, string> = {
  idle: 'bg-gray-600',
  connected: 'bg-green-600',
  listening: 'bg-blue-600',
  processing: 'bg-yellow-600',
  speaking: 'bg-purple-600',
  error: 'bg-red-600',
  completed: 'bg-gray-600',
};

const EVENT_ICONS: Record<string, string> = {
  call_received: '\u{1F4DE}',
  call_answered: '\u2713',
  greeting_completed: '\u{1F50A}',
  media_connected: '\u{1F517}',
  stt_connected: '\u{1F399}',
  caller_speech_final: '\u{1F399}',
  caller_transcript: '\u{1F399}',
  agent_processing: '\u{1F9E0}',
  agent_response: '\u2713',
  tts_processing: '\u{1F50A}',
  tts_first_audio: '\u25B6',
  tts_completed: '\u2713',
  audio_streaming: '\u{1F4E1}',
  turn_completed: '\u2713',
  call_completed: '\u{1F4DE}',
  error: '\u26A0',
};

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString('en-GB', { hour12: false });
  } catch {
    return '';
  }
}

function deriveState(
  prev: TelephonyCallState,
  event: TelephonyEvent,
): TelephonyCallState {
  switch (event.type) {
    case 'call_received':
    case 'call_answered':
    case 'greeting_completed':
    case 'media_connected':
      return 'connected';
    case 'stt_connected':
      return 'listening';
    case 'caller_speech_final':
    case 'caller_transcript':
      return 'listening';
    case 'agent_processing':
      return 'processing';
    case 'agent_response':
      return 'processing';
    case 'tts_processing':
    case 'tts_first_audio':
    case 'tts_completed':
    case 'audio_streaming':
      return 'speaking';
    case 'turn_completed':
      return 'listening';
    case 'call_completed':
      return 'completed';
    case 'error':
      return 'error';
    default:
      return prev;
  }
}

export default function TelephoneObservability() {
  const [events, setEvents] = useState<TelephonyEvent[]>([]);
  const [callState, setCallState] = useState<TelephonyCallState>('idle');
  const [callId, setCallId] = useState<string>('');
  const [connected, setConnected] = useState(false);
  const [turns, setTurns] = useState<TelephonyTurnInfo[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const timelineRef = useRef<HTMLDivElement>(null);

  // Auto-scroll timeline
  useEffect(() => {
    if (timelineRef.current) {
      timelineRef.current.scrollTop = timelineRef.current.scrollHeight;
    }
  }, [events]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(OBSERVE_WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (e) => {
      try {
        const event: TelephonyEvent = JSON.parse(e.data);
        setEvents((prev) => {
          const next = [...prev, event];
          return next.length > MAX_EVENTS ? next.slice(-MAX_EVENTS) : next;
        });
        setCallState((prev) => deriveState(prev, event));

        // Track call ID from first event
        if (event.call_id && !callId) {
          setCallId(event.call_id);
        }

        // Accumulate turn info
        if (event.turn) {
          setTurns((prev) => {
            const existing = prev.find((t) => t.turn === event.turn);
            if (existing) {
              // Update existing turn with new metadata
              const updated = { ...existing };
              if (event.type === 'caller_transcript') {
                updated.transcript =
                  (event.metadata?.text as string) || event.message;
              }
              if (event.type === 'agent_processing') {
                updated.speech_final_to_utterance_end_ms =
                  event.metadata?.speech_final_to_utterance_end_ms as
                    | number
                    | undefined;
                updated.utterance_end_to_agent_ms =
                  event.metadata?.utterance_end_to_agent_ms as
                    | number
                    | undefined;
                updated.speech_final_to_agent_ms =
                  event.metadata?.speech_final_to_agent_ms as
                    | number
                    | undefined;
                updated.release_reason = event.metadata?.release_reason as
                  | string
                  | undefined;
                updated.settle_ms = event.metadata?.settle_ms as
                  | number
                  | undefined;
              }
              if (event.type === 'agent_response') {
                updated.streamed = event.metadata?.streamed as
                  | boolean
                  | undefined;
                updated.llm_ttft_ms = event.metadata?.llm_ttft_ms as
                  | number
                  | undefined;
                updated.llm_first_sentence_ms = event.metadata?.llm_first_sentence_ms as
                  | number
                  | undefined;
                updated.llm_total_ms = event.metadata?.llm_total_ms as
                  | number
                  | undefined;
                updated.initial_llm_total_ms = event.metadata?.initial_llm_total_ms as
                  | number
                  | undefined;
                updated.tool_execution_ms = event.metadata?.tool_execution_ms as
                  | number
                  | undefined;
                updated.tool_detection_ms = event.metadata?.tool_detection_ms as
                  | number
                  | undefined;
                updated.final_llm_total_ms = event.metadata?.final_llm_total_ms as
                  | number
                  | undefined;
              }
              if (event.type === 'tts_completed') {
                updated.tts_ttfa_ms = event.metadata?.tts_ttfa_ms as
                  | number
                  | undefined;
                updated.tts_total_ms = event.metadata?.tts_total_ms as
                  | number
                  | undefined;
                updated.audio_bytes = event.metadata?.bytes as
                  | number
                  | undefined;
                updated.audio_chunks = event.metadata?.chunks as
                  | number
                  | undefined;
              }
              if (event.type === 'turn_completed') {
                updated.speech_to_first_audio_ms = event.metadata?.speech_to_first_audio_ms as
                  | number
                  | undefined;
                updated.turn_total_ms = event.metadata?.turn_total_ms as
                  | number
                  | undefined;
                // Also capture LLM/TTS metrics if not already set
                if (updated.llm_ttft_ms == null) {
                  updated.llm_ttft_ms = event.metadata?.llm_ttft_ms as number | undefined;
                }
                if (updated.llm_first_sentence_ms == null) {
                  updated.llm_first_sentence_ms = event.metadata?.llm_first_sentence_ms as number | undefined;
                }
                if (updated.llm_total_ms == null) {
                  updated.llm_total_ms = event.metadata?.llm_total_ms as number | undefined;
                }
                if (updated.tts_ttfa_ms == null) {
                  updated.tts_ttfa_ms = event.metadata?.tts_ttfa_ms as number | undefined;
                }
                if (updated.tts_total_ms == null) {
                  updated.tts_total_ms = event.metadata?.tts_total_ms as number | undefined;
                }
                if (updated.tool_execution_ms == null) {
                  updated.tool_execution_ms = event.metadata?.tool_execution_ms as number | undefined;
                }
                if (updated.tool_detection_ms == null) {
                  updated.tool_detection_ms = event.metadata?.tool_detection_ms as number | undefined;
                }
                if (updated.final_llm_total_ms == null) {
                  updated.final_llm_total_ms = event.metadata?.final_llm_total_ms as number | undefined;
                }
                if (updated.initial_llm_total_ms == null) {
                  updated.initial_llm_total_ms = event.metadata?.initial_llm_total_ms as number | undefined;
                }
              }
              return prev.map((t) =>
                t.turn === event.turn ? updated : t,
              );
            }
            // New turn
            const newTurn: TelephonyTurnInfo = { turn: event.turn };
            if (event.type === 'caller_transcript') {
              newTurn.transcript =
                (event.metadata?.text as string) || event.message;
            }
            if (event.type === 'agent_processing') {
              newTurn.speech_final_to_utterance_end_ms =
                event.metadata?.speech_final_to_utterance_end_ms as
                  | number
                  | undefined;
              newTurn.utterance_end_to_agent_ms =
                event.metadata?.utterance_end_to_agent_ms as
                  | number
                  | undefined;
              newTurn.speech_final_to_agent_ms =
                event.metadata?.speech_final_to_agent_ms as
                  | number
                  | undefined;
              newTurn.release_reason = event.metadata?.release_reason as
                | string
                | undefined;
              newTurn.settle_ms = event.metadata?.settle_ms as
                | number
                | undefined;
            }
            if (event.type === 'agent_response') {
              newTurn.streamed = event.metadata?.streamed as
                | boolean
                | undefined;
              newTurn.llm_ttft_ms = event.metadata?.llm_ttft_ms as
                | number
                | undefined;
              newTurn.llm_first_sentence_ms = event.metadata?.llm_first_sentence_ms as
                | number
                | undefined;
              newTurn.llm_total_ms = event.metadata?.llm_total_ms as
                | number
                | undefined;
              newTurn.initial_llm_total_ms = event.metadata?.initial_llm_total_ms as
                | number
                | undefined;
              newTurn.tool_execution_ms = event.metadata?.tool_execution_ms as
                | number
                | undefined;
              newTurn.tool_detection_ms = event.metadata?.tool_detection_ms as
                | number
                | undefined;
              newTurn.final_llm_total_ms = event.metadata?.final_llm_total_ms as
                | number
                | undefined;
            }
            if (event.type === 'tts_completed') {
              newTurn.tts_ttfa_ms = event.metadata?.tts_ttfa_ms as
                | number
                | undefined;
              newTurn.tts_total_ms = event.metadata?.tts_total_ms as
                | number
                | undefined;
              newTurn.audio_bytes = event.metadata?.bytes as
                | number
                | undefined;
              newTurn.audio_chunks = event.metadata?.chunks as
                | number
                | undefined;
            }
            if (event.type === 'turn_completed') {
              newTurn.speech_to_first_audio_ms = event.metadata?.speech_to_first_audio_ms as
                | number
                | undefined;
              newTurn.turn_total_ms = event.metadata?.turn_total_ms as
                | number
                | undefined;
              if (newTurn.llm_ttft_ms == null) {
                newTurn.llm_ttft_ms = event.metadata?.llm_ttft_ms as number | undefined;
              }
              if (newTurn.llm_first_sentence_ms == null) {
                newTurn.llm_first_sentence_ms = event.metadata?.llm_first_sentence_ms as number | undefined;
              }
              if (newTurn.llm_total_ms == null) {
                newTurn.llm_total_ms = event.metadata?.llm_total_ms as number | undefined;
              }
              if (newTurn.tts_ttfa_ms == null) {
                newTurn.tts_ttfa_ms = event.metadata?.tts_ttfa_ms as number | undefined;
              }
              if (newTurn.tts_total_ms == null) {
                newTurn.tts_total_ms = event.metadata?.tts_total_ms as number | undefined;
              }
              if (newTurn.tool_execution_ms == null) {
                newTurn.tool_execution_ms = event.metadata?.tool_execution_ms as number | undefined;
              }
              if (newTurn.tool_detection_ms == null) {
                newTurn.tool_detection_ms = event.metadata?.tool_detection_ms as number | undefined;
              }
              if (newTurn.final_llm_total_ms == null) {
                newTurn.final_llm_total_ms = event.metadata?.final_llm_total_ms as number | undefined;
              }
              if (newTurn.initial_llm_total_ms == null) {
                newTurn.initial_llm_total_ms = event.metadata?.initial_llm_total_ms as number | undefined;
              }
            }
            return [...prev, newTurn];
          });
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
  }, [callId]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
  }, []);

  const reset = useCallback(() => {
    setEvents([]);
    setCallState('idle');
    setCallId('');
    setTurns([]);
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  const shortCallId = callId
    ? callId.length > 16
      ? `${callId.slice(0, 8)}...${callId.slice(-6)}`
      : callId
    : '';

  return (
    <div className="max-w-4xl mx-auto px-6 py-6 w-full flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Telephone Voice AI</h2>
          <p className="text-xs text-gray-500">
            Phase 8A — Live call observability
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={reset}
            className="px-3 py-1.5 rounded text-sm bg-gray-800 text-gray-400 hover:text-gray-200"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Connection + State */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span
              className={`inline-block w-3 h-3 rounded-full ${
                connected ? 'bg-green-500' : 'bg-gray-600'
              }`}
            />
            <span className="text-sm text-gray-400">
              {connected ? 'Observer connected' : 'Observer disconnected'}
            </span>
          </div>
          {shortCallId && (
            <span className="text-xs text-gray-600 font-mono">
              Call: {shortCallId}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          <span
            className={`inline-block px-3 py-1 rounded-full text-sm font-medium text-white ${STATE_COLORS[callState]}`}
          >
            {STATE_LABELS[callState]}
          </span>
        </div>
      </div>

      {/* Provider Info */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {[
          { label: 'Provider', value: 'Telnyx' },
          { label: 'STT', value: 'Deepgram Nova-3' },
          { label: 'LLM', value: 'GPT-4o-mini' },
          { label: 'TTS', value: 'ElevenLabs Flash v2.5' },
        ].map((p) => (
          <div
            key={p.label}
            className="bg-gray-900 border border-gray-800 rounded-lg p-3 text-center"
          >
            <div className="text-xs text-gray-500">{p.label}</div>
            <div className="text-sm text-gray-300 font-medium">{p.value}</div>
          </div>
        ))}
      </div>

      {/* Turn Cards */}
      {turns.length > 0 && (
        <div className="flex flex-col gap-2">
          <h3 className="text-sm font-medium text-gray-400">Turns</h3>
          {turns.map((t) => (
            <div
              key={t.turn}
              className="bg-gray-900 border border-gray-800 rounded-lg p-3"
            >
              <div className="text-xs text-blue-400 font-medium mb-1">
                TURN {t.turn}{t.tool_execution_ms != null ? ' \u00B7 Tool' : ''}
              </div>
              {t.transcript && (
                <div className="text-sm text-gray-300 mb-2">
                  <span className="text-gray-500">Caller: </span>
                  {t.transcript}
                </div>
              )}
              {/* Key metrics - structured display */}
              {t.tool_execution_ms != null ? (
                /* Tool turn: 6-metric layout */
                <div className="grid grid-cols-2 md:grid-cols-6 gap-2 text-xs">
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-gray-500 mb-0.5">{'\u26A1'} Initial LLM total</div>
                    <div className="text-cyan-300 font-mono">
                      {t.initial_llm_total_ms != null ? `${(t.initial_llm_total_ms / 1000).toFixed(2)}s` : (t.llm_ttft_ms != null ? `${(t.llm_ttft_ms / 1000).toFixed(2)}s` : '--')}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-gray-500 mb-0.5">{'\u{1F527}'} Tool execution</div>
                    <div className="text-orange-300 font-mono">
                      {t.tool_execution_ms != null ? `${(t.tool_execution_ms / 1000).toFixed(2)}s` : '--'}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-gray-500 mb-0.5">{'\u{1F9E0}'} Final LLM total</div>
                    <div className="text-blue-300 font-mono">
                      {t.final_llm_total_ms != null ? `${(t.final_llm_total_ms / 1000).toFixed(2)}s` : '--'}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-gray-500 mb-0.5">{'\u{1F50A}'} TTS first audio</div>
                    <div className="text-purple-300 font-mono">
                      {t.tts_ttfa_ms != null ? `${(t.tts_ttfa_ms / 1000).toFixed(2)}s` : '--'}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-gray-500 mb-0.5">{'\u{1F4DE}'} Speech {'\u2192'} audio</div>
                    <div className="text-green-300 font-mono">
                      {t.speech_to_first_audio_ms != null ? `${(t.speech_to_first_audio_ms / 1000).toFixed(2)}s` : '--'}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-gray-500 mb-0.5">{'\u23F1'} Total turn</div>
                    <div className="text-yellow-300 font-mono">
                      {t.turn_total_ms != null ? `${(t.turn_total_ms / 1000).toFixed(2)}s` : '--'}
                    </div>
                  </div>
                </div>
              ) : (
                /* Normal turn: 5-metric layout */
                <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-gray-500 mb-0.5">LLM TTFT</div>
                    <div className="text-cyan-300 font-mono">
                      {t.llm_ttft_ms != null ? `${(t.llm_ttft_ms / 1000).toFixed(2)}s` : '--'}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-gray-500 mb-0.5">First sentence</div>
                    <div className="text-cyan-200 font-mono">
                      {t.llm_first_sentence_ms != null ? `${(t.llm_first_sentence_ms / 1000).toFixed(2)}s` : '--'}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-gray-500 mb-0.5">TTS first audio</div>
                    <div className="text-purple-300 font-mono">
                      {t.tts_ttfa_ms != null ? `${(t.tts_ttfa_ms / 1000).toFixed(2)}s` : '--'}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-gray-500 mb-0.5">Speech {'\u2192'} audio</div>
                    <div className="text-green-300 font-mono">
                      {t.speech_to_first_audio_ms != null ? `${(t.speech_to_first_audio_ms / 1000).toFixed(2)}s` : '--'}
                    </div>
                  </div>
                  <div className="bg-gray-800 rounded p-2">
                    <div className="text-gray-500 mb-0.5">Total turn</div>
                    <div className="text-yellow-300 font-mono">
                      {t.turn_total_ms != null ? `${(t.turn_total_ms / 1000).toFixed(2)}s` : '--'}
                    </div>
                  </div>
                </div>
              )}
              {/* Diagnostic details */}
              <div className="flex flex-wrap gap-3 text-xs text-gray-500 mt-2">
                {t.speech_final_to_utterance_end_ms != null && (
                  <span>
                    Speech→UtteranceEnd: {(t.speech_final_to_utterance_end_ms / 1000).toFixed(2)}s
                  </span>
                )}
                {t.release_reason && (
                  <span
                    className={
                      t.release_reason === 'speech_final_settle'
                        ? 'text-green-400'
                        : 'text-gray-400'
                    }
                  >
                    {t.release_reason === 'speech_final_settle'
                      ? `Early release (${t.settle_ms}ms)`
                      : 'Normal release'}
                  </span>
                )}
                {t.streamed && (
                  <span className="text-cyan-400">Streamed</span>
                )}
                {t.tool_detection_ms != null && (
                  <span className="text-orange-400">Tool detect: {(t.tool_detection_ms / 1000).toFixed(2)}s</span>
                )}
                {t.llm_total_ms != null && (
                  <span>LLM total: {(t.llm_total_ms / 1000).toFixed(2)}s</span>
                )}
                {t.tts_total_ms != null && (
                  <span>TTS total: {(t.tts_total_ms / 1000).toFixed(2)}s</span>
                )}
                {t.audio_bytes != null && (
                  <span>Audio: {(t.audio_bytes / 1024).toFixed(0)} KB</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Event Timeline */}
      <div className="flex flex-col gap-2 flex-1 min-h-0">
        <h3 className="text-sm font-medium text-gray-400">
          Live Pipeline ({events.length} events)
        </h3>
        <div
          ref={timelineRef}
          className="bg-gray-900 border border-gray-800 rounded-lg p-3 overflow-y-auto max-h-96 flex flex-col gap-1"
        >
          {events.length === 0 && (
            <div className="text-sm text-gray-600 text-center py-8">
              Waiting for telephone call events...
            </div>
          )}
          {events.map((ev, i) => {
            const icon = EVENT_ICONS[ev.type] || '\u2022';
            const isError = ev.type === 'error';
            const time = formatTime(ev.timestamp);
            const meta = ev.metadata
              ? Object.entries(ev.metadata)
                  .filter(([k, v]) => k !== 'text' && v !== null)
                  .map(([k, v]) => {
                    // Duration metrics (milliseconds)
                    if (k.endsWith('_ms') && typeof v === 'number')
                      return `${k.replace(/_ms$/, '')}: ${(v / 1000).toFixed(2)}s`;
                    if (k === 'playback_ms' && typeof v === 'number')
                      return `${(v / 1000).toFixed(1)}s voice`;
                    if (k === 'bytes' && typeof v === 'number')
                      return `${(v / 1024).toFixed(0)} KB`;
                    if (k === 'chunks' && typeof v === 'number')
                      return `${v} frames`;
                    if (k === 'speech_final_to_utterance_end_ms' && typeof v === 'number')
                      return `speech→utterance ${(v / 1000).toFixed(2)}s`;
                    if (k === 'utterance_end_to_agent_ms' && typeof v === 'number')
                      return `utterance→agent ${(v / 1000).toFixed(2)}s`;
                    if (k === 'speech_final_to_agent_ms' && typeof v === 'number')
                      return `total ${(v / 1000).toFixed(2)}s`;
                    if (k === 'last_word_end' && typeof v === 'number')
                      return `last word ${v.toFixed(2)}s`;
                    if (k === 'model') return String(v);
                    return null;
                  })
                  .filter(Boolean)
                  .join(' \u00B7 ')
              : '';

            return (
              <div
                key={i}
                className={`flex gap-2 text-sm py-0.5 ${
                  isError ? 'text-red-400' : 'text-gray-300'
                }`}
              >
                <span className="text-gray-600 font-mono text-xs w-20 shrink-0">
                  {time}
                </span>
                <span className="w-5 text-center">{icon}</span>
                <span className="flex-1">{ev.message}</span>
                {meta && (
                  <span className="text-xs text-gray-500 shrink-0">{meta}</span>
                )}
                {ev.turn != null && (
                  <span className="text-xs text-blue-500 shrink-0">
                    T{ev.turn}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
