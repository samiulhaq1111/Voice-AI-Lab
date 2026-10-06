/** Voice Chat — ChatGPT-style realtime voice conversation UI.
 *
 * Presentation only: the realtime engine (WebSocket, PCM AudioWorklet, mic
 * lifecycle, server events, ElevenLabs + Browser TTS playback, audio queues,
 * session lifecycle) lives in useRealtimeVoice.
 *
 * Debug Logs: a right-side drawer shows the realtime trace (uppercase
 * RealtimeTraceEvent lines flowing through the engine's existing
 * onDiagnostic channel — no second logging system). "Copy Logs" exports the
 * complete chronological trace as plain text; "Clear Logs" only clears the
 * drawer (the live session keeps running untouched).
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  ProviderAvailability,
  RealtimeLogEntry,
  RealtimeTraceEvent,
} from '../types';
import { getProviders } from '../services/api';
import useRealtimeVoice from './useRealtimeVoice';

/** Subtle per-category colors for the trace rows (existing palette only). */
const TRACE_EVENT_COLORS: Record<RealtimeTraceEvent, string> = {
  // Session / connection
  SESSION_START: 'text-emerald-400',
  SESSION_STOP: 'text-emerald-400',
  WS_CONNECTED: 'text-emerald-400',
  WS_DISCONNECTED: 'text-emerald-400',
  SESSION_ERROR: 'text-red-400',
  // Microphone / STT
  MIC_STARTED: 'text-sky-400',
  MIC_STOPPED: 'text-sky-400',
  STT_PARTIAL: 'text-sky-400',
  STT_FINAL: 'text-sky-400',
  SPEECH_FINAL: 'text-sky-400',
  UTTERANCE_END: 'text-sky-400',
  TURN_RELEASED: 'text-sky-400',
  // Turn / agent
  TURN_CREATED: 'text-emerald-400',
  AGENT_PROCESSING: 'text-emerald-400',
  AGENT_RESPONSE_STARTED: 'text-emerald-400',
  AGENT_RESPONSE_COMPLETED: 'text-emerald-400',
  AGENT_FAILED: 'text-red-400',
  TURN_METRICS: 'text-emerald-400',
  // LLM
  LLM_REQUEST: 'text-violet-400',
  LLM_FIRST_TOKEN: 'text-violet-400',
  LLM_FIRST_SENTENCE: 'text-violet-400',
  LLM_COMPLETED: 'text-violet-400',
  LLM_ERROR: 'text-red-400',
  // Tools
  TOOL_PROGRESS: 'text-amber-400',
  TOOL_CALL: 'text-amber-400',
  TOOL_RESULT: 'text-amber-400',
  TOOL_ERROR: 'text-red-400',
  TOOL_PROGRESS_DUPLICATE_SUPPRESSED: 'text-amber-400',
  // TTS
  TTS_START: 'text-fuchsia-400',
  TTS_SEGMENT_START: 'text-fuchsia-400',
  TTS_FIRST_AUDIO: 'text-fuchsia-400',
  TTS_SEGMENT_COMPLETED: 'text-fuchsia-400',
  TTS_COMPLETED: 'text-fuchsia-400',
  TTS_ERROR: 'text-red-400',
  // Audio / playback
  AUDIO_RECEIVED: 'text-blue-400',
  AUDIO_ENQUEUED: 'text-blue-400',
  AUDIO_PLAYING: 'text-blue-400',
  AUDIO_ENDED: 'text-blue-400',
  AUDIO_QUEUE_WAIT: 'text-blue-400',
  AUDIO_QUEUE_CLEARED: 'text-blue-400',
  AUDIO_DROPPED: 'text-red-400',
  // Barge-in
  BARGE_IN_DETECTED: 'text-red-400',
  PLAYBACK_CANCELLED: 'text-red-400',
  STALE_AUDIO_DROPPED: 'text-red-400',
  BARGE_IN_RESUMED: 'text-red-400',
};

/** Trace rows only — ordinary lowercase diagnostics are filtered out. */
const TRACE_EVENTS = new Set<string>(Object.keys(TRACE_EVENT_COLORS));

/** Flood guard: the drawer never keeps more than this many entries. */
const TRACE_MAX_ENTRIES = 2000;

/** One row in the Debug Logs drawer / copied trace. */
interface TraceEntry {
  time: string;
  type: RealtimeTraceEvent;
  detail: string;
  turn: number | null;
}

/** HH:mm:ss.SSS local wall-clock time for one trace row. */
function formatTraceTime(d: Date): string {
  const pad = (n: number, w = 2) => String(n).padStart(w, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(
    d.getMilliseconds(),
    3,
  )}`;
}

/** Display/copy line: `HH:mm:ss.SSS [T<n>] EVENT key=value …` ([T-] when the
 * entry has no turn — session-level lines). */
function traceLine(e: TraceEntry): string {
  const turn = e.turn !== null ? `[T${e.turn}]` : '[T-]';
  return `${e.time} ${turn} ${e.type}${e.detail ? ` ${e.detail}` : ''}`;
}

export default function VoiceChat() {
  const [providers, setProviders] = useState<ProviderAvailability | null>(null);
  // STT provider selection — sent per session in START (stt_provider).
  // Deepgram (default) preserves existing behavior; 'qwen' = Qwen ASR.
  const [selectedSTTProvider, setSelectedSTTProvider] = useState('deepgram');
  const [selectedLLMProvider, setSelectedLLMProvider] = useState('openrouter');
  const [llmModel, setLlmModel] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  // --- Debug Logs drawer state -------------------------------------------
  const [debugOpen, setDebugOpen] = useState(false);
  const [traceEntries, setTraceEntries] = useState<TraceEntry[]>([]);
  const [traceCopied, setTraceCopied] = useState(false);
  const traceBoxRef = useRef<HTMLDivElement>(null);
  // Auto-scroll only while the user is near the bottom of the drawer
  // (scrolling up suspends the forced scroll until they return to it).
  const traceAtBottomRef = useRef(true);
  const traceMetaRef = useRef({ started: '', llm: '' });
  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** Realtime trace sink: receives every diagnostic from the engine and
   * keeps only the uppercase trace events. */
  const handleDiagnostic = useCallback(
    (type: RealtimeLogEntry['type'], detail: string, turn?: number | null) => {
      if (!TRACE_EVENTS.has(type)) return;
      const evt = type as RealtimeTraceEvent;
      if (evt === 'SESSION_START') {
        const meta = traceMetaRef.current;
        if (!meta.started) meta.started = formatTraceTime(new Date());
        const llm = /llm=(\S+)/.exec(detail);
        meta.llm = llm ? llm[1] : '';
      }
      setTraceEntries((prev) => {
        const next = [
          ...prev,
          {
            time: formatTraceTime(new Date()),
            type: evt,
            detail,
            turn: turn ?? null,
          },
        ];
        return next.length > TRACE_MAX_ENTRIES
          ? next.slice(next.length - TRACE_MAX_ENTRIES)
          : next;
      });
    },
    [],
  );

  const {
    connState,
    micActive,
    sessionId,
    error,
    ttsProcessing,
    conversation,
    currentTurn,
    ttsMode,
    browserTtsSupported,
    start,
    stop,
    setTtsMode,
  } = useRealtimeVoice({ onDiagnostic: handleDiagnostic });

  // Load providers on mount (same catalogue as the other screens).
  useEffect(() => {
    getProviders().then(setProviders).catch(() => {});
  }, []);

  // Auto-scroll to the latest message (same pattern as TextChat).
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [
    conversation.entries.length,
    currentTurn,
    conversation.agentProcessing,
    conversation.assistantPartial,
    ttsProcessing,
  ]);

  // Drawer: keep pinning to the newest row while the user is near the
  // bottom; once they scroll up, new entries stop yanking the view.
  useEffect(() => {
    if (!debugOpen || !traceAtBottomRef.current) return;
    const el = traceBoxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [traceEntries, debugOpen]);

  // Drawer: Escape closes it.
  useEffect(() => {
    if (!debugOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setDebugOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [debugOpen]);

  // Copied-indicator timer cleanup on unmount.
  useEffect(
    () => () => {
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
    },
    [],
  );

  /** Copy the COMPLETE trace (not just the visible rows) as plain text —
   * directly pasteable into a debugging chat. */
  const handleCopyTrace = useCallback(async () => {
    const meta = traceMetaRef.current;
    const lines = [
      '=== VOICE CHAT REALTIME TRACE ===',
      `Session: ${sessionId ?? 'n/a'}`,
      `Started: ${meta.started || 'n/a'}`,
      `TTS Mode: ${ttsMode}`,
      `Model: ${meta.llm || 'default'}`,
      `Events: ${traceEntries.length}`,
      '==================================',
      ...traceEntries.map(traceLine),
    ];
    const text = lines.join('\n');
    let ok = false;
    try {
      await navigator.clipboard.writeText(text);
      ok = true;
    } catch {
      // Fallback for non-secure contexts: hidden textarea + execCommand.
      try {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        ok = document.execCommand('copy');
        ta.remove();
      } catch {
        ok = false;
      }
    }
    if (ok) {
      setTraceCopied(true);
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
      copiedTimerRef.current = setTimeout(() => setTraceCopied(false), 1500);
    }
  }, [sessionId, ttsMode, traceEntries]);

  /** Clear ONLY the drawer history — never touches the live session. */
  const handleClearTrace = useCallback(() => {
    setTraceEntries([]);
    traceAtBottomRef.current = true;
  }, []);

  const isConnected = connState === 'connected';
  const isBusy = connState !== 'disconnected';

  const handleStart = useCallback(() => {
    start({
      sttProvider: selectedSTTProvider,
      llmProvider: selectedLLMProvider,
      llmModel,
    });
  }, [start, selectedSTTProvider, selectedLLMProvider, llmModel]);

  // Split the shared catalogue into free and paid groups for the dropdown.
  const llmProv = providers?.llm.find((p) => p.provider === selectedLLMProvider);
  const paidModels = llmProv?.paid_models ?? [];
  const freeModels = (llmProv?.models ?? []).filter((m) => !paidModels.includes(m));

  const statusLabel = isConnected
    ? conversation.agentProcessing
      ? conversation.assistantPartial || conversation.toolProgress
        ? 'Responding…'
        : 'Thinking…'
      : ttsProcessing
        ? 'Speaking…'
        : micActive
          ? 'Listening… tap Stop to end'
          : 'Connected'
    : connState === 'connecting'
      ? 'Connecting…'
      : 'Tap Start and speak';

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div>
          <h2 className="text-lg font-semibold text-white">Voice Chat</h2>
          <p className="text-xs text-gray-500">Voice AI Lab</p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <button
            onClick={() => setDebugOpen(true)}
            title="Open the realtime trace drawer"
            className="px-2 py-1 rounded border border-gray-700 text-gray-300 hover:bg-gray-800"
          >
            Debug Logs{traceEntries.length > 0 ? ` (${traceEntries.length})` : ''}
          </button>
          <span
            className={`w-2 h-2 rounded-full ${
              isConnected
                ? 'bg-green-400'
                : connState === 'connecting'
                  ? 'bg-yellow-400 animate-pulse'
                  : 'bg-gray-600'
            }`}
          />
          <span
            className={
              isConnected
                ? 'text-green-400'
                : connState === 'connecting'
                  ? 'text-yellow-400'
                  : 'text-gray-500'
            }
          >
            {connState}
          </span>
        </div>
      </div>

      {/* Compact STT + LLM + TTS selection (applies to the next session start) */}
      <div className="flex flex-wrap items-center gap-2 px-4 py-2 border-b border-gray-800 bg-gray-900/50">
        <label className="text-xs text-gray-400">STT:</label>
        <select
          value={selectedSTTProvider}
          onChange={(e) => setSelectedSTTProvider(e.target.value)}
          disabled={isBusy}
          title="Streaming speech-to-text provider for the next session"
          className="text-xs bg-gray-800 text-gray-200 rounded px-2 py-1 border border-gray-700 disabled:opacity-50"
        >
          <option value="deepgram">Deepgram</option>
          <option value="qwen">Qwen ASR</option>
        </select>
        <label className="text-xs text-gray-400">LLM:</label>
        <select
          value={selectedLLMProvider}
          onChange={(e) => setSelectedLLMProvider(e.target.value)}
          disabled={isBusy}
          className="text-xs bg-gray-800 text-gray-200 rounded px-2 py-1 border border-gray-700 disabled:opacity-50"
        >
          {providers?.llm.map((p) => (
            <option key={p.provider} value={p.provider}>
              {p.provider} {p.configured ? '(configured)' : '(not configured)'}
            </option>
          )) || <option value="openrouter">openrouter</option>}
        </select>
        <select
          value={llmModel}
          onChange={(e) => setLlmModel(e.target.value)}
          disabled={isBusy}
          className="text-xs bg-gray-800 text-gray-200 rounded px-2 py-1 border border-gray-700 max-w-[240px] disabled:opacity-50"
        >
          <option value="">
            default ({llmProv?.default_model || 'nvidia/nemotron-3.5-lightning:free'})
          </option>
          {freeModels.length > 0 && (
            <optgroup label="FREE / EXPERIMENTAL">
              {freeModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </optgroup>
          )}
          {paidModels.length > 0 && (
            <optgroup label="PAID / PAYG">
              {paidModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        <span className="ml-auto flex items-center gap-1">
          <span className="text-xs text-gray-500">TTS:</span>
          <button
            onClick={() => setTtsMode('elevenlabs')}
            className={
              ttsMode === 'elevenlabs'
                ? 'px-2 py-0.5 bg-blue-600 text-white rounded text-xs font-medium'
                : 'px-2 py-0.5 bg-gray-700 text-gray-300 rounded text-xs hover:bg-gray-600'
            }
          >
            ElevenLabs
          </button>
          <button
            onClick={() => setTtsMode('browser')}
            disabled={!browserTtsSupported}
            title={
              browserTtsSupported
                ? 'Speak responses locally via speechSynthesis'
                : 'speechSynthesis is not available in this browser'
            }
            className={
              ttsMode === 'browser'
                ? 'px-2 py-0.5 bg-blue-600 text-white rounded text-xs font-medium'
                : 'px-2 py-0.5 bg-gray-700 text-gray-300 rounded text-xs hover:bg-gray-600 disabled:opacity-40 disabled:hover:bg-gray-700 disabled:cursor-not-allowed'
            }
          >
            Browser
          </button>
        </span>
      </div>

      {/* Conversation */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {conversation.entries.length === 0 && !currentTurn && (
          <div className="text-center text-gray-600 mt-12">
            Tap Start and speak — the conversation will appear here.
          </div>
        )}

        {conversation.entries.map((entry, i) => (
          <div
            key={i}
            className={`flex ${entry.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`rounded-2xl px-4 py-2 max-w-[85%] text-sm whitespace-pre-wrap break-words border ${
                entry.role === 'user'
                  ? 'bg-blue-900/40 border-blue-800 text-blue-100'
                  : 'bg-gray-800 border-gray-700 text-gray-100'
              }`}
            >
              {entry.text}
            </div>
          </div>
        ))}

        {/* Current turn — ONE evolving user bubble while the user speaks. */}
        {currentTurn && (
          <div className="flex justify-end">
            <div className="rounded-2xl px-4 py-2 max-w-[85%] text-sm whitespace-pre-wrap break-words border bg-blue-900/40 border-blue-800 text-blue-100 opacity-80 italic">
              {currentTurn}
            </div>
          </div>
        )}

        {/* Progressive assistant response — grows while the LLM streams
            (agent_delta events), finalized by agent_response. */}
        {conversation.assistantPartial && (
          <div className="flex justify-start">
            <div className="rounded-2xl px-4 py-2 max-w-[85%] text-sm whitespace-pre-wrap break-words border bg-gray-800 border-gray-700 text-gray-100 opacity-90">
              {conversation.assistantPartial}
            </div>
          </div>
        )}

        {/* Temporary tool-progress status — deterministic ack sent right
            before a tool executes; spoken via the TTS pipeline separately.
            agent_response clears it when the final answer arrives. */}
        {conversation.toolProgress && (
          <div className="flex justify-start">
            <div className="rounded-2xl px-4 py-2 max-w-[85%] text-sm whitespace-pre-wrap break-words border bg-gray-800 border-gray-700 text-gray-100 opacity-80 italic">
              {conversation.toolProgress}
            </div>
          </div>
        )}

        {/* In-flight states: Thinking only until the first delta arrives. */}
        {conversation.agentProcessing &&
          !conversation.assistantPartial &&
          !conversation.toolProgress && (
            <div className="flex justify-start">
              <div className="rounded-2xl px-4 py-2 max-w-[85%] text-sm border bg-gray-800 border-gray-700 text-gray-400">
                <span className="animate-pulse">Thinking…</span>
              </div>
            </div>
          )}
        {!conversation.agentProcessing && ttsProcessing && (
          <div className="flex justify-start">
            <div className="rounded-2xl px-4 py-2 max-w-[85%] text-sm border bg-gray-800 border-gray-700 text-gray-400">
              <span className="animate-pulse">Speaking…</span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Error */}
      {error && (
        <div className="mx-4 mb-2 bg-red-900/30 border border-red-700 rounded-lg px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Mic control */}
      <div className="px-4 py-5 border-t border-gray-800 flex flex-col items-center gap-2">
        {isConnected ? (
          <button
            onClick={stop}
            title="End the voice session"
            className="w-16 h-16 rounded-full bg-gray-600 text-white text-sm font-semibold hover:bg-gray-500"
          >
            Stop
          </button>
        ) : (
          <button
            onClick={handleStart}
            disabled={isBusy}
            title="Start the voice session"
            className="w-16 h-16 rounded-full bg-blue-600 text-white text-sm font-semibold hover:bg-blue-500 disabled:opacity-50"
          >
            Start
          </button>
        )}
        <span className="text-xs text-gray-500">{statusLabel}</span>
      </div>

      {/* Debug Logs drawer — right-side overlay, closed by default. The
          backdrop and panel are fixed; the main layout never changes. */}
      <div
        aria-hidden={!debugOpen}
        onClick={() => setDebugOpen(false)}
        className={`fixed inset-0 z-40 bg-black/50 transition-opacity duration-300 ${
          debugOpen ? 'opacity-100' : 'pointer-events-none opacity-0'
        }`}
      />
      <aside
        aria-hidden={!debugOpen}
        className={`fixed inset-y-0 right-0 z-50 flex w-[480px] max-w-[90vw] flex-col border-l border-gray-700 bg-gray-950 shadow-2xl transition-transform duration-300 ${
          debugOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        {/* Sticky header (outside the scroll area) */}
        <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-800 bg-gray-900">
          <h3 className="text-xs font-semibold tracking-wider text-gray-200">
            REALTIME DEBUG LOGS
            {traceEntries.length > 0 ? ` (${traceEntries.length})` : ''}
          </h3>
          <div className="ml-auto flex items-center gap-1.5">
            <button
              onClick={handleCopyTrace}
              title="Copy the complete trace as plain text"
              className="text-xs px-2 py-1 rounded border border-gray-700 text-gray-300 hover:bg-gray-800"
            >
              {traceCopied ? 'Copied' : 'Copy Logs'}
            </button>
            <button
              onClick={handleClearTrace}
              title="Clear the trace display (the session keeps running)"
              className="text-xs px-2 py-1 rounded border border-gray-700 text-gray-300 hover:bg-gray-800"
            >
              Clear Logs
            </button>
            <button
              onClick={() => setDebugOpen(false)}
              title="Close the drawer (Esc)"
              className="text-xs px-2 py-1 rounded border border-gray-700 text-gray-300 hover:bg-gray-800"
            >
              Close
            </button>
          </div>
        </div>

        {/* Scrollable trace body */}
        <div
          ref={traceBoxRef}
          onScroll={() => {
            const el = traceBoxRef.current;
            if (!el) return;
            traceAtBottomRef.current =
              el.scrollHeight - el.scrollTop - el.clientHeight < 48;
          }}
          className="flex-1 overflow-y-auto bg-black/40 px-3 py-2 font-mono text-[11px] leading-5"
        >
          {traceEntries.length === 0 ? (
            <div className="text-gray-600 mt-6 text-center">
              No trace entries yet — start a session.
            </div>
          ) : (
            traceEntries.map((e, i) => (
              <div key={i} className="whitespace-pre-wrap break-words">
                <span className="text-gray-500">{e.time}</span>{' '}
                <span className="text-gray-500">
                  {e.turn !== null ? `[T${e.turn}]` : '[T-]'}
                </span>{' '}
                <span className={TRACE_EVENT_COLORS[e.type]}>{e.type}</span>
                {e.detail ? <span className="text-gray-400"> {e.detail}</span> : null}
              </div>
            ))
          )}
        </div>
      </aside>
    </div>
  );
}
