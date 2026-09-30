/** Realtime Voice Agent — developer/debug UI (Phase 6B).
 *
 * The realtime engine (WebSocket, PCM AudioWorklet, mic lifecycle, server
 * event handling, epoch/session guards, STOP handling, ElevenLabs + Browser
 * TTS playback, audio queues, browser_timing reporting, session state,
 * conversation state, turn metrics, latency state) lives in
 * useRealtimeVoice; this screen renders the developer panels on top of it:
 * Events log, raw PCM diagnostics, session timings, turn latency, session ID
 * and the ElevenLabs/Browser TTS controls.
 */

import { useCallback, useEffect, useState } from 'react';
import type { ProviderAvailability, RealtimeLogEntry } from '../types';
import { getProviders } from '../services/api';
import useRealtimeVoice from './useRealtimeVoice';

const MAX_LOG_ENTRIES = 50;

export default function RealtimeStt() {
  const [providers, setProviders] = useState<ProviderAvailability | null>(null);
  const [log, setLog] = useState<RealtimeLogEntry[]>([]);

  // LLM provider/model selection
  const [selectedLLMProvider, setSelectedLLMProvider] = useState('openrouter');
  const [llmModel, setLlmModel] = useState('');

  // Engine diagnostics → Events panel (the hook forwards its lines verbatim).
  const handleDiagnostic = useCallback(
    (type: RealtimeLogEntry['type'], detail: string) => {
      const time = new Date().toLocaleTimeString([], { hour12: false });
      setLog((prev) => [...prev.slice(-(MAX_LOG_ENTRIES - 1)), { time, type, detail }]);
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
    timings,
    turnMetrics,
    browserPlaybackLatencyMs,
    ttsMode,
    browserTtsSupported,
    start,
    stop,
    setTtsMode,
  } = useRealtimeVoice({ onDiagnostic: handleDiagnostic });

  // Load providers on mount
  useEffect(() => {
    getProviders().then(setProviders).catch(() => {});
  }, []);

  const handleStart = useCallback(() => {
    setLog([]);
    start({ llmProvider: selectedLLMProvider, llmModel });
  }, [start, selectedLLMProvider, llmModel]);

  const isConnected = connState === 'connected';
  const isBusy = connState !== 'disconnected';

  return (
    <div className="border-t border-gray-800">
      {/* Section header — clearly separate from turn-based voice */}
      <div className="flex items-center justify-between px-4 py-3">
        <h3 className="text-sm font-semibold text-white uppercase tracking-wide">
          Realtime Voice Agent
        </h3>
        <div className="flex items-center gap-3 text-xs">
          <span>
            Connection:{' '}
            <span
              className={
                connState === 'connected'
                  ? 'text-green-400'
                  : connState === 'connecting'
                    ? 'text-yellow-400'
                    : 'text-gray-500'
              }
            >
              {connState}
            </span>
          </span>
          <span>
            Mic:{' '}
            <span className={micActive ? 'text-red-400' : 'text-gray-500'}>
              {micActive ? 'live' : 'off'}
            </span>
          </span>
          {/* Phase 6L: response TTS mode — ElevenLabs (server MP3) vs Browser
              (Web Speech API). Compact switch; doubles as the "TTS: …"
              diagnostic label for manual benchmarking. */}
          <span className="flex items-center gap-1">
            <span className="text-gray-500">TTS:</span>
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
              Browser{!browserTtsSupported ? ' (unavailable)' : ''}
            </button>
          </span>
          {isConnected ? (
            <button
              onClick={stop}
              className="px-4 py-1.5 bg-gray-600 text-white rounded text-sm font-medium hover:bg-gray-500"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={isBusy}
              className="px-4 py-1.5 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-500 disabled:opacity-50"
            >
              Start
            </button>
          )}
        </div>
      </div>

      {/* LLM Configuration */}
      {!isConnected && (
        <div className="px-4 pb-3">
          <div className="flex flex-wrap gap-3 items-end">
            <div className="flex-1 min-w-[150px]">
              <label className="block text-xs text-gray-500 mb-1 font-medium uppercase">
                LLM Provider
              </label>
              <select
                value={selectedLLMProvider}
                onChange={(e) => setSelectedLLMProvider(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-white"
              >
                {providers?.llm.map((p) => (
                  <option key={p.provider} value={p.provider} disabled={!p.configured}>
                    {p.provider} {!p.configured ? '(not configured)' : ''}
                  </option>
                )) || <option value="openrouter">openrouter</option>}
              </select>
            </div>
            <div className="flex-1 min-w-[200px]">
              <label className="block text-xs text-gray-500 mb-1 font-medium uppercase">
                LLM Model (optional)
              </label>
              <select
                value={llmModel}
                onChange={(e) => setLlmModel(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-white"
              >
                <option value="">Default</option>
                {providers?.llm
                  .find((p) => p.provider === selectedLLMProvider)
                  ?.models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
              </select>
            </div>
          </div>
        </div>
      )}

      {/* Session ID */}
      {sessionId && (
        <div className="px-4 pb-2">
          <div className="text-xs text-gray-500">
            Session: <span className="text-gray-300 font-mono">{sessionId.slice(0, 8)}…</span>
          </div>
        </div>
      )}

      <div className="px-4 pb-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Left: transcripts + agent response */}
        <div className="space-y-2 min-w-0">
          {/* Conversation — ChatGPT-style: the current user turn evolves in
              place while speaking and is committed ONCE on dispatch. Every
              raw event stays visible in the Events panel on the right. */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2">
            <div className="text-xs text-gray-500 mb-1 font-medium uppercase">
              Conversation
              {conversation.agentProcessing && (
                <span className="ml-2 text-yellow-400 animate-pulse">thinking…</span>
              )}
              {!conversation.agentProcessing && ttsProcessing && (
                <span className="ml-2 text-green-400 animate-pulse">speaking…</span>
              )}
            </div>
            <div className="space-y-2 max-h-80 overflow-y-auto">
              {conversation.entries.length === 0 &&
                !currentTurn && (
                  <div className="text-sm text-gray-600">—</div>
                )}
              {conversation.entries.map((entry, i) => (
                <div key={i} className="text-sm break-words">
                  <span
                    className={
                      entry.role === 'user'
                        ? 'text-green-400 font-semibold'
                        : 'text-blue-400 font-semibold'
                    }
                  >
                    {entry.role === 'user' ? 'USER' : 'ASSISTANT'}:
                  </span>{' '}
                  <span
                    className={
                      entry.role === 'user' ? 'text-green-100' : 'text-blue-100'
                    }
                  >
                    {entry.text}
                  </span>
                  {entry.role === 'assistant' &&
                    ((entry.toolCalls ?? 0) > 0 ||
                      (entry.iterations ?? 0) > 0) && (
                      <span className="ml-2 text-xs text-gray-500">
                        {(entry.toolCalls ?? 0) > 0 && (
                          <span>tools: {entry.toolCalls} </span>
                        )}
                        {(entry.iterations ?? 0) > 0 && (
                          <span>iterations: {entry.iterations}</span>
                        )}
                      </span>
                    )}
                </div>
              ))}
            </div>
          </div>
          {/* Current turn — ONE evolving message while the user speaks. */}
          {currentTurn && (
            <div className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2">
              <div className="text-xs text-gray-500 mb-1 font-medium uppercase">
                Current turn
              </div>
              <div className="text-sm text-yellow-200 break-words min-h-[1.25rem]">
                {currentTurn}
              </div>
            </div>
          )}
          {timings && (
            <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs font-mono space-y-1">
              <div className="text-gray-500 uppercase font-sans font-medium">
                Session timings (server)
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">first partial</span>
                <span className="text-gray-200">
                  {timings.first_partial_ms != null
                    ? `${Math.round(timings.first_partial_ms)} ms`
                    : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">first final</span>
                <span className="text-gray-200">
                  {timings.first_final_ms != null
                    ? `${Math.round(timings.first_final_ms)} ms`
                    : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">session duration</span>
                <span className="text-blue-300">
                  {timings.session_duration_ms != null
                    ? `${Math.round(timings.session_duration_ms)} ms`
                    : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">audio</span>
                <span className="text-gray-200">
                  {timings.audio_chunks} chunks / {(timings.audio_bytes / 1024).toFixed(1)} KB
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">agent responses</span>
                <span className="text-green-300">{timings.agent_response_count}</span>
              </div>
            </div>
          )}
          {/* Per-turn latency metrics — server durations clearly separated from browser */}
          {turnMetrics && (
            <div className="bg-gray-900 border border-indigo-800 rounded-lg px-3 py-2 text-xs font-mono space-y-1">
              <div className="text-indigo-400 uppercase font-sans font-medium">
                Turn {turnMetrics.turn} latency (server)
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">utterance_end → agent</span>
                <span className="text-gray-200">
                  {turnMetrics.utterance_end_to_agent_ms != null
                    ? `${Math.round(turnMetrics.utterance_end_to_agent_ms)} ms`
                    : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">agent processing</span>
                <span className="text-gray-200">
                  {turnMetrics.agent_processing_ms != null
                    ? `${Math.round(turnMetrics.agent_processing_ms)} ms`
                    : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">TTS duration</span>
                <span className="text-gray-200">
                  {turnMetrics.tts_duration_ms != null
                    ? `${Math.round(turnMetrics.tts_duration_ms)} ms`
                    : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between border-t border-gray-800 pt-1">
                <span className="text-indigo-400">utterance_end → audio sent</span>
                <span className="text-indigo-200 font-semibold">
                  {turnMetrics.utterance_end_to_audio_sent_ms != null
                    ? `${Math.round(turnMetrics.utterance_end_to_audio_sent_ms)} ms`
                    : 'N/A'}
                </span>
              </div>
              {browserPlaybackLatencyMs != null && (
                <div className="flex justify-between border-t border-gray-800 pt-1">
                  <span className="text-amber-500">first audio playback (browser)</span>
                  <span className="text-amber-300">
                    {browserPlaybackLatencyMs} ms
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: event log */}
        <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 min-w-0">
          <div className="text-xs text-gray-500 mb-1 font-medium uppercase">Events</div>
          <div className="text-xs font-mono max-h-56 overflow-y-auto space-y-0.5">
            {log.length === 0 ? (
              <div className="text-gray-600">No events yet</div>
            ) : (
              log.map((entry, i) => (
                <div key={i} className="flex gap-2">
                  <span className="text-gray-600 shrink-0">[{entry.time}]</span>
                  <span
                    className={
                      entry.type === 'error'
                        ? 'text-red-400'
                        : entry.type === 'turn_metrics'
                          ? 'text-indigo-300'
                          : entry.type === 'agent_response'
                            ? 'text-blue-300'
                            : entry.type === 'agent_processing'
                              ? 'text-yellow-400'
                              : entry.type === 'transcript_final'
                                ? 'text-green-300'
                                : entry.type === 'transcript_partial'
                                  ? 'text-yellow-300'
                                  : 'text-gray-400'
                    }
                  >
                    {entry.type}
                    {entry.detail ? ` ${entry.detail}` : ''}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="mx-4 mb-3 bg-red-900/30 border border-red-700 rounded-lg px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}
    </div>
  );
}
