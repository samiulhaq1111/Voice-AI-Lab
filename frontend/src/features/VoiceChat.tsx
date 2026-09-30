/** Voice Chat — ChatGPT-style realtime voice conversation UI.
 *
 * Presentation only: the realtime engine (WebSocket, PCM AudioWorklet, mic
 * lifecycle, server events, ElevenLabs + Browser TTS playback, audio queues,
 * session lifecycle) lives in useRealtimeVoice. Diagnostics (Events log, raw
 * PCM lines, session timings, turn latency, session ID) intentionally stay
 * in the Voice (Dev) screen.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { ProviderAvailability } from '../types';
import { getProviders } from '../services/api';
import useRealtimeVoice from './useRealtimeVoice';

export default function VoiceChat() {
  const [providers, setProviders] = useState<ProviderAvailability | null>(null);
  const [selectedLLMProvider, setSelectedLLMProvider] = useState('openrouter');
  const [llmModel, setLlmModel] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  const {
    connState,
    micActive,
    error,
    ttsProcessing,
    conversation,
    currentTurn,
    ttsMode,
    browserTtsSupported,
    start,
    stop,
    setTtsMode,
  } = useRealtimeVoice();

  // Load providers on mount (same catalogue as the other screens).
  useEffect(() => {
    getProviders().then(setProviders).catch(() => {});
  }, []);

  // Auto-scroll to the latest message (same pattern as TextChat).
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation.entries.length, currentTurn, conversation.agentProcessing, ttsProcessing]);

  const isConnected = connState === 'connected';
  const isBusy = connState !== 'disconnected';

  const handleStart = useCallback(() => {
    start({ llmProvider: selectedLLMProvider, llmModel });
  }, [start, selectedLLMProvider, llmModel]);

  // Split the shared catalogue into free and paid groups for the dropdown.
  const llmProv = providers?.llm.find((p) => p.provider === selectedLLMProvider);
  const paidModels = llmProv?.paid_models ?? [];
  const freeModels = (llmProv?.models ?? []).filter((m) => !paidModels.includes(m));

  const statusLabel = isConnected
    ? conversation.agentProcessing
      ? 'Thinking…'
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

      {/* Compact LLM + TTS selection (applies to the next session start) */}
      <div className="flex flex-wrap items-center gap-2 px-4 py-2 border-b border-gray-800 bg-gray-900/50">
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

        {/* Subtle in-flight states while the agent works / TTS plays. */}
        {conversation.agentProcessing && (
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
    </div>
  );
}
