/** Realtime Voice Agent — streaming voice with AgentRuntime integration (Phase 6B).
 *
 * Dedicated realtime path, fully independent of the turn-based Voice flow:
 *   getUserMedia → AudioContext(native rate) → AudioWorklet → Int16 PCM → WS binary
 *   → Deepgram Streaming STT → transcript events → AgentRuntime → agent_response
 *
 * IMPORTANT: the AudioContext is created at its NATIVE sample rate.
 * Forcing a non-native rate (e.g. 16 kHz) makes MediaStreamAudioSourceNode
 * emit silent audio in Chrome, so the actual context rate is reported to the
 * server in the START message instead (Deepgram accepts 8–48 kHz linear16).
 *
 * Protocol (matches backend app/api/voice_realtime.py):
 *   client → server: {"type":"start",...,"llm_provider":"...","llm_model":"..."}
 *                    | binary PCM | {"type":"stop"}
 *   server → client: session_started | transcript_partial | transcript_final
 *                    | utterance_end | agent_processing | agent_response
 *                    | completed | error
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  ProviderAvailability,
  RealtimeConnectionState,
  RealtimeEvent,
  RealtimeLogEntry,
  RealtimeStartMessage,
  RealtimeTimings,
} from '../types';
import { getProviders } from '../services/api';

const WS_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(
  'http',
  'ws',
);
const REALTIME_WS_URL = `${WS_BASE}/api/v1/voice/realtime/ws`;

/** AudioWorklet source: converts Float32 mic frames to Int16 PCM. */
const PCM_WORKLET_CODE = `
class PCMProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const f32 = input[0];
      const i16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        const s = Math.max(-1, Math.min(1, f32[i]));
        i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(i16.buffer, [i16.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm-processor', PCMProcessor);
`;

const MAX_LOG_ENTRIES = 50;

export default function RealtimeStt() {
  const [providers, setProviders] = useState<ProviderAvailability | null>(null);
  const [connState, setConnState] = useState<RealtimeConnectionState>('disconnected');
  const [micActive, setMicActive] = useState(false);
  const [partialText, setPartialText] = useState('');
  const [finalText, setFinalText] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [agentProcessing, setAgentProcessing] = useState(false);
  const [agentResponse, setAgentResponse] = useState<string | null>(null);
  const [agentToolCalls, setAgentToolCalls] = useState(0);
  const [agentIterations, setAgentIterations] = useState(0);
  const [ttsProcessing, setTtsProcessing] = useState(false);
  const [log, setLog] = useState<RealtimeLogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [timings, setTimings] = useState<RealtimeTimings | null>(null);

  // LLM provider/model selection
  const [selectedLLMProvider, setSelectedLLMProvider] = useState('openrouter');
  const [llmModel, setLlmModel] = useState('');

  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const silentSinkRef = useRef<GainNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const workletUrlRef = useRef<string | null>(null);
  const stopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closingRef = useRef(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Load providers on mount
  useEffect(() => {
    getProviders().then(setProviders).catch(() => {});
  }, []);

  const addLog = useCallback((type: RealtimeLogEntry['type'], detail: string) => {
    const time = new Date().toLocaleTimeString([], { hour12: false });
    setLog((prev) => [...prev.slice(-(MAX_LOG_ENTRIES - 1)), { time, type, detail }]);
  }, []);

  // Cleanup on unmount — release everything
  useEffect(() => {
    return () => {
      closingRef.current = true;
      if (stopTimeoutRef.current) clearTimeout(stopTimeoutRef.current);
      teardownAudio();
      wsRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Stop mic tracks, disconnect worklet, close AudioContext. */
  const teardownAudio = useCallback(() => {
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (silentSinkRef.current) {
      silentSinkRef.current.disconnect();
      silentSinkRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    if (workletUrlRef.current) {
      URL.revokeObjectURL(workletUrlRef.current);
      workletUrlRef.current = null;
    }
    setMicActive(false);
  }, []);

  const handleServerEvent = useCallback(
    (event: RealtimeEvent) => {
      switch (event.type) {
        case 'session_started':
          setConnState('connected');
          setSessionId(event.session_id);
          addLog(
            'session_started',
            `id=${event.session_id.slice(0, 8)}… provider=${event.provider} model=${event.model}`,
          );
          break;
        case 'transcript_partial':
          setPartialText(event.text);
          addLog('transcript_partial', `"${event.text}"`);
          break;
        case 'transcript_final':
          // Accumulate into the session transcript. Deepgram emits one final
          // per endpointed segment, so continuous speech produces several
          // finals — earlier segments must NOT be replaced by newer ones.
          setFinalText((prev) => (prev ? `${prev} ${event.text}` : event.text));
          setPartialText('');
          addLog('transcript_final', `"${event.text}"`);
          break;
        case 'utterance_end':
          addLog('utterance_end', '');
          break;
        case 'agent_processing':
          setAgentProcessing(true);
          setAgentResponse(null);
          setAgentToolCalls(0);
          setAgentIterations(0);
          addLog('agent_processing', '');
          break;
        case 'agent_response':
          setAgentProcessing(false);
          setAgentResponse(event.text);
          setAgentToolCalls(event.tool_calls);
          setAgentIterations(event.iterations);
          addLog(
            'agent_response',
            `text="${event.text.slice(0, 50)}${event.text.length > 50 ? '…' : ''}" tools=${event.tool_calls} iterations=${event.iterations}`,
          );
          break;
        case 'tts_processing':
          setTtsProcessing(true);
          addLog('tts_processing', '');
          break;
        case 'audio': {
          setTtsProcessing(false);
          // Decode base64 audio and play
          const binary = atob(event.data);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
          }
          const blob = new Blob([bytes], { type: event.format || 'audio/mpeg' });
          const url = URL.createObjectURL(blob);
          addLog('audio', `format=${event.format} size=${bytes.length}B`);

          // Create or reuse audio element for playback
          if (!audioRef.current) {
            audioRef.current = new Audio();
          }
          const audioEl = audioRef.current;
          // Clean up previous URL
          if (audioEl.src) {
            URL.revokeObjectURL(audioEl.src);
          }
          audioEl.src = url;
          audioEl.onended = () => {
            URL.revokeObjectURL(url);
            addLog('audio', 'playback ended');
          };
          audioEl.play().catch((e) => {
            console.error('[REALTIME] Audio playback failed:', e);
            addLog('error', `Audio playback blocked: ${e.message}`);
          });
          break;
        }
        case 'completed':
          setTimings(event.timings);
          addLog(
            'completed',
            `chunks=${event.timings.audio_chunks} bytes=${event.timings.audio_bytes} agent_responses=${event.timings.agent_response_count}`,
          );
          // Graceful shutdown — clear force-close timer and disconnect
          if (stopTimeoutRef.current) {
            clearTimeout(stopTimeoutRef.current);
            stopTimeoutRef.current = null;
          }
          teardownAudio();
          wsRef.current?.close();
          break;
        case 'error':
          setError(event.message);
          addLog('error', `${event.stage ? `[${event.stage}] ` : ''}${event.message}`);
          teardownAudio();
          break;
      }
    },
    [addLog, teardownAudio],
  );

  const startRealtime = useCallback(async () => {
    if (wsRef.current || connState !== 'disconnected') return; // prevent duplicates
    setError(null);
    setPartialText('');
    setFinalText('');
    setSessionId(null);
    setAgentProcessing(false);
    setAgentResponse(null);
    setAgentToolCalls(0);
    setAgentIterations(0);
    setTimings(null);
    setLog([]);
    closingRef.current = false;

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Microphone not supported in this browser');
      return;
    }
    if (!navigator.mediaDevices || typeof AudioContext === 'undefined') {
      setError('Web Audio API not supported in this browser');
      return;
    }

    try {
      // 1. Microphone permission + capture
      addLog('mic', 'requesting permission…');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      micStreamRef.current = stream;

      // 2. AudioContext at NATIVE rate — forcing e.g. 16 kHz makes
      // MediaStreamSourceNode output silence in Chrome (known issue).
      // The actual rate is reported to the server in the START message.
      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;
      addLog('mic', `granted (native ${audioCtx.sampleRate} Hz mono)`);
      addLog('mic', `audio context rate=${audioCtx.sampleRate} Hz`);

      // 3. AudioWorklet (Blob URL) — Float32 → Int16 PCM
      const blob = new Blob([PCM_WORKLET_CODE], { type: 'application/javascript' });
      const workletUrl = URL.createObjectURL(blob);
      workletUrlRef.current = workletUrl;
      await audioCtx.audioWorklet.addModule(workletUrl);
      const worklet = new AudioWorkletNode(audioCtx, 'pcm-processor');
      workletNodeRef.current = worklet;
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(worklet);

      // 4. WebSocket connect
      setConnState('connecting');
      addLog('ws', `connecting ${REALTIME_WS_URL.replace(/^wss?:\/\//, '')}…`);
      const ws = new WebSocket(REALTIME_WS_URL);
      wsRef.current = ws;
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        const startMsg: RealtimeStartMessage = {
          type: 'start',
          sample_rate: audioCtx.sampleRate,
          channels: 1,
          encoding: 'linear16',
          language: 'en',
        };
        // Add LLM configuration if selected
        if (selectedLLMProvider) {
          startMsg.llm_provider = selectedLLMProvider;
        }
        if (llmModel) {
          startMsg.llm_model = llmModel;
        }
        ws.send(JSON.stringify(startMsg));
        addLog(
          'ws',
          `START sent sample_rate=${audioCtx.sampleRate} llm=${selectedLLMProvider}/${llmModel || 'default'}`,
        );

        // 5. Worklet PCM frames → WS binary frames
        worklet.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(e.data);
          }
        };
        // Silent sink keeps the audio graph pulled without audible output
        const silentSink = audioCtx.createGain();
        silentSink.gain.value = 0;
        worklet.connect(silentSink);
        silentSink.connect(audioCtx.destination);
        silentSinkRef.current = silentSink;
        setMicActive(true);
      };

      ws.onmessage = (e) => {
        try {
          handleServerEvent(JSON.parse(e.data as string) as RealtimeEvent);
        } catch {
          // ignore non-JSON frames
        }
      };

      ws.onerror = () => {
        setError('Realtime WebSocket connection error');
        addLog('error', 'WebSocket connection error');
      };

      ws.onclose = () => {
        setConnState('disconnected');
        teardownAudio();
        if (!closingRef.current) {
          addLog('ws', 'closed');
        }
        wsRef.current = null;
      };
    } catch (err: unknown) {
      teardownAudio();
      setConnState('disconnected');
      const msg = err instanceof Error ? err.message : 'Unknown error';
      if (msg.includes('Permission') || msg.includes('denied') || msg.includes('NotAllowed')) {
        addLog('mic', 'permission denied');
        setError('Microphone permission denied. Please allow microphone access.');
      } else if (msg.includes('audioWorklet') || msg.includes('AudioWorklet')) {
        setError('AudioWorklet is not supported in this browser.');
      } else {
        setError(`Realtime start failed: ${msg}`);
      }
    }
  }, [addLog, connState, handleServerEvent, teardownAudio]);

  const stopRealtime = useCallback(() => {
    closingRef.current = true;
    addLog('ws', 'STOP sent');

    // Detach audio graph first so no more PCM frames are produced
    teardownAudio();

    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop' }));
      // Fallback: force-close if server never sends completed.
      // Timeout is generous because AgentRuntime with tool calls can take
      // 30-60+ seconds. The backend waits for pending processing before
      // sending completed, so this is a safety net for genuine hangs.
      stopTimeoutRef.current = setTimeout(() => {
        if (wsRef.current) {
          addLog('ws', 'force closed (no completed event)');
          ws.close();
        }
      }, 90000);
    } else {
      setConnState('disconnected');
    }
  }, [addLog, teardownAudio]);

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
          {isConnected ? (
            <button
              onClick={stopRealtime}
              className="px-4 py-1.5 bg-gray-600 text-white rounded text-sm font-medium hover:bg-gray-500"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={startRealtime}
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
          <div className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 min-h-[3rem]">
            <div className="text-xs text-gray-500 mb-1 font-medium uppercase">
              Live transcript
            </div>
            <div className="text-sm text-yellow-200 break-words min-h-[1.25rem]">
              {partialText || <span className="text-gray-600">…</span>}
            </div>
          </div>
          <div className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2">
            <div className="text-xs text-gray-500 mb-1 font-medium uppercase">
              User utterance
            </div>
            <div className="text-sm text-green-200 break-words min-h-[1.25rem]">
              {finalText || <span className="text-gray-600">—</span>}
            </div>
          </div>
          {/* Agent response */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2">
            <div className="text-xs text-gray-500 mb-1 font-medium uppercase">
              Assistant response
              {agentProcessing && (
                <span className="ml-2 text-yellow-400 animate-pulse">processing…</span>
              )}
            </div>
            <div className="text-sm text-blue-200 break-words min-h-[1.25rem]">
              {agentResponse || (agentProcessing ? '' : <span className="text-gray-600">—</span>)}
            </div>
            {agentResponse && (agentToolCalls > 0 || agentIterations > 0) && (
              <div className="text-xs text-gray-500 mt-1">
                {agentToolCalls > 0 && <span>tools: {agentToolCalls} </span>}
                {agentIterations > 0 && <span>iterations: {agentIterations}</span>}
              </div>
            )}
            {ttsProcessing && (
              <div className="text-xs text-green-400 mt-1 animate-pulse">
                Speaking…
              </div>
            )}
          </div>
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
