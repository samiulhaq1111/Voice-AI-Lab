/** Voice Test — turn-based voice pipeline dev UI. */

import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  BrowserTurnTimings,
  ConnectionState,
  ProviderAvailability,
  RecordingState,
  VoiceEvent,
  VoiceToolCall,
  VoiceTurnMetrics,
} from '../types';
import { getProviders } from '../services/api';

const WS_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(
  'http',
  'ws',
);

interface ToolCallEntry {
  name: string;
  args: Record<string, unknown>;
}

export default function VoiceTest() {
  const [providers, setProviders] = useState<ProviderAvailability | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [recordingState, setRecordingState] = useState<RecordingState>('idle');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [transcript, setTranscript] = useState('');
  const [assistantText, setAssistantText] = useState('');
  const [toolCalls, setToolCalls] = useState<ToolCallEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);

  // Provider/model selection (same pattern as chat screen)
  const [selectedLLMProvider, setSelectedLLMProvider] = useState('openrouter');
  const [llmModel, setLlmModel] = useState('');
  const [ttsModel, setTtsModel] = useState('');

  // Phase 5A benchmark metrics
  const [serverMetrics, setServerMetrics] = useState<VoiceTurnMetrics | null>(null);
  const [browserTimings, setBrowserTimings] = useState<BrowserTurnTimings | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const recordStartRef = useRef<number>(0);
  const processingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Browser timing marks (performance.now())
  const turnStartRef = useRef<number>(0);
  const recordingStartRef = useRef<number>(0);
  const recordingStopRef = useRef<number>(0);
  const audioSentRef = useRef<number>(0);
  const audioReceivedRef = useRef<number>(0);

  // Load providers on mount
  useEffect(() => {
    getProviders().then(setProviders).catch(() => {});
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsRef.current?.close();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      if (timerRef.current) clearInterval(timerRef.current);
      if (processingTimeoutRef.current) clearTimeout(processingTimeoutRef.current);
    };
  }, []);

  const getWsUrl = useCallback(() => {
    return `${WS_BASE}/api/v1/voice/ws`;
  }, []);

  const handleEvent = useCallback((event: VoiceEvent) => {
    console.log(`[VOICE:UI] Server event type=${event.type}`);
    switch (event.type) {
      case 'session_started':
        setSessionId(event.session_id);
        setConnectionState('connected');
        break;
      case 'processing':
        setRecordingState('processing');
        setAssistantText('');
        setToolCalls([]);
        console.log('[VOICE:UI] Waiting for agent response');
        // Start processing timeout (90s — based on provider timeout expectations)
        if (processingTimeoutRef.current) clearTimeout(processingTimeoutRef.current);
        processingTimeoutRef.current = setTimeout(() => {
          console.error('[VOICE:UI] Voice turn timeout processing_ms=90000');
          setError('Voice processing timed out. Please try again.');
          setRecordingState('idle');
        }, 90000);
        break;
      case 'transcript':
        if (event.final) {
          setTranscript(event.text);
          console.log(`[VOICE:UI] Transcript receive length=${event.text.length}`);
        }
        break;
      case 'agent_response':
        setAssistantText(event.text);
        console.log(`[VOICE:UI] Agent response received length=${event.text.length}`);
        console.log('[VOICE:UI] Waiting for TTS');
        break;
      case 'tool_call': {
        const tc = event as VoiceToolCall;
        setToolCalls((prev) => [...prev, { name: tc.name, args: tc.arguments }]);
        console.log(`[VOICE:UI] Tool call received name=${tc.name}`);
        break;
      }
      case 'audio': {
        // Decode base64 audio and play
        audioReceivedRef.current = performance.now();
        const binary = atob(event.data);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: event.format || 'audio/mpeg' });
        const url = URL.createObjectURL(blob);
        setRecordingState('playing');
        console.log(
          `[VOICE:UI] Audio received format=${event.format} size_bytes=${bytes.length} ` +
            `audio_receive_ms=${Math.round(audioReceivedRef.current - audioSentRef.current)}`,
        );
        if (audioRef.current) {
          audioRef.current.src = url;
          audioRef.current.onended = () => {
            console.log('[VOICE:UI] Audio playback completed');
            setRecordingState('idle');
            URL.revokeObjectURL(url);
          };
          const playbackStart = performance.now();
          console.log(
            `[VOICE:UI] Audio playback started playback_start_latency_ms=${Math.round(playbackStart - audioReceivedRef.current)}`,
          );
          audioRef.current.play().catch((e) => {
            console.error(`[VOICE:UI] Voice error message=${e.message}`);
            setError(`Audio playback failed: ${e.message}`);
            setRecordingState('idle');
          });
          // Record browser-side timings for this turn
          setBrowserTimings({
            recording_duration_ms: Math.round(
              recordingStopRef.current - recordingStartRef.current,
            ),
            audio_upload_ms: Math.round(audioSentRef.current - recordingStopRef.current),
            server_processing_ms: Math.round(audioReceivedRef.current - audioSentRef.current),
            audio_receive_ms: Math.round(audioReceivedRef.current - audioSentRef.current),
            playback_start_latency_ms: Math.round(playbackStart - audioReceivedRef.current),
            total_turn_duration_ms: Math.round(playbackStart - turnStartRef.current),
          });
        }
        break;
      }
      case 'metrics':
        setServerMetrics(event.data);
        console.log(
          `[VOICE:BENCH] metrics received turn_id=${event.data.turn_id} ` +
            `stt_ms=${event.data.stt_latency_ms ?? 'N/A'} ` +
            `llm_ms=${event.data.llm_latency_ms ?? 'N/A'} ` +
            `tts_ms=${event.data.tts_latency_ms ?? 'N/A'} ` +
            `total_ms=${event.data.total_processing_ms ?? 'N/A'}`,
        );
        break;
      case 'completed':
        console.log('[VOICE:UI] Voice turn completed');
        setRecordingState('idle');
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        if (processingTimeoutRef.current) {
          clearTimeout(processingTimeoutRef.current);
          processingTimeoutRef.current = null;
        }
        break;
      case 'error':
        console.error(`[VOICE:UI] Voice turn failed message=${event.message}`);
        setError(event.message);
        setRecordingState('idle');
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        if (processingTimeoutRef.current) {
          clearTimeout(processingTimeoutRef.current);
          processingTimeoutRef.current = null;
        }
        break;
    }
  }, []);

  const startRecording = useCallback(async () => {
    console.log('[VOICE:UI] Start clicked');
    turnStartRef.current = performance.now();
    setError(null);
    setTranscript('');
    setAssistantText('');
    setToolCalls([]);
    setDuration(0);
    setServerMetrics(null);
    setBrowserTimings(null);

    // Check browser support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError('Microphone not supported in this browser');
      return;
    }

    // Request microphone permission
    try {
      console.log('[VOICE:UI] Requesting microphone permission');
      const micStart = performance.now();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      console.log(
        `[VOICE:UI] Microphone permission granted duration_ms=${Math.round(performance.now() - micStart)}`,
      );

      // Connect WebSocket
      setConnectionState('connecting');
      console.log('[VOICE:UI] Creating WebSocket');
      const wsStart = performance.now();
      const ws = new WebSocket(getWsUrl());
      wsRef.current = ws;
      console.log('[VOICE:UI] WebSocket connecting');

      // Chunk counter for summary logging
      recordStartRef.current = performance.now();

      ws.onopen = () => {
        console.log(
          `[VOICE:UI] WebSocket connected duration_ms=${Math.round(performance.now() - wsStart)}`,
        );

        // Send START with configuration
        const sttProv = providers?.stt[0];
        const ttsProv = providers?.tts[0];

        ws.send(
          JSON.stringify({
            type: 'start',
            session_id: sessionId,
            configuration: {
              stt_provider: sttProv?.provider || 'deepgram',
              stt_model: sttProv?.models[0] || 'nova-3',
              llm_provider: selectedLLMProvider || 'openrouter',
              llm_model: llmModel || '',
              tts_provider: ttsProv?.provider || 'elevenlabs',
              tts_model: ttsModel || '',
              tts_voice: ttsProv?.default_voice || '',
            },
          }),
        );
        console.log(`[VOICE:UI] START event sent session_id=${sessionId || 'null'}`);

        // Start MediaRecorder
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm';
        const recorder = new MediaRecorder(stream, { mimeType });
        mediaRecorderRef.current = recorder;
        console.log(`[VOICE:UI] MediaRecorder created mime_type=${mimeType}`);

        // Capture the final blob in ondataavailable, send in onstop
        let finalBlob: Blob | null = null;

        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            finalBlob = e.data;
          }
        };

        recorder.onstop = () => {
          recordingStopRef.current = performance.now();
          console.log(
            `[VOICE:UI] Recording stopped recording_duration_ms=${Math.round(recordingStopRef.current - recordingStartRef.current)}`,
          );

          if (!finalBlob || finalBlob.size === 0) {
            console.error('[VOICE:UI] Final audio blob is empty');
            setError('Recording produced no audio data');
            setRecordingState('idle');
            return;
          }

          console.log(
            `[VOICE:UI] Final audio blob created size_bytes=${finalBlob.size} type=${finalBlob.type || mimeType}`,
          );

          // Convert blob to ArrayBuffer and send, THEN send STOP
          finalBlob.arrayBuffer().then((buffer) => {
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              wsRef.current.send(buffer);
              audioSentRef.current = performance.now();
              console.log(
                `[VOICE:UI] Audio sent size_bytes=${buffer.byteLength} type=${finalBlob.type || mimeType} ` +
                  `audio_upload_ms=${Math.round(audioSentRef.current - recordingStopRef.current)}`,
              );

              // Send STOP only after audio is sent
              wsRef.current.send(JSON.stringify({ type: 'stop' }));
              console.log('[VOICE:UI] STOP event sent');
            } else {
              console.error(
                `[VOICE:UI] Cannot send audio socket_state=${wsRef.current?.readyState ?? 'null'}`,
              );
              setError('WebSocket is not open — cannot send audio');
            }
          });
        };

        recorder.start(); // No timeslice — single blob on stop
        recordingStartRef.current = performance.now();
        setRecordingState('recording');
        console.log('[VOICE:UI] Recording started');

        // Start duration timer
        timerRef.current = setInterval(() => {
          setDuration((d) => d + 1);
        }, 1000);
      };

      ws.onmessage = (e) => {
        try {
          const event: VoiceEvent = JSON.parse(e.data);
          handleEvent(event);
        } catch {
          // Ignore non-JSON messages
        }
      };

      ws.onerror = () => {
        console.error('[VOICE:UI] Voice error message=WebSocket connection error');
        setError('WebSocket connection error');
        setConnectionState('disconnected');
      };

      ws.onclose = (e) => {
        console.log(
          `[VOICE:UI] WebSocket closed code=${e.code} reason=${e.reason || '(none)'}`,
        );
        setConnectionState('disconnected');
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
      };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      if (msg.includes('Permission') || msg.includes('denied')) {
        console.error('[VOICE:UI] Microphone permission denied');
        setError('Microphone permission denied. Please allow microphone access.');
      } else {
        console.error(`[VOICE:UI] Voice error message=${msg}`);
        setError(`Microphone error: ${msg}`);
      }
    }
  }, [getWsUrl, handleEvent, providers, sessionId, selectedLLMProvider, llmModel, ttsModel]);

  const stopRecording = useCallback(() => {
    console.log('[VOICE:UI] Recording stop requested');

    // Stop MediaRecorder — this triggers ondataavailable then onstop
    // Audio send + STOP send happen in onstop handler (correct order)
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    // Stop microphone
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    setRecordingState('processing');
  }, []);

  const formatDuration = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const sttProv = providers?.stt[0];
  const ttsProv = providers?.tts[0];

  const fmtMs = (v: number | null | undefined) => (v == null ? 'N/A' : `${Math.round(v)} ms`);
  const fmtBytes = (v: number | null | undefined) =>
    v == null ? 'N/A' : `${(v / 1024).toFixed(1)} KB`;
  const fmtNum = (v: number | null | undefined) => (v == null ? 'N/A' : String(v));

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h2 className="text-lg font-semibold text-white">Voice Test</h2>
        <div className="flex items-center gap-3">
          {sessionId && (
            <span className="text-xs text-gray-500 font-mono">
              Session: {sessionId.slice(0, 8)}...
            </span>
          )}
        </div>
      </div>

      {/* Provider/Model Selection */}
      <div className="px-4 py-3 border-b border-gray-800 bg-gray-900/50">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {/* STT Configuration */}
          <div className="space-y-1">
            <label className="text-xs text-gray-400 font-medium">STT</label>
            <div className="text-xs text-gray-300">
              {sttProv?.provider || 'deepgram'} ({sttProv?.models[0] || 'nova-3'})
            </div>
          </div>

          {/* LLM Configuration */}
          <div className="space-y-1">
            <div className="flex gap-2">
              <div className="flex-1 min-w-0">
                <label className="text-xs text-gray-400 font-medium">LLM Provider</label>
                <select
                  value={selectedLLMProvider}
                  onChange={(e) => setSelectedLLMProvider(e.target.value)}
                  className="w-full text-xs bg-gray-800 text-gray-200 rounded px-2 py-1.5 border border-gray-700 min-w-0"
                >
                  {providers?.llm.map((p) => (
                    <option key={p.provider} value={p.provider}>
                      {p.provider} {p.configured ? '(configured)' : '(not configured)'}
                    </option>
                  )) || <option value="openrouter">openrouter</option>}
                </select>
              </div>
            </div>
            <div>
              <label className="text-xs text-gray-400 font-medium">Model</label>
              <select
                value={llmModel}
                onChange={(e) => setLlmModel(e.target.value)}
                className="w-full text-xs bg-gray-800 text-gray-200 rounded px-2 py-1.5 border border-gray-700 min-w-0"
              >
                <option value="">
                  default ({providers?.llm.find((p) => p.provider === selectedLLMProvider)?.default_model || 'nvidia/nemotron-3.5-lightning:free'})
                </option>
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

          {/* TTS Configuration */}
          <div className="space-y-1">
            <div>
              <label className="text-xs text-gray-400 font-medium">TTS</label>
              <div className="text-xs text-gray-300">
                {ttsProv?.provider || 'elevenlabs'}
              </div>
            </div>
            <div>
              <label className="text-xs text-gray-400 font-medium">Model</label>
              <select
                value={ttsModel}
                onChange={(e) => setTtsModel(e.target.value)}
                className="w-full text-xs bg-gray-800 text-gray-200 rounded px-2 py-1.5 border border-gray-700 min-w-0"
              >
                <option value="">
                  default ({ttsProv?.default_model || 'eleven_flash_v2_5'})
                </option>
                {ttsProv?.models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
        <div className="flex items-center gap-4 text-xs">
          <span>
            Connection:{' '}
            <span
              className={
                connectionState === 'connected'
                  ? 'text-green-400'
                  : connectionState === 'connecting'
                    ? 'text-yellow-400'
                    : 'text-gray-500'
              }
            >
              {connectionState}
            </span>
          </span>
          <span>
            Recording:{' '}
            <span
              className={
                recordingState === 'recording'
                  ? 'text-red-400'
                  : recordingState === 'processing'
                    ? 'text-yellow-400'
                    : recordingState === 'playing'
                      ? 'text-blue-400'
                      : 'text-gray-500'
              }
            >
              {recordingState}
            </span>
          </span>
          <span className="text-gray-400 font-mono">{formatDuration(duration)}</span>
        </div>
        <div className="flex gap-2">
          {recordingState === 'idle' || recordingState === 'playing' ? (
            <button
              onClick={startRecording}
              disabled={connectionState === 'connecting'}
              className="px-4 py-1.5 bg-red-600 text-white rounded text-sm font-medium hover:bg-red-500 disabled:opacity-50"
            >
              Start
            </button>
          ) : recordingState === 'recording' ? (
            <button
              onClick={stopRecording}
              className="px-4 py-1.5 bg-gray-600 text-white rounded text-sm font-medium hover:bg-gray-500"
            >
              Stop
            </button>
          ) : (
            <button
              disabled
              className="px-4 py-1.5 bg-gray-700 text-gray-400 rounded text-sm font-medium cursor-not-allowed opacity-50"
            >
              Processing...
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4 max-w-full">
        {/* Transcript */}
        {transcript && (
          <div className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 max-w-full">
            <div className="text-xs text-gray-500 mb-1 font-medium uppercase">Transcript</div>
            <div className="text-sm text-gray-200 break-words">{transcript}</div>
          </div>
        )}

        {/* Assistant */}
        {assistantText && (
          <div className="bg-blue-900/30 border border-blue-800 rounded-lg px-4 py-2 max-w-full">
            <div className="text-xs text-blue-400 mb-1 font-medium uppercase">Assistant</div>
            <div className="text-sm text-blue-100 break-words">{assistantText}</div>
          </div>
        )}

        {/* Tool Calls */}
        {toolCalls.length > 0 && (
          <div className="space-y-1 max-w-full">
            <div className="text-xs text-gray-500 font-medium uppercase">Tool Calls</div>
            {toolCalls.map((tc, i) => (
              <div
                key={i}
                className="text-xs bg-orange-900/20 border border-orange-800/50 rounded px-2 py-1 text-orange-300 font-mono break-all"
              >
                {tc.name}({JSON.stringify(tc.args)})
              </div>
            ))}
          </div>
        )}

        {/* Benchmark metrics (Phase 5A) */}
        {serverMetrics && (
          <div className="bg-gray-900 border border-gray-700 rounded-lg px-4 py-3">
            <div className="text-xs text-gray-500 mb-2 font-medium uppercase">Benchmark</div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-gray-500">STT</span>
                <span className="text-gray-200">{fmtMs(serverMetrics.stt_latency_ms)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Tools</span>
                <span className="text-gray-200">{fmtNum(serverMetrics.tool_count)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">LLM</span>
                <span className="text-gray-200">{fmtMs(serverMetrics.llm_latency_ms)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Tokens</span>
                <span className="text-gray-200">{fmtNum(serverMetrics.total_tokens)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">TTS</span>
                <span className="text-gray-200">{fmtMs(serverMetrics.tts_latency_ms)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Audio</span>
                <span className="text-gray-200">{fmtBytes(serverMetrics.tts_audio_bytes)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Server</span>
                <span className="text-gray-200">{fmtMs(serverMetrics.total_processing_ms)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Status</span>
                <span className={serverMetrics.success ? 'text-green-400' : 'text-red-400'}>
                  {serverMetrics.success ? 'Success' : `Failed (${serverMetrics.error_stage})`}
                </span>
              </div>
              {browserTimings && (
                <>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Recording</span>
                    <span className="text-gray-200">
                      {fmtMs(browserTimings.recording_duration_ms)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Total</span>
                    <span className="text-blue-300">
                      {fmtMs(browserTimings.total_turn_duration_ms)}
                    </span>
                  </div>
                </>
              )}
            </div>
            <div className="mt-2 pt-2 border-t border-gray-800 text-xs text-gray-600 font-mono break-all">
              turn_id={serverMetrics.turn_id.slice(0, 8)}...
            </div>
          </div>
        )}

        {/* Empty state */}
        {!transcript && !assistantText && recordingState === 'idle' && (
          <div className="text-center text-gray-600 mt-12">
            Click Start to begin a voice turn.
          </div>
        )}

        {/* Processing indicator */}
        {recordingState === 'processing' && !assistantText && (
          <div className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2">
            <div className="text-sm text-gray-400 animate-pulse">Processing...</div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mx-4 mb-2 bg-red-900/30 border border-red-700 rounded-lg px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Hidden audio element */}
      <audio ref={audioRef} className="hidden" />
    </div>
  );
}
