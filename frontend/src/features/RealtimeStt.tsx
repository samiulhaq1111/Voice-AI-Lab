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
 *   client → server: {"type":"start",...,"llm_provider":"...","llm_model":"...",
 *                     "tts_mode":"elevenlabs"|"browser"} | binary PCM
 *                    | {"type":"tts_mode","mode":"browser"} | {"type":"stop"}
 *   server → client: session_started | transcript_partial | transcript_final
 *                    | utterance_end | agent_processing | agent_response
 *                    | tts_text (Phase 6L browser TTS) | completed | error
 */

import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import type {
  ProviderAvailability,
  RealtimeBrowserTimingMessage,
  RealtimeConnectionState,
  RealtimeEvent,
  RealtimeLogEntry,
  RealtimeStartMessage,
  RealtimeTimings,
  RealtimeTtsMode,
  RealtimeTtsModeMessage,
  RealtimeTurnMetrics,
} from '../types';
import { getProviders } from '../services/api';
import {
  conversationReducer,
  createConversationState,
  currentTurnText,
} from './realtimeTurnModel';
import {
  beginNextSegment,
  clearAudioQueue,
  createAudioQueueState,
  endCurrentSegment,
  enqueueSegment,
  markTurnReported,
} from './realtimeAudioQueue';
import type { AudioQueueState } from './realtimeAudioQueue';

const WS_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(
  'http',
  'ws',
);
const REALTIME_WS_URL = `${WS_BASE}/api/v1/voice/realtime/ws`;

/** AudioWorklet source: converts Float32 mic frames to Int16 PCM and
 * batches them into ~50 ms frames before posting to the main thread.
 *
 * The Web Audio render quantum stays 128 samples. Samples are accumulated
 * across process() callbacks and a frame is posted the moment it is full —
 * no timers. At 48 kHz: 2400 samples = 4800 bytes per frame (~20 frames/s),
 * replacing the previous 256-byte frame per 2.67 ms (~375 messages/s). */
const PCM_WORKLET_CODE = `
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frameSamples = Math.round(sampleRate * 0.05); // ~50 ms @ native rate
    this.durationMs = Math.round((this.frameSamples / sampleRate) * 1000);
    this.buffer = new Int16Array(this.frameSamples);
    this.offset = 0;
    this.quanta = 0;
    this.frames = 0;
    // STOP path: the main thread sends {type:'reset'} before disconnecting
    // so a partially accumulated frame is discarded, never flushed.
    this.port.onmessage = (event) => {
      if (event.data && event.data.type === 'reset') {
        this.offset = 0;
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;

    const f32 = input[0];
    this.quanta += 1;
    for (let i = 0; i < f32.length; i++) {
      const s = Math.max(-1, Math.min(1, f32[i]));
      this.buffer[this.offset++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.offset === this.frameSamples) {
        const frame = this.buffer;
        this.frames += 1;
        this.port.postMessage(
          {
            type: 'pcm_frame',
            buffer: frame.buffer,
            samples: this.frameSamples,
            bytes: frame.byteLength,
            durationMs: this.durationMs,
            quanta: this.quanta,
            frame: this.frames,
          },
          [frame.buffer],
        );
        this.buffer = new Int16Array(this.frameSamples);
        this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm-processor', PCMProcessor);
`;

/** One ~50 ms Int16 PCM frame posted by the PCM worklet. */
interface PcmFrameMessage {
  type: 'pcm_frame';
  /** Raw mono Int16 PCM at the native AudioContext sample rate. */
  buffer: ArrayBuffer;
  samples: number;
  bytes: number;
  durationMs: number;
  /** Cumulative render quanta consumed when the frame was posted. */
  quanta: number;
  /** Cumulative frame counter (1-based). */
  frame: number;
}

const MAX_LOG_ENTRIES = 50;

/** Phase 6L: Web Speech API availability, checked once at module load.
 * When false the Browser TTS option is visibly disabled — never silently
 * broken. */
const BROWSER_TTS_SUPPORTED =
  typeof window !== 'undefined' &&
  'speechSynthesis' in window &&
  typeof SpeechSynthesisUtterance !== 'undefined';

/** Phase 6L: one queued browser-TTS sentence (from a tts_text event). */
interface BrowserTtsItem {
  turn: number | null;
  segmentNo: number;
  text: string;
  /** performance.now() when the tts_text event was received. */
  receivedAt: number;
}

/** Phase 6M: best available English voice, LOCAL (offline) voices first.
 *
 * Remote network voices (e.g. Chrome's "Google …" voices, localService=false)
 * synthesize audio in the cloud, delaying onstart by hundreds of ms. Order:
 * en-US local → any English local → en-US → any English → null (browser
 * default). Never hard-codes a voice name; call only when speechSynthesis
 * is available (BROWSER_TTS_SUPPORTED). */
function pickBrowserVoice(): SpeechSynthesisVoice | null {
  const voices = window.speechSynthesis.getVoices();
  return (
    voices.find((v) => v.localService && v.lang === 'en-US') ??
    voices.find((v) => v.localService && v.lang?.startsWith('en')) ??
    voices.find((v) => v.lang === 'en-US') ??
    voices.find((v) => v.lang?.startsWith('en')) ??
    null
  );
}

/** Phase 6M: one-shot inaudible speechSynthesis warm-up.
 *
 * Chrome initializes its speech engine on the FIRST speak() of the page — a
 * one-time cost that otherwise lands on the user's first real sentence
 * (observed ~500-1000ms before onstart). Speaking a single space at volume
 * 0 during a user gesture (Start click / Browser-mode switch) moves that
 * engine init off the response critical path. Inaudible, not assistant
 * text, never enters the conversation or the browser TTS queue, and runs at
 * most once per component mount. STOP/teardown's speechSynthesis.cancel()
 * clears it if the session ends first. */
function warmUpBrowserSpeech(): void {
  try {
    const utterance = new SpeechSynthesisUtterance(' ');
    utterance.volume = 0;
    // Deliberately no handlers and no queue integration: warm-up is inert.
    window.speechSynthesis.speak(utterance);
  } catch {
    // Warm-up is best-effort only.
  }
}

export default function RealtimeStt() {
  const [providers, setProviders] = useState<ProviderAvailability | null>(null);
  const [connState, setConnState] = useState<RealtimeConnectionState>('disconnected');
  const [micActive, setMicActive] = useState(false);
  // Phase 6E: conversation state lives in a pure reducer — the current user
  // turn evolves in place while speaking and commits ONCE on dispatch.
  const [conversation, dispatchConversation] = useReducer(
    conversationReducer,
    undefined,
    createConversationState,
  );
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [ttsProcessing, setTtsProcessing] = useState(false);
  const [log, setLog] = useState<RealtimeLogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [timings, setTimings] = useState<RealtimeTimings | null>(null);
  const [turnMetrics, setTurnMetrics] = useState<RealtimeTurnMetrics | null>(null);
  const [browserPlaybackLatencyMs, setBrowserPlaybackLatencyMs] = useState<number | null>(
    null,
  );

  // LLM provider/model selection
  const [selectedLLMProvider, setSelectedLLMProvider] = useState('openrouter');
  const [llmModel, setLlmModel] = useState('');
  // Phase 6L: response TTS mode — 'elevenlabs' (server MP3, default) or
  // 'browser' (tts_text events spoken locally via speechSynthesis).
  const [ttsMode, setTtsMode] = useState<RealtimeTtsMode>('elevenlabs');

  const wsRef = useRef<WebSocket | null>(null);
  // Generation counter: incremented on every startRealtime(). Socket
  // callbacks from an old generation are ignored so a previous session can
  // never update the state of a new one.
  const sessionEpochRef = useRef(0);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const silentSinkRef = useRef<GainNode | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const workletUrlRef = useRef<string | null>(null);
  const stopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closingRef = useRef(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  // Phase 6H: ordered playback queue for sentence audio segments; the first
  // segment of a turn starts as soon as it arrives, later segments play in
  // strict sentence order via the <audio> 'ended' handler.
  const audioQueueRef = useRef<AudioQueueState>(createAudioQueueState());
  // Phase 6K.1: the last segment that actually started playing, so a later
  // segment can name the predecessor it FIFO-waited behind instead of
  // mislabeling that intentional wait as "browser latency". Purely diagnostic.
  const lastPlayedSegmentRef = useRef<{
    turn: number | null;
    segmentNo: number | null;
  } | null>(null);
  // Phase 6L: browser-native TTS queue — strictly sequential (one sentence
  // speaks at a time), entirely separate from the ElevenLabs audio queue.
  const browserTtsQueueRef = useRef<BrowserTtsItem[]>([]);
  const browserTtsSpeakingRef = useRef(false);
  // Sensible English voice for browser TTS (asynchronously populated in
  // Chrome via onvoiceschanged); no voice-selection UI yet.
  const browserVoiceRef = useRef<SpeechSynthesisVoice | null>(null);
  // Phase 6M: one-shot engine warm-up guard (per component mount) and the
  // last logged voice key, so the browser_tts_voice diagnostic line is not
  // repeated for every segment.
  const browserTtsWarmupDoneRef = useRef(false);
  const lastVoiceLogKeyRef = useRef<string | null>(null);

  // Load providers on mount
  useEffect(() => {
    getProviders().then(setProviders).catch(() => {});
  }, []);

  // Phase 6L/6M: pick a sensible English voice for browser TTS — LOCAL
  // (offline) voices first, because remote network voices delay onstart.
  // Voices populate asynchronously in Chrome (onvoiceschanged); re-pick
  // whenever they load. The first utterance also re-resolves synchronously
  // if this is still null, so late voices are picked up without blocking.
  useEffect(() => {
    if (!BROWSER_TTS_SUPPORTED) return;
    const refreshVoice = () => {
      browserVoiceRef.current = pickBrowserVoice();
    };
    refreshVoice();
    window.speechSynthesis.addEventListener('voiceschanged', refreshVoice);
    return () => {
      window.speechSynthesis.removeEventListener('voiceschanged', refreshVoice);
    };
  }, []);

  const addLog = useCallback((type: RealtimeLogEntry['type'], detail: string) => {
    const time = new Date().toLocaleTimeString([], { hour12: false });
    setLog((prev) => [...prev.slice(-(MAX_LOG_ENTRIES - 1)), { time, type, detail }]);
  }, []);

  // Cleanup on unmount — release everything
  useEffect(() => {
    return () => {
      // Invalidate this component instance: no stale WebSocket callback may
      // act after unmount/remount, and all session resources are released.
      sessionEpochRef.current += 1;
      closingRef.current = true;
      if (stopTimeoutRef.current) clearTimeout(stopTimeoutRef.current);
      teardownAudio();
      wsRef.current?.close();
      wsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Stop mic tracks, disconnect worklet, close AudioContext. */
  const teardownAudio = useCallback(() => {
    if (workletNodeRef.current) {
      // STOP path: detach the frame handler first (no frame is forwarded
      // after STOP), then reset the worklet PCM accumulator. The node is
      // disconnected below, so no partial frame is ever sent.
      workletNodeRef.current.port.onmessage = null;
      try {
        workletNodeRef.current.port.postMessage({ type: 'reset' });
      } catch {
        // Port already closed — the accumulator dies with the worklet node.
      }
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
    // Stop any queued/playing session audio and drop stale media handlers so
    // a previous session's TTS can never keep playing or update this UI.
    const audioEl = audioRef.current;
    if (audioEl) {
      audioEl.onplaying = null;
      audioEl.onended = null;
      audioEl.onerror = null;
      audioEl.pause();
      if (audioEl.src) {
        URL.revokeObjectURL(audioEl.src);
        audioEl.removeAttribute('src');
      }
    }
    // Phase 6H: drop every queued sentence segment (and its object URL) so a
    // stopped session can never play stale audio from earlier sentences.
    const cleared = clearAudioQueue(audioQueueRef.current);
    audioQueueRef.current = cleared.state;
    for (const url of cleared.revokeUrls) {
      URL.revokeObjectURL(url);
    }
    // Phase 6L: stop browser-native speech and drop every pending tts_text
    // sentence — a stopped/old session must never speak stale text. Runs on
    // STOP, completed, error, close and unmount (no-op in elevenlabs mode).
    if (BROWSER_TTS_SUPPORTED) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // speechSynthesis not ready — nothing to cancel.
      }
    }
    browserTtsQueueRef.current = [];
    browserTtsSpeakingRef.current = false;
    setMicActive(false);
  }, []);

  /** Phase 6H: play the next queued sentence segment (strict order).
   *
   * Phase 6K.1: onplaying reports true first-segment startup latency when
   * nothing played before it, and the intentional FIFO queue wait behind the
   * previous segment otherwise — never mislabeled as "browser latency".
   *
   * Declared as a named function expression so the 'ended'/'error'/blocked
   * handlers can safely re-invoke it. Every handler first checks that its
   * segment is still current, so a double failure signal (e.g. onerror plus
   * a rejected play()) can never skip or kill a later segment.
   */
  const playNextSegment = useCallback(
    function playNext(): void {
      const audioEl = audioRef.current;
      if (!audioEl) return;
      const promoted = beginNextSegment(audioQueueRef.current);
      audioQueueRef.current = promoted.state;
      const segment = promoted.next;
      if (!segment) return;

      if (audioEl.src) {
        URL.revokeObjectURL(audioEl.src);
      }
      audioEl.src = segment.url;

      const finishSegment = (logLine: string) => {
        // Ignore stale signals: the element may already be playing another
        // segment (onerror + rejected play() can both fire for one broken
        // segment — only the first signal advances the queue).
        if (audioQueueRef.current.current !== segment) return;
        const finished = endCurrentSegment(audioQueueRef.current);
        audioQueueRef.current = finished.state;
        if (finished.revokeUrl) {
          URL.revokeObjectURL(finished.revokeUrl);
        }
        if (logLine) addLog('audio', logLine);
        playNext();
      };

      // Capture actual audio playback start (browser-local measurement).
      // Phase 6K.1: the same WS-received → playing clock measurement gets an
      // honest label per segment — true startup latency when nothing played
      // before it, and the intentional FIFO queue wait behind a predecessor
      // otherwise. Only the FIRST playing segment of a turn updates the UI
      // latency field and reports to the backend.
      audioEl.onplaying = () => {
        if (audioQueueRef.current.current !== segment) return;
        const playingAt = performance.now();
        const elapsedMs = Math.round(playingAt - segment.receivedAt);
        const previous = lastPlayedSegmentRef.current;
        lastPlayedSegmentRef.current = {
          turn: segment.turn,
          segmentNo: segment.segmentNo ?? null,
        };
        let firstOfTurn = true;
        if (segment.turn !== null) {
          const claim = markTurnReported(audioQueueRef.current, segment.turn);
          audioQueueRef.current = claim.state;
          firstOfTurn = claim.first;
        } else {
          firstOfTurn = previous === null;
        }
        const predecessor =
          previous === null
            ? null
            : previous.segmentNo == null
              ? 'previous segment'
              : previous.turn !== null && previous.turn === segment.turn
                ? `segment ${previous.segmentNo}`
                : `turn ${previous.turn} segment ${previous.segmentNo}`;
        if (predecessor === null) {
          console.log(
            `[VOICE:UI] audio_playing first_segment_playback_latency_ms=${elapsedMs}`,
          );
          addLog('audio', `playing (first-segment playback latency=${elapsedMs}ms)`);
        } else if (firstOfTurn) {
          // First audio of a new turn that still waited behind the previous
          // turn's tail — honest label, but it owns this turn's report/UI.
          console.log(`[VOICE:UI] audio_playing queue_wait_ms=${elapsedMs}`);
          addLog(
            'audio',
            `playing (first of turn, queued behind ${predecessor}, wait=${elapsedMs}ms)`,
          );
        } else {
          console.log(`[VOICE:UI] audio_playing queue_wait_ms=${elapsedMs}`);
          addLog('audio', `playing (queued behind ${predecessor}, wait=${elapsedMs}ms)`);
        }
        if (firstOfTurn) {
          setBrowserPlaybackLatencyMs(elapsedMs);
        }
        // Phase 6G: report WS-received → playing back to the backend so it
        // can log the full utterance → playing breakdown for this turn.
        // Phase 6H/6K.1: only the FIRST playing segment of a turn reports —
        // later segments only waited in the FIFO queue, by design.
        if (firstOfTurn && segment.turn !== null) {
          const ws = wsRef.current;
          if (ws && ws.readyState === WebSocket.OPEN) {
            const report: RealtimeBrowserTimingMessage = {
              type: 'browser_timing',
              turn: segment.turn,
              ws_transit_ms: segment.wsTransitMs,
              received_to_playing_ms: elapsedMs,
            };
            ws.send(JSON.stringify(report));
            addLog(
              'browser_timing',
              `turn=${segment.turn} transit=${segment.wsTransitMs ?? 'n/a'}ms playing_in=${elapsedMs}ms`,
            );
          }
        }
      };
      audioEl.onended = () => finishSegment('segment ended');
      audioEl.onerror = () => {
        addLog('error', 'Audio segment playback failed');
        finishSegment('');
      };
      audioEl.play().catch((e) => {
        console.error('[REALTIME] Audio playback failed:', e);
        addLog('error', `Audio playback blocked: ${e.message}`);
        finishSegment('');
      });
    },
    [addLog],
  );

  /** Phase 6L: speak the next queued browser-TTS sentence (strict order).
   *
   * One sentence speaks at a time — sentence 2 never interrupts sentence 1:
   * the next utterance is only built after the current one's onend/onerror
   * advances the queue. The session-epoch guard makes stale utterances from
   * an old session inert (teardown also cancels + clears the queue).
   *
   * Phase 6M: per-stage diagnostics decompose the tts_text → speech delay —
   * queue_ms (our FIFO), engine_ms (speak() → onstart: the browser engine
   * under investigation), startup_ms (receipt → onstart: the user-facing
   * metric, renamed from start_latency) — and the resolved voice is logged
   * once per distinct selection.
   */
  const speakNextBrowserTts = useCallback(
    function speakNext(epoch: number): void {
      if (!BROWSER_TTS_SUPPORTED) return;
      if (browserTtsSpeakingRef.current) return;
      const item = browserTtsQueueRef.current.shift();
      if (!item) return;
      browserTtsSpeakingRef.current = true;
      // Phase 6M: voices may have finished loading since the last pick —
      // resolve synchronously right before the utterance so the first
      // sentence never falls back to the browser default voice needlessly.
      // Never blocks: getVoices() just returns whatever is loaded.
      if (browserVoiceRef.current === null) {
        browserVoiceRef.current = pickBrowserVoice();
      }
      const voice = browserVoiceRef.current;
      const preparedAt = performance.now();
      const utterance = new SpeechSynthesisUtterance(item.text);
      if (voice) {
        utterance.voice = voice;
      }
      const turnLabel = item.turn ?? 'n/a';
      addLog(
        'browser_tts',
        `browser_tts_prepare turn=${turnLabel} segment=${item.segmentNo}` +
          ` queue_ms=${Math.round(preparedAt - item.receivedAt)}`,
      );
      const voiceKey = voice
        ? `${voice.name}|${voice.lang}|${voice.localService}`
        : 'default';
      if (voiceKey !== lastVoiceLogKeyRef.current) {
        lastVoiceLogKeyRef.current = voiceKey;
        const voiceLine = voice
          ? `browser_tts_voice name="${voice.name}" lang="${voice.lang}" localService=${voice.localService}`
          : `browser_tts_voice name="(browser default)" lang="n/a" localService=n/a` +
            ` voices_loaded=${window.speechSynthesis.getVoices().length}`;
        addLog('browser_tts', voiceLine);
        console.log(`[VOICE:UI] ${voiceLine}`);
      }
      addLog(
        'browser_tts',
        `browser_tts_speak_called turn=${turnLabel} segment=${item.segmentNo}`,
      );
      const speakCalledAt = performance.now();
      let startedAt: number | null = null;
      utterance.onstart = () => {
        if (sessionEpochRef.current !== epoch) return;
        startedAt = performance.now();
        setTtsProcessing(false);
        const startupMs = Math.round(startedAt - item.receivedAt);
        const engineMs = Math.round(startedAt - speakCalledAt);
        // tts_text received → speech actually started (NOT network latency).
        console.log(
          `[VOICE:UI] browser_tts_started turn=${turnLabel} segment=${item.segmentNo}` +
            ` startup_ms=${startupMs} engine_ms=${engineMs}`,
        );
        addLog(
          'browser_tts',
          `browser_tts_started turn=${turnLabel} segment=${item.segmentNo}` +
            ` startup_ms=${startupMs} engine_ms=${engineMs}`,
        );
      };
      let finished = false;
      const finish = (detail: string) => {
        // onend + onerror can both fire for one utterance — advance once.
        if (finished) return;
        finished = true;
        browserTtsSpeakingRef.current = false;
        if (sessionEpochRef.current !== epoch) return;
        addLog('browser_tts', detail);
        speakNext(epoch);
      };
      utterance.onend = () => {
        const durationMs =
          startedAt != null ? Math.round(performance.now() - startedAt) : null;
        finish(
          `browser_tts_ended turn=${turnLabel} segment=${item.segmentNo}` +
            (durationMs != null ? ` duration=${durationMs}ms` : ''),
        );
      };
      utterance.onerror = (e) => {
        finish(
          `browser_tts_error turn=${turnLabel} segment=${item.segmentNo}` +
            ` error=${e.error ?? 'unknown'}`,
        );
      };
      window.speechSynthesis.speak(utterance);
    },
    [addLog],
  );

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
          // One evolving current turn: partials replace in place and are
          // never committed as conversation messages.
          dispatchConversation({ type: 'partial', text: event.text });
          addLog('transcript_partial', `"${event.text}"`);
          break;
        case 'transcript_final':
          // Finalized segments accumulate into the same turn; the turn is
          // committed only when the backend dispatches it (agent_processing).
          dispatchConversation({ type: 'final', text: event.text });
          addLog('transcript_final', `"${event.text}"`);
          break;
        case 'utterance_end':
          addLog('utterance_end', '');
          break;
        case 'agent_processing':
          // The turn was released by the backend: commit ONE user message.
          dispatchConversation({ type: 'agent_processing' });
          addLog('agent_processing', '');
          break;
        case 'agent_response':
          dispatchConversation({
            type: 'agent_response',
            text: event.text,
            toolCalls: event.tool_calls,
            iterations: event.iterations,
          });
          addLog(
            'agent_response',
            `text="${event.text.slice(0, 50)}${event.text.length > 50 ? '…' : ''}" tools=${event.tool_calls} iterations=${event.iterations}`,
          );
          break;
        case 'tts_processing':
          setTtsProcessing(true);
          addLog('tts_processing', '');
          break;
        case 'tts_text': {
          // Phase 6L browser TTS: queue the sentence text; speechSynthesis
          // speaks strictly in sentence order (one at a time). No audio
          // events exist in this mode.
          const turnNo = typeof event.turn === 'number' ? event.turn : null;
          const segmentNo = typeof event.segment === 'number' ? event.segment : 1;
          addLog('tts_text', `turn=${turnNo ?? 'n/a'} segment=${segmentNo}`);
          browserTtsQueueRef.current.push({
            turn: turnNo,
            segmentNo,
            text: event.text,
            receivedAt: performance.now(),
          });
          speakNextBrowserTts(sessionEpochRef.current);
          break;
        }
        case 'audio': {
          setTtsProcessing(false);
          // Phase 6G: capture the backend turn + server send timestamp so the
          // playback-start report below can be correlated server-side.
          const audioTurn = typeof event.turn === 'number' ? event.turn : null;
          const wsTransitMs =
            typeof event.sent_epoch_ms === 'number'
              ? Math.max(0, Math.round(Date.now() - event.sent_epoch_ms))
              : null;
          // Decode base64 audio segment
          const binary = atob(event.data);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
          }
          const blob = new Blob([bytes], { type: event.format || 'audio/mpeg' });
          const url = URL.createObjectURL(blob);
          const segmentNo = typeof event.segment === 'number' ? event.segment : 1;
          addLog(
            'audio',
            `segment=${segmentNo} format=${event.format} size=${bytes.length}B`,
          );

          // Create or reuse audio element for playback
          if (!audioRef.current) {
            audioRef.current = new Audio();
          }
          // Phase 6H: segments are queued and played strictly in sentence
          // order; the first segment of a turn starts as soon as it arrives.
          // Phase 6K.1: the segment index rides along purely for queue-wait
          // diagnostics ("queued behind segment N").
          audioQueueRef.current = enqueueSegment(audioQueueRef.current, {
            url,
            turn: audioTurn,
            segmentNo,
            wsTransitMs,
            receivedAt: performance.now(),
          });
          playNextSegment();
          break;
        }
        case 'turn_metrics': {
          setTurnMetrics(event.data);
          const d = event.data;
          addLog(
            'turn_metrics',
            `turn=${d.turn} utterance_to_agent=${d.utterance_end_to_agent_ms ?? 'N/A'}ms ` +
              `agent=${d.agent_processing_ms ?? 'N/A'}ms tts=${d.tts_duration_ms ?? 'N/A'}ms ` +
              `total=${d.utterance_end_to_audio_sent_ms ?? 'N/A'}ms`,
          );
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
          dispatchConversation({ type: 'agent_failed' });
          setTtsProcessing(false);
          setError(event.message);
          addLog('error', `${event.stage ? `[${event.stage}] ` : ''}${event.message}`);
          teardownAudio();
          break;
      }
    },
    [addLog, teardownAudio, playNextSegment, speakNextBrowserTts],
  );

  const startRealtime = useCallback(async () => {
    if (wsRef.current || connState !== 'disconnected') return; // prevent duplicates
    // New session generation: invalidates every callback from a previous
    // socket or audio graph that is still unwinding in the background.
    const epoch = ++sessionEpochRef.current;
    setError(null);
    dispatchConversation({ type: 'reset' });
    setSessionId(null);
    setTimings(null);
    setTurnMetrics(null);
    setBrowserPlaybackLatencyMs(null);
    audioQueueRef.current = createAudioQueueState();
    lastPlayedSegmentRef.current = null;
    // Phase 6L: no stale browser speech may leak into the new session.
    if (BROWSER_TTS_SUPPORTED) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // speechSynthesis not ready — nothing to cancel.
      }
    }
    browserTtsQueueRef.current = [];
    browserTtsSpeakingRef.current = false;
    // Phase 6M: each session re-logs its resolved voice once.
    lastVoiceLogKeyRef.current = null;
    setLog([]);
    closingRef.current = false;

    // Phase 6M: warm the speech engine once, inside this Start user gesture,
    // so Chrome's one-time engine initialization (~0.5-1s before onstart)
    // does not land on the first real sentence of a browser-TTS session.
    // Silent: a single space at volume 0, never queued, never shown as
    // assistant text, at most once per component mount.
    if (ttsMode === 'browser' && !browserTtsWarmupDoneRef.current) {
      browserTtsWarmupDoneRef.current = true;
      warmUpBrowserSpeech();
      addLog('browser_tts', 'browser_tts_warmup (silent, volume=0)');
    }

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
        if (sessionEpochRef.current !== epoch) return;
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
        // Phase 6L: response TTS mode — elevenlabs (server MP3) or browser
        // (tts_text events + local speechSynthesis).
        startMsg.tts_mode = ttsMode;
        ws.send(JSON.stringify(startMsg));
        addLog(
          'ws',
          `START sent sample_rate=${audioCtx.sampleRate} llm=${selectedLLMProvider}/${llmModel || 'default'} tts=${ttsMode}`,
        );

        // 5. Worklet PCM frames → WS binary frames.
        // The worklet batches ~50 ms Int16 frames (2400 samples / 4800 bytes
        // @ 48 kHz, ~20 msg/s). One binary WS message per full frame; the
        // backend protocol is unchanged.
        worklet.port.onmessage = (e: MessageEvent<PcmFrameMessage>) => {
          if (sessionEpochRef.current !== epoch) return;
          const msg = e.data;
          if (!msg || msg.type !== 'pcm_frame') return;
          if (ws.readyState !== WebSocket.OPEN) return;
          ws.send(msg.buffer);
          // Temporary diagnostics — first frames, then every 100th (~5 s)
          if (msg.frame <= 3 || msg.frame % 100 === 0) {
            const line =
              `pcm_batch samples=${msg.samples} bytes=${msg.bytes} ` +
              `duration_ms=${msg.durationMs} quanta_received=${msg.quanta} ` +
              `frames_sent=${msg.frame}`;
            console.log(`[VOICE:UI] ${line}`);
            addLog('mic', line);
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
        if (sessionEpochRef.current !== epoch) return;
        try {
          handleServerEvent(JSON.parse(e.data as string) as RealtimeEvent);
        } catch {
          // ignore non-JSON frames
        }
      };

      ws.onerror = () => {
        if (sessionEpochRef.current !== epoch) return;
        setError('Realtime WebSocket connection error');
        addLog('error', 'WebSocket connection error');
      };

      ws.onclose = () => {
        if (sessionEpochRef.current !== epoch) {
          // Socket of a previous session — never touch current state.
          return;
        }
        setConnState('disconnected');
        teardownAudio();
        if (!closingRef.current) {
          addLog('ws', 'closed');
        }
        if (wsRef.current === ws) {
          wsRef.current = null;
        }
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
  }, [addLog, connState, handleServerEvent, teardownAudio, ttsMode]);

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

  /** Phase 6L: switch the realtime response TTS mode.
   *
   * Browser speech is cancelled and its queue cleared on EVERY switch (no
   * browser speech outlives the mode it belongs to). The backend applies
   * the new mode from the NEXT turn — an already-playing ElevenLabs response
   * is never interrupted.
   */
  const handleTtsModeSelect = useCallback(
    (mode: RealtimeTtsMode) => {
      if (mode === ttsMode) return;
      if (mode === 'browser' && !BROWSER_TTS_SUPPORTED) {
        setError('Browser speech synthesis is not available in this browser');
        addLog('error', 'Browser TTS unavailable (no speechSynthesis)');
        return;
      }
      setTtsMode(mode);
      if (BROWSER_TTS_SUPPORTED) {
        try {
          window.speechSynthesis.cancel();
        } catch {
          // speechSynthesis not ready — nothing to cancel.
        }
      }
      browserTtsQueueRef.current = [];
      browserTtsSpeakingRef.current = false;
      // Phase 6M: switching to browser mode is also a user gesture — warm
      // the engine once so the first browser-mode sentence starts fast.
      if (mode === 'browser' && !browserTtsWarmupDoneRef.current) {
        browserTtsWarmupDoneRef.current = true;
        warmUpBrowserSpeech();
        addLog('browser_tts', 'browser_tts_warmup (silent, volume=0)');
      }
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        const msg: RealtimeTtsModeMessage = { type: 'tts_mode', mode };
        ws.send(JSON.stringify(msg));
        addLog('ws', `tts_mode=${mode} (applies to next turn)`);
      } else {
        addLog('ws', `tts_mode=${mode}`);
      }
    },
    [addLog, ttsMode],
  );

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
              onClick={() => handleTtsModeSelect('elevenlabs')}
              className={
                ttsMode === 'elevenlabs'
                  ? 'px-2 py-0.5 bg-blue-600 text-white rounded text-xs font-medium'
                  : 'px-2 py-0.5 bg-gray-700 text-gray-300 rounded text-xs hover:bg-gray-600'
              }
            >
              ElevenLabs
            </button>
            <button
              onClick={() => handleTtsModeSelect('browser')}
              disabled={!BROWSER_TTS_SUPPORTED}
              title={
                BROWSER_TTS_SUPPORTED
                  ? 'Speak responses locally via speechSynthesis'
                  : 'speechSynthesis is not available in this browser'
              }
              className={
                ttsMode === 'browser'
                  ? 'px-2 py-0.5 bg-blue-600 text-white rounded text-xs font-medium'
                  : 'px-2 py-0.5 bg-gray-700 text-gray-300 rounded text-xs hover:bg-gray-600 disabled:opacity-40 disabled:hover:bg-gray-700 disabled:cursor-not-allowed'
              }
            >
              Browser{!BROWSER_TTS_SUPPORTED ? ' (unavailable)' : ''}
            </button>
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
                !currentTurnText(conversation) && (
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
          {currentTurnText(conversation) && (
            <div className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2">
              <div className="text-xs text-gray-500 mb-1 font-medium uppercase">
                Current turn
              </div>
              <div className="text-sm text-yellow-200 break-words min-h-[1.25rem]">
                {currentTurnText(conversation)}
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
