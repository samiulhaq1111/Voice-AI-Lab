/** Realtime Voice Agent engine — shared by the developer screen (RealtimeStt)
 * and the Voice Chat UI (VoiceChat). Extracted verbatim from RealtimeStt;
 * the streaming path is unchanged:
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
 *                    | utterance_end | agent_processing | agent_delta
 *                    | tool_progress | agent_response
 *                    | tts_text (Phase 6L browser TTS) | completed | error
 */

import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import type {
  RealtimeBrowserTimingMessage,
  RealtimeConnectionState,
  RealtimeEvent,
  RealtimeLogEntry,
  RealtimeStartMessage,
  RealtimeTimings,
  RealtimeTtsMode,
  RealtimeTtsModeMessage,
  RealtimeTraceEvent,
  RealtimeTurnMetrics,
} from '../types';
import {
  conversationReducer,
  createConversationState,
  currentTurnText,
} from './realtimeTurnModel';
import type { ConversationState } from './realtimeTurnModel';
import {
  beginNextSegment,
  clearAudioQueue,
  createAudioQueueState,
  endCurrentSegment,
  enqueueSegment,
  markTurnReported,
} from './realtimeAudioQueue';
import type { AudioQueueState, AudioSegment } from './realtimeAudioQueue';

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

/** Phase 6L: Web Speech API availability, checked once at module load.
 * When false the Browser TTS option is visibly disabled — never silently
 * broken. */
const BROWSER_TTS_SUPPORTED =
  typeof window !== 'undefined' &&
  'speechSynthesis' in window &&
  typeof SpeechSynthesisUtterance !== 'undefined';

/** Phase 2: MediaSource MP3 support, checked once at module load. When true
 * the START message advertises "audio_stream": true and the backend
 * delivers sentence audio incrementally (audio_chunk/audio_end) — playback
 * starts before the sentence finishes synthesizing. When false the legacy
 * complete-segment protocol is used, unchanged. */
const AUDIO_STREAM_SUPPORTED = (() => {
  try {
    return (
      typeof MediaSource !== 'undefined' &&
      MediaSource.isTypeSupported('audio/mpeg')
    );
  } catch {
    return false;
  }
})();

/** Phase 2: base64 → bytes for received audio payloads. Returns an
 * ArrayBuffer so streamed chunks are directly appendBuffer-compatible. */
function decodeBase64Audio(data: string): ArrayBuffer {
  const binary = atob(data);
  const buffer = new ArrayBuffer(binary.length);
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return buffer;
}

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
 * one-time cost that otherwise lands on the user's first real sentence. A
 * single space at volume 0 during a user gesture (Start click / Browser-mode
 * switch) moves that engine init off the response critical path. Inaudible,
 * never queued, never part of the conversation; at most once per hook mount.
 * STOP/teardown's speechSynthesis.cancel() clears it if the session ends
 * first. */
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

/** Trace helper: milliseconds since a performance.now() mark ('n/a' when unset). */
function since(mark: number): string {
  return mark > 0 ? `${Math.round(performance.now() - mark)}ms` : 'n/a';
}

/** Trace helper: flatten whitespace, make embedded quotes safe, truncate. */
function clip(text: string, max: number): string {
  const flat = text.replace(/\s+/g, ' ').replace(/"/g, "'").trim();
  return flat.length > max ? `${flat.slice(0, max)}…` : flat;
}

/** Selected LLM provider/model for one realtime session start (sent in START). */
export interface RealtimeStartConfig {
  llmProvider?: string;
  llmModel?: string;
}

export interface UseRealtimeVoiceOptions {
  /** Optional sink for the engine's diagnostic lines. The developer screen
   * feeds its Events panel through this; Voice Chat feeds its Debug Logs
   * drawer. For trace lines `type` is a RealtimeTraceEvent (uppercase). */
  onDiagnostic?: (
    type: RealtimeLogEntry['type'],
    detail: string,
    turn?: number | null,
  ) => void;
}

export interface UseRealtimeVoiceResult {
  connState: RealtimeConnectionState;
  micActive: boolean;
  sessionId: string | null;
  error: string | null;
  ttsProcessing: boolean;
  conversation: ConversationState;
  currentTurn: string;
  timings: RealtimeTimings | null;
  turnMetrics: RealtimeTurnMetrics | null;
  browserPlaybackLatencyMs: number | null;
  ttsMode: RealtimeTtsMode;
  browserTtsSupported: boolean;
  start: (config: RealtimeStartConfig) => Promise<void>;
  stop: () => void;
  setTtsMode: (mode: RealtimeTtsMode) => void;
}

export default function useRealtimeVoice(
  options: UseRealtimeVoiceOptions = {},
): UseRealtimeVoiceResult {
  // Diagnostic sink held in a ref so the engine callbacks keep the same
  // stable identities the original component had (addLog had no deps there).
  const onDiagnosticRef = useRef(options.onDiagnostic);
  useEffect(() => {
    onDiagnosticRef.current = options.onDiagnostic;
  });

  const addLog = useCallback(
    (type: RealtimeLogEntry['type'], detail: string, turn?: number | null) => {
      onDiagnosticRef.current?.(type, detail, turn);
    },
    [],
  );

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
  const [error, setError] = useState<string | null>(null);
  const [timings, setTimings] = useState<RealtimeTimings | null>(null);
  const [turnMetrics, setTurnMetrics] = useState<RealtimeTurnMetrics | null>(null);
  const [browserPlaybackLatencyMs, setBrowserPlaybackLatencyMs] = useState<number | null>(
    null,
  );

  // Phase 6L: response TTS mode — 'elevenlabs' (server MP3, default) or
  // 'browser' (tts_text events spoken locally via speechSynthesis).
  const [ttsModeState, setTtsModeState] = useState<RealtimeTtsMode>('elevenlabs');

  const wsRef = useRef<WebSocket | null>(null);
  // Generation counter: incremented on every start(). Socket
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
  // Phase 6M: one-shot engine warm-up guard (per hook mount) and the last
  // logged voice key, so the browser_tts_voice diagnostic line is not
  // repeated for every segment.
  const browserTtsWarmupDoneRef = useRef(false);
  const lastVoiceLogKeyRef = useRef<string | null>(null);
  // Barge-in (interrupt while the assistant speaks): turn-level staleness
  // gate built on the server turn labels. latestMediaTurnRef = highest
  // assistant turn seen; lastMetricsTurnRef = last completed turn;
  // activeTurnRef = turn currently being produced (0 = idle);
  // bargeInCutoffRef = set on interruption — frames with turn <= cutoff are
  // discarded until a newer turn's output arrives.
  const latestMediaTurnRef = useRef(0);
  const lastMetricsTurnRef = useRef(0);
  const activeTurnRef = useRef(0);
  const bargeInCutoffRef = useRef<number | null>(null);
  // Phase 2: streamed (audio_chunk) segments by `${turn}:${segment}` so
  // later chunks and audio_end of a segment reach the same queue item.
  const streamedSegmentsRef = useRef(new Map<string, AudioSegment>());

  // --- Realtime trace (Voice Chat "Debug Logs" drawer) --------------------
  // Trace lines ride the SAME onDiagnostic channel the developer screen's
  // Events panel uses — this is not a second logging system. Uppercase event
  // names let the drawer filter trace lines from ordinary diagnostics.
  const lastTraceAtRef = useRef(0);
  const trace = useCallback(
    (event: RealtimeTraceEvent, detail: string, turn: number | null) => {
      const nowMs = performance.now();
      const gap =
        lastTraceAtRef.current > 0
          ? ` since_prev=${Math.round(nowMs - lastTraceAtRef.current)}ms`
          : '';
      lastTraceAtRef.current = nowMs;
      addLog(event, `${detail}${gap}`.trim(), turn);
    },
    [addLog],
  );
  // Turn label for events emitted between turns: the in-flight turn when one
  // is active, otherwise one past the last completed turn.
  const pendingSpeechTurn = useCallback(
    () =>
      activeTurnRef.current !== 0
        ? activeTurnRef.current
        : lastMetricsTurnRef.current + 1,
    [],
  );
  // Per-speech-episode marks: STT_PARTIAL logs the first partial immediately
  // and then at most once per second; finals feed UTTERANCE_END/TURN_RELEASED.
  const speechMarksRef = useRef({
    startedAt: 0,
    lastPartialLogAt: 0,
    partialsSkipped: 0,
    speechFinalSeen: false,
    finals: 0,
  });
  const noteSpeechStarted = useCallback(() => {
    const marks = speechMarksRef.current;
    if (marks.startedAt !== 0) return;
    marks.startedAt = performance.now();
    marks.lastPartialLogAt = 0;
    marks.partialsSkipped = 0;
    marks.speechFinalSeen = false;
    marks.finals = 0;
  }, []);
  // Per-turn marks (re-armed on agent_processing): cheap since_* references
  // for latency lines plus the once-per-turn trace gates.
  const traceTurnRef = useRef({
    turn: 0,
    startedAt: 0,
    llmRequestAt: 0,
    firstTokenSeen: false,
    firstSentenceSeen: false,
    firstAudioSeen: false,
    responseStarted: false,
    ackedTools: new Set<string>(),
  });
  // Session-scoped trace metadata (START config) for LLM_REQUEST/SESSION_START.
  const sessionTraceRef = useRef({ llmProvider: '', llmModel: '' });
  // Live TTS mode for trace lines emitted outside the render closure.
  const ttsModeRef = useRef(ttsModeState);
  useEffect(() => {
    ttsModeRef.current = ttsModeState;
  }, [ttsModeState]);

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

  // Cleanup on unmount — release everything
  useEffect(() => {
    return () => {
      // Invalidate this hook instance: no stale WebSocket callback may act
      // after unmount, and all session resources are released.
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
    const hadMic = micStreamRef.current !== null;
    const hadPlayback =
      audioQueueRef.current.current !== null ||
      audioQueueRef.current.pending.length > 0 ||
      browserTtsQueueRef.current.length > 0 ||
      browserTtsSpeakingRef.current;
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
    // Phase 2: streamed-segment registry dies with the queue — later chunks
    // of a torn-down session have nowhere to land and are ignored.
    streamedSegmentsRef.current.clear();
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
    // Barge-in gate: a torn-down session has nothing left to invalidate.
    bargeInCutoffRef.current = null;
    setMicActive(false);
    // Trace: only report what was actually running (teardown is idempotent).
    if (hadMic) trace('MIC_STOPPED', '', null);
    if (hadPlayback) trace('AUDIO_QUEUE_CLEARED', 'reason=teardown', null);
  }, [trace]);

  /** Barge-in: cancel ONLY assistant playback (ElevenLabs audio queue and
   * browser speech). The microphone, AudioWorklet, AudioContext and the
   * WebSocket keep running — the user's utterance becomes the next turn.
   * Playback-only subset of teardownAudio(); no session teardown. */
  const cancelAssistantPlayback = useCallback(() => {
    const wasPlaying =
      audioQueueRef.current.current !== null ||
      audioQueueRef.current.pending.length > 0 ||
      browserTtsQueueRef.current.length > 0 ||
      browserTtsSpeakingRef.current;
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
    const cleared = clearAudioQueue(audioQueueRef.current);
    audioQueueRef.current = cleared.state;
    for (const url of cleared.revokeUrls) {
      URL.revokeObjectURL(url);
    }
    // Phase 2: streamed-segment registry dies with the queue.
    streamedSegmentsRef.current.clear();
    // The interrupted segment must not label the next turn's diagnostics.
    lastPlayedSegmentRef.current = null;
    if (BROWSER_TTS_SUPPORTED) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // speechSynthesis not ready — nothing to cancel.
      }
    }
    browserTtsQueueRef.current = [];
    browserTtsSpeakingRef.current = false;
    // Trace: the interruption pair (BARGE_IN_DETECTED is emitted by
    // maybeBargeIn just before this runs). Turn = the cutoff turn.
    if (wasPlaying) {
      trace('PLAYBACK_CANCELLED', 'reason=barge_in', bargeInCutoffRef.current);
      trace('AUDIO_QUEUE_CLEARED', 'reason=barge_in', bargeInCutoffRef.current);
    }
  }, [trace]);

  /** Barge-in trigger: cut the assistant's playback the moment the user
   * speaks again — but only while assistant audio is active or a response
   * is in flight. The first qualifying interruption sets the turn cutoff
   * from the server turn labels; repeat partials of the same episode are
   * inert (no duplicate log), while interrupting a newer in-flight turn
   * extends the cutoff. start()/teardown() reset the gate. */
  const maybeBargeIn = useCallback(
    (source: string, text: string) => {
      const playbackActive =
        audioQueueRef.current.current !== null ||
        audioQueueRef.current.pending.length > 0 ||
        browserTtsQueueRef.current.length > 0 ||
        browserTtsSpeakingRef.current;
      if (!playbackActive && activeTurnRef.current === 0) return;
      const cutoff = Math.max(
        latestMediaTurnRef.current,
        activeTurnRef.current,
      );
      if (bargeInCutoffRef.current !== null && cutoff <= bargeInCutoffRef.current) {
        return; // already covered by an earlier interruption
      }
      bargeInCutoffRef.current = cutoff;
      trace(
        'BARGE_IN_DETECTED',
        `source=${source} text="${clip(text, 40)}" cutoff_turn=${cutoff}`,
        cutoff,
      );
      cancelAssistantPlayback();
      addLog('barge_in', `source=${source} cutoff_turn=${cutoff}`);
    },
    [trace, cancelAssistantPlayback, addLog],
  );

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
      const segLabel = segment.segmentNo != null ? String(segment.segmentNo) : 'n/a';
      let playbackStartedAt: number | null = null;

      if (audioEl.src) {
        URL.revokeObjectURL(audioEl.src);
      }
      audioEl.src = segment.url;

      // Phase 2: streamed segment — wire the MediaSource pump. Chunks that
      // arrived while this segment waited in the queue are appended in
      // order here; later chunks are appended as they arrive (one
      // appendBuffer at a time — MSE requires updateend chaining). Playback
      // starts as soon as the element has decodable data: no wait for the
      // complete sentence.
      const streamState = segment.stream ?? null;
      if (streamState) {
        const mediaSource = streamState.mediaSource;
        const failSegment = (reason: string) => {
          streamState.failed = true;
          streamState.onData = undefined;
          addLog('error', `Streamed audio failed: ${reason}`);
          trace('AUDIO_DROPPED', `segment=${segLabel} reason=${reason}`, segment.turn);
          finishSegment('');
        };
        const pump = () => {
          if (streamState.failed) return;
          if (audioQueueRef.current.current !== segment) return;
          const sb = streamState.sourceBuffer;
          if (!sb || sb.updating) return;
          const chunk = streamState.queue.shift();
          if (chunk) {
            try {
              sb.appendBuffer(chunk);
            } catch {
              failSegment('append_error');
            }
            return;
          }
          if (streamState.ended && mediaSource.readyState === 'open') {
            try {
              mediaSource.endOfStream();
            } catch {
              // Already ended / detached — nothing to finalize.
            }
          }
        };
        streamState.onData = pump;
        mediaSource.addEventListener('sourceopen', () => {
          if (streamState.failed) return;
          try {
            streamState.sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg');
          } catch {
            failSegment('sourcebuffer_error');
            return;
          }
          streamState.sourceBuffer.addEventListener('updateend', pump);
          pump();
        });
        mediaSource.addEventListener('error', () =>
          failSegment('media_source_error'),
        );
      }

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
        if (segment.stream) {
          // Phase 2: the streamed registry entry is done — late messages
          // for this segment are ignored from here on.
          streamedSegmentsRef.current.delete(
            `${segment.turn ?? 'n/a'}:${segment.segmentNo ?? 1}`,
          );
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
        playbackStartedAt = playingAt;
        if (predecessor === null) {
          console.log(
            `[VOICE:UI] audio_playing first_segment_playback_latency_ms=${elapsedMs}`,
          );
          addLog('audio', `playing (first-segment playback latency=${elapsedMs}ms)`);
          trace(
            'AUDIO_PLAYING',
            `segment=${segLabel} mode=elevenlabs first_segment_latency=${elapsedMs}ms`,
            segment.turn,
          );
        } else if (firstOfTurn) {
          // First audio of a new turn that still waited behind the previous
          // turn's tail — honest label, but it owns this turn's report/UI.
          console.log(`[VOICE:UI] audio_playing queue_wait_ms=${elapsedMs}`);
          addLog(
            'audio',
            `playing (first of turn, queued behind ${predecessor}, wait=${elapsedMs}ms)`,
          );
          trace('AUDIO_PLAYING', `segment=${segLabel} mode=elevenlabs`, segment.turn);
          trace(
            'AUDIO_QUEUE_WAIT',
            `segment=${segLabel} queued_behind=${predecessor} wait=${elapsedMs}ms`,
            segment.turn,
          );
        } else {
          console.log(`[VOICE:UI] audio_playing queue_wait_ms=${elapsedMs}`);
          addLog('audio', `playing (queued behind ${predecessor}, wait=${elapsedMs}ms)`);
          trace('AUDIO_PLAYING', `segment=${segLabel} mode=elevenlabs`, segment.turn);
          trace(
            'AUDIO_QUEUE_WAIT',
            `segment=${segLabel} queued_behind=${predecessor} wait=${elapsedMs}ms`,
            segment.turn,
          );
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
      audioEl.onended = () => {
        if (audioQueueRef.current.current === segment) {
          const playedMs =
            playbackStartedAt != null
              ? Math.round(performance.now() - playbackStartedAt)
              : null;
          trace(
            'AUDIO_ENDED',
            `segment=${segLabel}${playedMs != null ? ` played_ms=${playedMs}` : ''}`,
            segment.turn,
          );
          trace(
            'TTS_SEGMENT_COMPLETED',
            `mode=elevenlabs segment=${segLabel}` +
              (playedMs != null ? ` duration=${playedMs}ms` : ''),
            segment.turn,
          );
        }
        finishSegment('segment ended');
      };
      audioEl.onerror = () => {
        addLog('error', 'Audio segment playback failed');
        trace('AUDIO_DROPPED', `segment=${segLabel} reason=playback_error`, segment.turn);
        finishSegment('');
      };
      audioEl.play().catch((e) => {
        console.error('[REALTIME] Audio playback failed:', e);
        addLog('error', `Audio playback blocked: ${e.message}`);
        trace('AUDIO_DROPPED', `segment=${segLabel} reason=play_blocked`, segment.turn);
        finishSegment('');
      });
    },
    [addLog, trace],
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
      trace('TTS_SEGMENT_START', `mode=browser segment=${item.segmentNo}`, item.turn);
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
        const turnMarks = traceTurnRef.current;
        trace('AUDIO_PLAYING', `segment=${item.segmentNo} mode=browser`, item.turn);
        if (!turnMarks.firstAudioSeen) {
          turnMarks.firstAudioSeen = true;
          trace(
            'TTS_FIRST_AUDIO',
            `mode=browser segment=${item.segmentNo} startup_ms=${startupMs}` +
              ` engine_ms=${engineMs}`,
            item.turn,
          );
        }
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
        if (startedAt != null && sessionEpochRef.current === epoch) {
          trace('AUDIO_ENDED', `segment=${item.segmentNo} mode=browser`, item.turn);
          trace(
            'TTS_SEGMENT_COMPLETED',
            `mode=browser segment=${item.segmentNo}` +
              (durationMs != null ? ` duration=${durationMs}ms` : ''),
            item.turn,
          );
        }
        finish(
          `browser_tts_ended turn=${turnLabel} segment=${item.segmentNo}` +
            (durationMs != null ? ` duration=${durationMs}ms` : ''),
        );
      };
      utterance.onerror = (e) => {
        if (sessionEpochRef.current === epoch) {
          trace(
            'TTS_ERROR',
            `mode=browser segment=${item.segmentNo} error=${e.error ?? 'unknown'}`,
            item.turn,
          );
        }
        finish(
          `browser_tts_error turn=${turnLabel} segment=${item.segmentNo}` +
            ` error=${e.error ?? 'unknown'}`,
        );
      };
      window.speechSynthesis.speak(utterance);
    },
    [addLog, trace],
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
          trace(
            'SESSION_START',
            `session=${event.session_id} stt=${event.provider}/${event.model}` +
              ` sample_rate=${event.sample_rate} encoding=${event.encoding}` +
              ` tts_mode=${ttsModeRef.current}` +
              ` llm=${sessionTraceRef.current.llmProvider || 'default'}/${
                sessionTraceRef.current.llmModel || 'default'
              }`,
            null,
          );
          break;
        case 'transcript_partial': {
          // Earliest reliable user-speech signal: a new partial while the
          // assistant is still speaking interrupts its playback.
          maybeBargeIn('transcript_partial', event.text);
          // One evolving current turn: partials replace in place and are
          // never committed as conversation messages.
          dispatchConversation({ type: 'partial', text: event.text });
          addLog('transcript_partial', `"${event.text}"`);
          // Trace: first partial of each speech episode logs immediately,
          // then at most one line per second (skipped counts fold in).
          const marks = speechMarksRef.current;
          const nowMs = performance.now();
          if (marks.startedAt === 0) {
            noteSpeechStarted();
          }
          if (marks.lastPartialLogAt === 0) {
            marks.lastPartialLogAt = nowMs;
            trace('STT_PARTIAL', `text="${clip(event.text, 60)}"`, pendingSpeechTurn());
          } else if (nowMs - marks.lastPartialLogAt >= 1000) {
            const skipped = marks.partialsSkipped;
            marks.partialsSkipped = 0;
            marks.lastPartialLogAt = nowMs;
            trace(
              'STT_PARTIAL',
              `text="${clip(event.text, 60)}"${skipped > 0 ? ` throttled=${skipped}` : ''}`,
              pendingSpeechTurn(),
            );
          } else {
            marks.partialsSkipped += 1;
          }
          break;
        }
        case 'transcript_final': {
          // Backup trigger (partials usually fired already — idempotent).
          maybeBargeIn('transcript_final', event.text);
          // Finalized segments accumulate into the same turn; the turn is
          // committed only when the backend dispatches it (agent_processing).
          dispatchConversation({ type: 'final', text: event.text });
          addLog('transcript_final', `"${event.text}"`);
          const marks = speechMarksRef.current;
          if (marks.startedAt === 0) noteSpeechStarted();
          marks.finals += 1;
          trace(
            'STT_FINAL',
            `text="${clip(event.text, 60)}" since_speech_start=${since(marks.startedAt)}`,
            pendingSpeechTurn(),
          );
          if (event.speech_final) {
            marks.speechFinalSeen = true;
            trace('SPEECH_FINAL', `finals=${marks.finals}`, pendingSpeechTurn());
          }
          break;
        }
        case 'utterance_end':
          addLog('utterance_end', '');
          trace(
            'UTTERANCE_END',
            `finals=${speechMarksRef.current.finals}`,
            pendingSpeechTurn(),
          );
          break;
        case 'agent_processing': {
          // The turn was released by the backend: commit ONE user message.
          // Track the in-flight server turn (turn numbers come from
          // turn_metrics) so interrupting generation can cut its audio.
          activeTurnRef.current = lastMetricsTurnRef.current + 1;
          const releasedTurn = activeTurnRef.current;
          dispatchConversation({ type: 'agent_processing' });
          addLog('agent_processing', '');
          // Trace: the release reason is derived from this speech episode's
          // observed marks; the per-turn marks are re-armed so every since_*
          // reference belongs to this turn.
          const speechMarks = speechMarksRef.current;
          trace(
            'TURN_RELEASED',
            `reason=${speechMarks.speechFinalSeen ? 'speech_final' : 'utterance_end'}` +
              ` finals=${speechMarks.finals}`,
            releasedTurn,
          );
          trace('TURN_CREATED', `turn=${releasedTurn}`, releasedTurn);
          trace('AGENT_PROCESSING', `turn=${releasedTurn}`, releasedTurn);
          speechMarks.startedAt = 0;
          speechMarks.speechFinalSeen = false;
          speechMarks.finals = 0;
          const turnMarks = traceTurnRef.current;
          turnMarks.turn = releasedTurn;
          turnMarks.startedAt = performance.now();
          turnMarks.llmRequestAt = performance.now();
          turnMarks.firstTokenSeen = false;
          turnMarks.firstSentenceSeen = false;
          turnMarks.firstAudioSeen = false;
          turnMarks.responseStarted = false;
          turnMarks.ackedTools.clear();
          trace(
            'LLM_REQUEST',
            `provider=${sessionTraceRef.current.llmProvider || 'default'}` +
              ` model=${sessionTraceRef.current.llmModel || 'default'}`,
            releasedTurn,
          );
          break;
        }
        case 'agent_response': {
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
          // Trace: buffered mode has no agent_delta stream — the response
          // start is only observable here. Deltas never appear in the trace.
          const turnMarks = traceTurnRef.current;
          const turnNo = turnMarks.turn || null;
          if (!turnMarks.responseStarted) {
            turnMarks.responseStarted = true;
            trace(
              'AGENT_RESPONSE_STARTED',
              `mode=buffered since_llm_request=${since(turnMarks.llmRequestAt)}`,
              turnNo,
            );
          }
          trace(
            'AGENT_RESPONSE_COMPLETED',
            `chars=${event.text.length} tools=${event.tool_calls} iterations=${event.iterations}`,
            turnNo,
          );
          trace(
            'LLM_COMPLETED',
            `duration=${since(turnMarks.llmRequestAt)} tools=${event.tool_calls}` +
              ` iterations=${event.iterations}`,
            turnNo,
          );
          break;
        }
        case 'agent_delta': {
          // Progressive assistant text: accumulate the raw LLM delta into
          // the evolving assistant message. agent_response later commits
          // the authoritative full text (replaces the draft). Deliberately
          // not addLog()'d — per-token events would flood the dev log.
          dispatchConversation({ type: 'agent_delta', text: event.text });
          // Trace: exactly two marks per turn — first token and response
          // start. Individual deltas are never traced.
          const turnMarks = traceTurnRef.current;
          if (!turnMarks.firstTokenSeen) {
            turnMarks.firstTokenSeen = true;
            const turnNo = turnMarks.turn || null;
            trace(
              'LLM_FIRST_TOKEN',
              `since_turn_start=${since(turnMarks.startedAt)}` +
                ` since_llm_request=${since(turnMarks.llmRequestAt)}`,
              turnNo,
            );
            if (!turnMarks.responseStarted) {
              turnMarks.responseStarted = true;
              trace('AGENT_RESPONSE_STARTED', 'mode=streaming', turnNo);
            }
          }
          break;
        }
        case 'tool_progress': {
          // Deterministic ack sent right before a tool executes — shown as
          // a temporary status message. The backend also queues the same
          // text into the TTS sentence sink, so it is spoken while the tool
          // runs. agent_response later clears the status.
          dispatchConversation({ type: 'tool_progress', message: event.message });
          addLog('tool_progress', `${event.tool_name}: ${event.message}`);
          // Trace: one TOOL_PROGRESS + TOOL_CALL per tool_call_id; a repeated
          // id means the backend answered a duplicate ack — keep it visible
          // as a suppressed-duplicate line instead of hiding it.
          const turnMarks = traceTurnRef.current;
          const callLabel = event.tool_call_id ?? 'n/a';
          const key = event.tool_call_id ?? event.tool_name;
          const turnNo = turnMarks.turn || null;
          if (turnMarks.ackedTools.has(key)) {
            trace(
              'TOOL_PROGRESS_DUPLICATE_SUPPRESSED',
              `tool=${event.tool_name} call_id=${callLabel}`,
              turnNo,
            );
          } else {
            turnMarks.ackedTools.add(key);
            trace('TOOL_PROGRESS', `tool=${event.tool_name} call_id=${callLabel}`, turnNo);
            trace('TOOL_CALL', `tool=${event.tool_name} call_id=${callLabel}`, turnNo);
          }
          break;
        }
        case 'tool_result': {
          const turnMarks = traceTurnRef.current;
          const callLabel = event.tool_call_id ?? 'n/a';
          const line =
            `tool=${event.tool_name} call_id=${callLabel}` +
            ` duration=${event.duration_ms}ms success=${event.success}` +
            (event.error ? ` error="${clip(event.error, 60)}"` : '');
          if (event.success) {
            trace('TOOL_RESULT', line, turnMarks.turn || null);
          } else {
            trace('TOOL_ERROR', line, turnMarks.turn || null);
          }
          break;
        }
        case 'tts_processing': {
          setTtsProcessing(true);
          addLog('tts_processing', '');
          // Trace: the first-sentence mark and the start of the TTS stage
          // (tts_processing is sent once per turn, with the first sentence).
          const turnMarks = traceTurnRef.current;
          const turnNo = turnMarks.turn || null;
          if (!turnMarks.firstSentenceSeen) {
            turnMarks.firstSentenceSeen = true;
            trace(
              'LLM_FIRST_SENTENCE',
              `since_turn_start=${since(turnMarks.startedAt)}` +
                ` since_llm_request=${since(turnMarks.llmRequestAt)}`,
              turnNo,
            );
            trace('TTS_START', `mode=${ttsModeRef.current}`, turnNo);
          }
          break;
        }
        case 'tts_text': {
          // Phase 6L browser TTS: queue the sentence text; speechSynthesis
          // speaks strictly in sentence order (one at a time). No audio
          // events exist in this mode.
          const turnNo = typeof event.turn === 'number' ? event.turn : null;
          // Barge-in: discard the interrupted turn's late sentences; the
          // first frame of a newer turn lifts the gate.
          if (bargeInCutoffRef.current !== null) {
            if (turnNo === null || turnNo <= bargeInCutoffRef.current) {
              addLog(
                'tts_text',
                `barge_in drop turn=${event.turn ?? 'n/a'} cutoff=${bargeInCutoffRef.current}`,
              );
              trace(
                'STALE_AUDIO_DROPPED',
                `mode=browser segment=${event.segment ?? 'n/a'}` +
                  ` turn=${event.turn ?? 'n/a'} cutoff=${bargeInCutoffRef.current}`,
                bargeInCutoffRef.current,
              );
              break;
            }
            addLog('tts_text', `barge_in resume turn=${turnNo}`);
            trace('BARGE_IN_RESUMED', `turn=${turnNo} mode=browser`, turnNo);
            bargeInCutoffRef.current = null;
          }
          latestMediaTurnRef.current = Math.max(
            latestMediaTurnRef.current,
            turnNo ?? 0,
          );
          const segmentNo = typeof event.segment === 'number' ? event.segment : 1;
          addLog('tts_text', `turn=${turnNo ?? 'n/a'} segment=${segmentNo}`);
          browserTtsQueueRef.current.push({
            turn: turnNo,
            segmentNo,
            text: event.text,
            receivedAt: performance.now(),
          });
          trace(
            'AUDIO_RECEIVED',
            `segment=${segmentNo} mode=browser chars=${event.text.length}`,
            turnNo,
          );
          trace(
            'AUDIO_ENQUEUED',
            `segment=${segmentNo} mode=browser queue_pos=${browserTtsQueueRef.current.length}`,
            turnNo,
          );
          speakNextBrowserTts(sessionEpochRef.current);
          break;
        }
        case 'audio': {
          setTtsProcessing(false);
          // Phase 6G: capture the backend turn + server send timestamp so the
          // playback-start report below can be correlated server-side.
          const audioTurn = typeof event.turn === 'number' ? event.turn : null;
          // Barge-in: discard the interrupted turn's late segments before
          // decoding (no blob/object URL is created, so nothing leaks); the
          // first segment of a newer turn lifts the gate.
          if (bargeInCutoffRef.current !== null) {
            if (audioTurn === null || audioTurn <= bargeInCutoffRef.current) {
              addLog(
                'audio',
                `barge_in drop turn=${event.turn ?? 'n/a'} cutoff=${bargeInCutoffRef.current}`,
              );
              trace(
                'STALE_AUDIO_DROPPED',
                `mode=elevenlabs segment=${event.segment ?? 'n/a'}` +
                  ` turn=${event.turn ?? 'n/a'} cutoff=${bargeInCutoffRef.current}`,
                bargeInCutoffRef.current,
              );
              break;
            }
            addLog('audio', `barge_in resume turn=${audioTurn}`);
            trace('BARGE_IN_RESUMED', `turn=${audioTurn} mode=elevenlabs`, audioTurn);
            bargeInCutoffRef.current = null;
          }
          latestMediaTurnRef.current = Math.max(
            latestMediaTurnRef.current,
            audioTurn ?? 0,
          );
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
          trace(
            'AUDIO_RECEIVED',
            `segment=${segmentNo} mode=elevenlabs format=${event.format}` +
              ` size=${bytes.length}B transit=${wsTransitMs ?? 'n/a'}ms`,
            audioTurn,
          );
          const turnMarks = traceTurnRef.current;
          if (!turnMarks.firstAudioSeen) {
            turnMarks.firstAudioSeen = true;
            trace(
              'TTS_FIRST_AUDIO',
              `segment=${segmentNo} mode=elevenlabs` +
                ` since_turn_start=${since(turnMarks.startedAt)}` +
                ` since_llm_request=${since(turnMarks.llmRequestAt)}`,
              audioTurn,
            );
          }
          trace(
            'AUDIO_ENQUEUED',
            `segment=${segmentNo} mode=elevenlabs queue_pos=${audioQueueRef.current.pending.length}`,
            audioTurn,
          );
          playNextSegment();
          break;
        }
        case 'audio_chunk': {
          // Phase 2: incremental MP3 delivery. The first chunk of a segment
          // creates the queue item (MediaSource-backed); later chunks land
          // in the same item. Barge-in drops stale chunks like 'audio'.
          setTtsProcessing(false);
          const chunkTurn = typeof event.turn === 'number' ? event.turn : null;
          if (bargeInCutoffRef.current !== null) {
            if (chunkTurn === null || chunkTurn <= bargeInCutoffRef.current) {
              addLog(
                'audio',
                `barge_in drop chunk turn=${event.turn ?? 'n/a'} cutoff=${bargeInCutoffRef.current}`,
              );
              trace(
                'STALE_AUDIO_DROPPED',
                `mode=elevenlabs stream=true segment=${event.segment ?? 'n/a'}` +
                  ` turn=${event.turn ?? 'n/a'} cutoff=${bargeInCutoffRef.current}`,
                bargeInCutoffRef.current,
              );
              break;
            }
            addLog('audio', `barge_in resume turn=${chunkTurn}`);
            trace('BARGE_IN_RESUMED', `turn=${chunkTurn} mode=elevenlabs`, chunkTurn);
            bargeInCutoffRef.current = null;
          }
          latestMediaTurnRef.current = Math.max(
            latestMediaTurnRef.current,
            chunkTurn ?? 0,
          );
          const chunkSegmentNo =
            typeof event.segment === 'number' ? event.segment : 1;
          const streamKey = `${chunkTurn ?? 'n/a'}:${chunkSegmentNo}`;
          const chunkBytes = decodeBase64Audio(event.data);
          const existing = streamedSegmentsRef.current.get(streamKey) ?? null;
          if (!existing) {
            const mediaSource = new MediaSource();
            const url = URL.createObjectURL(mediaSource);
            const wsTransitMs =
              typeof event.sent_epoch_ms === 'number'
                ? Math.max(0, Math.round(Date.now() - event.sent_epoch_ms))
                : null;
            if (!audioRef.current) {
              audioRef.current = new Audio();
            }
            const streamedItem: AudioSegment = {
              url,
              turn: chunkTurn,
              segmentNo: chunkSegmentNo,
              wsTransitMs,
              receivedAt: performance.now(),
              stream: {
                mediaSource,
                sourceBuffer: null,
                queue: [chunkBytes],
                ended: false,
                failed: false,
              },
            };
            streamedSegmentsRef.current.set(streamKey, streamedItem);
            audioQueueRef.current = enqueueSegment(
              audioQueueRef.current,
              streamedItem,
            );
            addLog(
              'audio',
              `segment=${chunkSegmentNo} stream seq=${event.seq ?? 0} chunk=${chunkBytes.byteLength}B`,
            );
            trace(
              'AUDIO_RECEIVED',
              `segment=${chunkSegmentNo} mode=elevenlabs stream=true` +
                ` transit=${wsTransitMs ?? 'n/a'}ms`,
              chunkTurn,
            );
            const turnMarks = traceTurnRef.current;
            if (!turnMarks.firstAudioSeen) {
              turnMarks.firstAudioSeen = true;
              trace(
                'TTS_FIRST_AUDIO',
                `segment=${chunkSegmentNo} mode=elevenlabs stream=true` +
                  ` since_turn_start=${since(turnMarks.startedAt)}` +
                  ` since_llm_request=${since(turnMarks.llmRequestAt)}`,
                chunkTurn,
              );
            }
            trace(
              'AUDIO_ENQUEUED',
              `segment=${chunkSegmentNo} mode=elevenlabs stream=true queue_pos=${audioQueueRef.current.pending.length}`,
              chunkTurn,
            );
            playNextSegment();
          } else {
            existing.stream!.queue.push(chunkBytes);
            existing.stream!.onData?.();
          }
          break;
        }
        case 'audio_end': {
          // Phase 2: the segment is fully delivered — finalize its
          // MediaSource once every queued chunk has been appended.
          const endTurn = typeof event.turn === 'number' ? event.turn : null;
          const endSegmentNo =
            typeof event.segment === 'number' ? event.segment : 1;
          const streamedItem =
            streamedSegmentsRef.current.get(`${endTurn ?? 'n/a'}:${endSegmentNo}`) ??
            null;
          if (streamedItem?.stream) {
            streamedItem.stream.ended = true;
            streamedItem.stream.onData?.();
            addLog(
              'audio',
              `segment=${endSegmentNo} stream end seq=${event.seq ?? 'n/a'}`,
            );
          }
          break;
        }
        case 'turn_metrics': {
          setTurnMetrics(event.data);
          const d = event.data;
          // Turn fully produced (metrics arrive after its TTS consumer
          // completes): retire the in-flight marker, remember the number.
          lastMetricsTurnRef.current = d.turn;
          latestMediaTurnRef.current = Math.max(latestMediaTurnRef.current, d.turn);
          activeTurnRef.current = 0;
          addLog(
            'turn_metrics',
            `turn=${d.turn} utterance_to_agent=${d.utterance_end_to_agent_ms ?? 'N/A'}ms ` +
              `agent=${d.agent_processing_ms ?? 'N/A'}ms tts=${d.tts_duration_ms ?? 'N/A'}ms ` +
              `total=${d.utterance_end_to_audio_sent_ms ?? 'N/A'}ms`,
          );
          // Trace: the turn's TTS stage is complete once its metrics arrive;
          // every value below is the server's authoritative measurement.
          trace('TTS_COMPLETED', `duration=${d.tts_duration_ms ?? 'N/A'}ms`, d.turn);
          trace(
            'TURN_METRICS',
            `release=${d.utterance_end_to_release_ms ?? 'N/A'}ms` +
              ` agent=${d.agent_processing_ms ?? 'N/A'}ms` +
              ` llm_first_token=${d.llm_request_to_first_token_ms ?? 'N/A'}ms` +
              ` llm_first_sentence=${d.llm_request_to_first_sentence_ms ?? 'N/A'}ms` +
              ` llm_complete=${d.llm_request_to_complete_ms ?? 'N/A'}ms` +
              ` tts=${d.tts_duration_ms ?? 'N/A'}ms` +
              ` total=${d.utterance_end_to_audio_sent_ms ?? 'N/A'}ms`,
            d.turn,
          );
          break;
        }
        case 'completed':
          setTimings(event.timings);
          addLog(
            'completed',
            `chunks=${event.timings.audio_chunks} bytes=${event.timings.audio_bytes} agent_responses=${event.timings.agent_response_count}`,
          );
          trace(
            'SESSION_STOP',
            `reason=completed agent_responses=${event.timings.agent_response_count}`,
            null,
          );
          // Graceful shutdown — clear force-close timer and disconnect
          if (stopTimeoutRef.current) {
            clearTimeout(stopTimeoutRef.current);
            stopTimeoutRef.current = null;
          }
          teardownAudio();
          wsRef.current?.close();
          break;
        case 'error': {
          activeTurnRef.current = 0;
          dispatchConversation({ type: 'agent_failed' });
          setTtsProcessing(false);
          setError(event.message);
          addLog('error', `${event.stage ? `[${event.stage}] ` : ''}${event.message}`);
          // Trace: session-level failure plus the stage-specific line. The
          // backend reports stages 'stt' | 'agent' | 'tts'.
          const turnNo = traceTurnRef.current.turn || null;
          trace(
            'SESSION_ERROR',
            `stage=${event.stage ?? 'protocol'} message="${clip(event.message, 80)}"`,
            turnNo,
          );
          if (event.stage === 'agent') {
            trace('AGENT_FAILED', `message="${clip(event.message, 80)}"`, turnNo);
          } else if (event.stage === 'tts') {
            trace(
              'TTS_ERROR',
              `mode=${ttsModeRef.current} message="${clip(event.message, 80)}"`,
              turnNo,
            );
          }
          teardownAudio();
          break;
        }
      }
    },
    [
      addLog,
      teardownAudio,
      playNextSegment,
      speakNextBrowserTts,
      maybeBargeIn,
      trace,
      pendingSpeechTurn,
      noteSpeechStarted,
    ],
  );

  const start = useCallback(
    async (config: RealtimeStartConfig) => {
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
      // Barge-in gate + turn tracking: a fresh session starts clean.
      bargeInCutoffRef.current = null;
      latestMediaTurnRef.current = 0;
      lastMetricsTurnRef.current = 0;
      activeTurnRef.current = 0;
      // Trace state: fresh episode/turn marks and session metadata.
      lastTraceAtRef.current = 0;
      speechMarksRef.current.startedAt = 0;
      speechMarksRef.current.speechFinalSeen = false;
      speechMarksRef.current.finals = 0;
      const traceTurn = traceTurnRef.current;
      traceTurn.turn = 0;
      traceTurn.startedAt = 0;
      traceTurn.llmRequestAt = 0;
      traceTurn.firstTokenSeen = false;
      traceTurn.firstSentenceSeen = false;
      traceTurn.firstAudioSeen = false;
      traceTurn.responseStarted = false;
      traceTurn.ackedTools.clear();
      sessionTraceRef.current = {
        llmProvider: config.llmProvider ?? '',
        llmModel: config.llmModel ?? '',
      };
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
      closingRef.current = false;

      // Phase 6M: warm the speech engine once, inside this Start user gesture,
      // so Chrome's one-time engine initialization (~0.5-1s before onstart)
      // does not land on the first real sentence of a browser-TTS session.
      // Silent: a single space at volume 0, never queued, never shown as
      // assistant text, at most once per hook mount.
      if (ttsModeState === 'browser' && !browserTtsWarmupDoneRef.current) {
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
          if (config.llmProvider) {
            startMsg.llm_provider = config.llmProvider;
          }
          if (config.llmModel) {
            startMsg.llm_model = config.llmModel;
          }
          // Phase 6L: response TTS mode — elevenlabs (server MP3) or browser
          // (tts_text events + local speechSynthesis).
          startMsg.tts_mode = ttsModeState;
          // Phase 2: advertise incremental MP3 delivery support — only then
          // does the backend stream audio_chunk/audio_end for this session.
          startMsg.audio_stream = AUDIO_STREAM_SUPPORTED;
          ws.send(JSON.stringify(startMsg));
          addLog(
            'ws',
            `START sent sample_rate=${audioCtx.sampleRate} llm=${config.llmProvider}/${config.llmModel || 'default'} tts=${ttsModeState} audio_stream=${AUDIO_STREAM_SUPPORTED}`,
          );
          trace(
            'WS_CONNECTED',
            `url=${REALTIME_WS_URL.replace(/^wss?:\/\//, '')}` +
              ` rate=${audioCtx.sampleRate} tts_mode=${ttsModeState}`,
            null,
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
          trace(
            'MIC_STARTED',
            `rate=${audioCtx.sampleRate} channels=1 frame_ms=50`,
            null,
          );
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
          trace('SESSION_ERROR', 'stage=ws message="WebSocket connection error"', null);
        };

        ws.onclose = () => {
          if (sessionEpochRef.current !== epoch) {
            // Socket of a previous session — never touch current state.
            return;
          }
          setConnState('disconnected');
          trace('WS_DISCONNECTED', `expected=${closingRef.current}`, null);
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
        trace('SESSION_ERROR', `stage=start message="${clip(msg, 80)}"`, null);
      }
    },
    [addLog, connState, handleServerEvent, teardownAudio, ttsModeState, trace],
  );

  const stop = useCallback(() => {
    closingRef.current = true;
    addLog('ws', 'STOP sent');
    trace('SESSION_STOP', 'reason=user_stop', null);

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
  }, [addLog, teardownAudio, trace]);

  /** Phase 6L: switch the realtime response TTS mode.
   *
   * Browser speech is cancelled and its queue cleared on EVERY switch (no
   * browser speech outlives the mode it belongs to). The backend applies
   * the new mode from the NEXT turn — an already-playing ElevenLabs response
   * is never interrupted.
   */
  const setTtsMode = useCallback(
    (mode: RealtimeTtsMode) => {
      if (mode === ttsModeState) return;
      if (mode === 'browser' && !BROWSER_TTS_SUPPORTED) {
        setError('Browser speech synthesis is not available in this browser');
        addLog('error', 'Browser TTS unavailable (no speechSynthesis)');
        return;
      }
      setTtsModeState(mode);
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
    [addLog, ttsModeState],
  );

  const currentTurn = currentTurnText(conversation);

  return {
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
    ttsMode: ttsModeState,
    browserTtsSupported: BROWSER_TTS_SUPPORTED,
    start,
    stop,
    setTtsMode,
  };
}
