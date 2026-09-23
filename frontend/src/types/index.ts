/** API and domain types for the Voice AI Lab frontend. */

// --- Provider Types ---

export type ProviderType = 'stt' | 'llm' | 'tts';

export interface ProviderConfig {
  provider_type: ProviderType;
  provider_name: string;
  display_name: string;
  is_active: boolean;
}

// --- Session Types ---

export interface SessionConfig {
  system_prompt: string;
  stt_provider?: string;
  stt_model?: string;
  llm_provider?: string;
  llm_model?: string;
  tts_provider?: string;
  tts_model?: string;
  tts_voice?: string;
}

export interface VoiceSession {
  id: string;
  status: string;
  system_prompt?: string;
  stt_provider?: string;
  stt_model?: string;
  llm_provider?: string;
  llm_model?: string;
  tts_provider?: string;
  tts_model?: string;
  tts_voice?: string;
  duration_seconds?: number;
  total_cost?: number;
  message_count: number;
  created_at: string;
  ended_at?: string;
}

// --- Chat Types ---

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  tool_calls?: ToolCallInfo[];
}

export interface ToolCallInfo {
  id: string;
  function: {
    name: string;
    arguments: string;
  };
}

export interface ChatRequest {
  message: string;
  session_id?: string | null;
  provider?: string | null;
  model?: string | null;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  tool_calls: ToolCallInfo[];
  usage: Record<string, number>;
  iterations: number;
  latency_ms: number;
}

// --- Provider Availability ---

export interface ProviderAvailability {
  stt: ProviderOption[];
  llm: ProviderOption[];
  tts: ProviderOption[];
}

export interface ProviderOption {
  provider: string;
  models: string[];
  /** Model IDs that are paid/PAYG; the rest of `models` are free/experimental. */
  paid_models?: string[];
  configured: boolean;
  default_model?: string;
  default_voice?: string;
}

// --- Tool Types ---

export interface ToolInfo {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  enabled: boolean;
}

// --- Health ---

export interface HealthStatus {
  status: string;
  version: string;
  timestamp: string;
}

// --- Voice WebSocket Events ---

export type VoiceEventType =
  | 'session_started'
  | 'processing'
  | 'transcript'
  | 'agent_response'
  | 'tool_call'
  | 'tool_result'
  | 'audio'
  | 'metrics'
  | 'completed'
  | 'error';

export interface VoiceEventBase {
  type: VoiceEventType;
}

export interface VoiceSessionStarted extends VoiceEventBase {
  type: 'session_started';
  session_id: string;
}

export interface VoiceProcessing extends VoiceEventBase {
  type: 'processing';
}

export interface VoiceTranscript extends VoiceEventBase {
  type: 'transcript';
  text: string;
  final: boolean;
}

export interface VoiceAgentResponse extends VoiceEventBase {
  type: 'agent_response';
  text: string;
}

export interface VoiceToolCall extends VoiceEventBase {
  type: 'tool_call';
  name: string;
  arguments: Record<string, unknown>;
}

export interface VoiceAudio extends VoiceEventBase {
  type: 'audio';
  format: string;
  data: string; // base64
}

export interface VoiceCompleted extends VoiceEventBase {
  type: 'completed';
}

/** Server-side benchmark metrics for one voice turn (Phase 5A). */
export interface VoiceTurnMetrics {
  turn_id: string;
  session_id: string | null;
  stt_provider: string | null;
  stt_model: string | null;
  stt_latency_ms: number | null;
  stt_audio_bytes: number | null;
  stt_audio_duration_seconds: number | null;
  transcript_length: number | null;
  llm_provider: string | null;
  llm_model: string | null;
  llm_latency_ms: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  llm_iterations: number | null;
  tool_count: number;
  tool_success_count: number;
  tool_execution_ms: number | null;
  tts_provider: string | null;
  tts_model: string | null;
  tts_voice: string | null;
  tts_output_format: string | null;
  tts_latency_ms: number | null;
  tts_audio_bytes: number | null;
  tts_characters: number | null;
  total_processing_ms: number | null;
  success: boolean;
  error_stage: string | null;
  error_message: string | null;
}

export interface VoiceMetrics extends VoiceEventBase {
  type: 'metrics';
  data: VoiceTurnMetrics;
}

/** Client-side timings measured with performance.now(). */
export interface BrowserTurnTimings {
  recording_duration_ms: number | null;
  audio_upload_ms: number | null;
  server_processing_ms: number | null;
  audio_receive_ms: number | null;
  playback_start_latency_ms: number | null;
  total_turn_duration_ms: number | null;
}

export interface VoiceError extends VoiceEventBase {
  type: 'error';
  message: string;
}

export type VoiceEvent =
  | VoiceSessionStarted
  | VoiceProcessing
  | VoiceTranscript
  | VoiceAgentResponse
  | VoiceToolCall
  | VoiceAudio
  | VoiceMetrics
  | VoiceCompleted
  | VoiceError;

export type RecordingState = 'idle' | 'recording' | 'processing' | 'playing';
export type ConnectionState = 'disconnected' | 'connecting' | 'connected';

// --- Realtime Voice Events (Phase 6B) ---
// Protocol for WS /api/v1/voice/realtime/ws (streaming STT + AgentRuntime).

export type RealtimeEventType =
  | 'session_started'
  | 'transcript_partial'
  | 'transcript_final'
  | 'utterance_end'
  | 'agent_processing'
  | 'agent_response'
  | 'tts_processing'
  | 'audio'
  | 'completed'
  | 'error';

export interface RealtimeSessionStarted {
  type: 'session_started';
  session_id: string;
  provider: string;
  model: string;
  sample_rate: number;
  encoding: string;
}

export interface RealtimeTranscriptPartial {
  type: 'transcript_partial';
  text: string;
  confidence?: number;
}

export interface RealtimeTranscriptFinal {
  type: 'transcript_final';
  text: string;
  confidence?: number;
}

export interface RealtimeUtteranceEnd {
  type: 'utterance_end';
}

export interface RealtimeAgentProcessing {
  type: 'agent_processing';
}

export interface RealtimeAgentResponse {
  type: 'agent_response';
  text: string;
  tool_calls: number;
  iterations: number;
}

export interface RealtimeTTSProcessing {
  type: 'tts_processing';
}

export interface RealtimeAudio {
  type: 'audio';
  format: string;
  data: string; // base64
}

/** Server-side diagnostic timings for one realtime session. */
export interface RealtimeTimings {
  session_start_ms: number | null;
  first_audio_ms: number | null;
  first_partial_ms: number | null;
  first_final_ms: number | null;
  first_utterance_end_ms: number | null;
  session_duration_ms: number | null;
  audio_bytes: number;
  audio_chunks: number;
  partial_count: number;
  final_count: number;
  utterance_end_count: number;
  agent_response_count: number;
}

export interface RealtimeCompleted {
  type: 'completed';
  timings: RealtimeTimings;
}

export interface RealtimeError {
  type: 'error';
  message: string;
  /** Pipeline stage that failed (e.g. 'stt', 'agent'). Absent for protocol errors. */
  stage?: string;
}

export type RealtimeEvent =
  | RealtimeSessionStarted
  | RealtimeTranscriptPartial
  | RealtimeTranscriptFinal
  | RealtimeUtteranceEnd
  | RealtimeAgentProcessing
  | RealtimeAgentResponse
  | RealtimeTTSProcessing
  | RealtimeAudio
  | RealtimeCompleted
  | RealtimeError;

/** START message sent by the client (then binary PCM chunks, then stop). */
export interface RealtimeStartMessage {
  type: 'start';
  sample_rate: number;
  channels: number;
  encoding: 'linear16';
  language: string;
  model?: string;
  /** LLM provider override (optional). */
  llm_provider?: string;
  /** LLM model override (optional). */
  llm_model?: string;
}

export type RealtimeConnectionState = 'disconnected' | 'connecting' | 'connected';

/** One entry in the realtime event log UI. */
export interface RealtimeLogEntry {
  time: string;
  type: RealtimeEventType | 'mic' | 'ws';
  detail: string;
}

// --- Benchmark Types (Phase 5B) ---

export interface BenchmarkScenario {
  scenario_id: string;
  name: string;
  description: string;
  category: string;
  expected_tool_calls: number;
  include_tts: boolean;
}

export interface BenchmarkRunRequest {
  scenario_id: string;
  repetitions?: number;
  configuration_id?: string;
}

export interface BenchmarkRunResult {
  run_id: string;
  scenario_id: string;
  success: boolean;
  benchmark_mode: string;
  response_text: string;
  tool_calls: Array<{ function: { name: string; arguments: string } }>;
  usage: Record<string, number>;
  iterations: number;
  llm_latency_ms: number | null;
  tts_latency_ms: number | null;
  total_processing_ms: number | null;
  tool_execution_ms: number | null;
  tts_audio_bytes: number | null;
  tts_characters: number | null;
  expected_tool_calls: number;
  actual_tool_calls: number;
  tool_call_match: boolean;
  validation_errors: string[];
  llm_provider: string | null;
  llm_model: string | null;
  tts_provider: string | null;
  tts_model: string | null;
  benchmark_result_id: string | null;
  configuration_id: string | null;
}

export interface BenchmarkAggregation {
  scenario_id: string;
  count: number;
  success_count: number;
  failure_count: number;
  success_rate: number;
  llm_latency_ms: { avg: number | null; median: number | null; min: number | null; max: number | null };
  tts_latency_ms: { avg: number | null; median: number | null; min: number | null; max: number | null };
  total_processing_ms: { avg: number | null; median: number | null; min: number | null; max: number | null };
}

export interface BenchmarkBatchResult {
  runs: BenchmarkRunResult[];
  aggregation: BenchmarkAggregation | null;
}

// --- Benchmark Analytics Types (Phase 5C) ---

export interface LatencyStats {
  avg_ms: number | null;
  median_ms: number | null;
  min_ms: number | null;
  max_ms: number | null;
}

export interface BenchmarkOverallSummary {
  total_runs: number;
  successful_runs: number;
  failed_runs: number;
  success_rate: number;
  latency: LatencyStats;
  stt_latency: LatencyStats;
  llm_latency: LatencyStats;
  tts_latency: LatencyStats;
  tool_execution: LatencyStats;
  avg_prompt_tokens: number | null;
  avg_completion_tokens: number | null;
  avg_total_tokens: number | null;
  avg_tts_characters: number | null;
  avg_tts_audio_bytes: number | null;
}

export interface BenchmarkScenarioSummary {
  scenario_id: string;
  run_count: number;
  successful_runs: number;
  failed_runs: number;
  success_rate: number;
  latency: LatencyStats;
  stt_latency: LatencyStats;
  llm_latency: LatencyStats;
  tts_latency: LatencyStats;
  tool_execution: LatencyStats;
  avg_prompt_tokens: number | null;
  avg_completion_tokens: number | null;
  avg_total_tokens: number | null;
  avg_tts_characters: number | null;
}

export interface BenchmarkProviderSummary {
  provider: string | null;
  model: string | null;
  stage: string;
  run_count: number;
  successful_runs: number;
  success_rate: number;
  latency: LatencyStats;
  avg_prompt_tokens: number | null;
  avg_completion_tokens: number | null;
  avg_total_tokens: number | null;
  avg_tts_characters: number | null;
  avg_tts_audio_bytes: number | null;
}

export interface BenchmarkRecentResult {
  id: string;
  run_id: string | null;
  scenario_id: string | null;
  benchmark_mode: string | null;
  success: boolean;
  created_at: string;
  total_processing_ms: number | null;
  stt_latency_ms: number | null;
  llm_latency_ms: number | null;
  tts_latency_ms: number | null;
  tool_execution_ms: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  token_usage: number | null;
  tts_characters: number | null;
  tts_audio_bytes: number | null;
  llm_provider: string | null;
  llm_model: string | null;
  tts_provider: string | null;
  tts_model: string | null;
  total_cost: number | null;
  error_message: string | null;
  configuration_id: string | null;
}

// --- Benchmark Cost Types (Phase 5D) ---

export interface BenchmarkCostBreakdown {
  run_id: string | null;
  scenario_id: string | null;
  benchmark_mode: string | null;
  stt_cost: number | null;
  llm_input_cost: number | null;
  llm_output_cost: number | null;
  llm_total_cost: number | null;
  tts_cost: number | null;
  total_cost: number | null;
  currency: string;
  pricing_version: string;
  pricing_available: boolean;
  stt_pricing_source: string | null;
  llm_pricing_source: string | null;
  tts_pricing_source: string | null;
  stt_audio_duration_seconds: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  tts_characters: number | null;
}

export interface BenchmarkCostSummary {
  total_runs: number;
  runs_with_cost: number;
  total_cost: number | null;
  avg_cost: number | null;
  min_cost: number | null;
  max_cost: number | null;
  currency: string;
  pricing_version: string;
}

// --- Benchmark Comparison Types (Phase 5E) ---

export interface BenchmarkConfiguration {
  configuration_id: string;
  name: string;
  description: string;
  llm_provider: string;
  llm_model: string;
  tts_provider: string;
  tts_model: string;
  pricing_type: string;
  production_eligible: boolean;
}

export interface ComparisonConfigurationResult {
  configuration_id: string;
  configuration_name: string;
  pricing_type: string;
  production_eligible: boolean;
  run_count: number;
  successful_runs: number;
  failed_runs: number;
  success_rate: number;
  avg_total_latency_ms: number | null;
  median_total_latency_ms: number | null;
  min_total_latency_ms: number | null;
  max_total_latency_ms: number | null;
  avg_llm_latency_ms: number | null;
  avg_tts_latency_ms: number | null;
  avg_prompt_tokens: number | null;
  avg_completion_tokens: number | null;
  avg_total_tokens: number | null;
  avg_tts_characters: number | null;
  avg_cost: number | null;
  total_cost: number | null;
  cost_available: boolean;
  cost_note: string | null;
  runs: Array<Record<string, unknown>>;
}

export interface BenchmarkComparisonResult {
  comparison_id: string;
  scenario_id: string;
  repetitions: number;
  configurations: ComparisonConfigurationResult[];
  validation_errors: string[];
}

// --- Telephony Observability Types (Phase 8A) ---

export type TelephonyEventType =
  | 'call_received'
  | 'call_answered'
  | 'greeting_completed'
  | 'media_connected'
  | 'stt_connected'
  | 'caller_speech_final'
  | 'caller_transcript'
  | 'agent_processing'
  | 'agent_response'
  | 'tts_processing'
  | 'tts_first_audio'
  | 'tts_completed'
  | 'audio_streaming'
  | 'turn_completed'
  | 'call_completed'
  | 'error';

export type TelephonyCallState =
  | 'idle'
  | 'connected'
  | 'listening'
  | 'processing'
  | 'speaking'
  | 'error'
  | 'completed';

export interface TelephonyEvent {
  type: TelephonyEventType;
  call_id: string;
  timestamp: string;
  message: string;
  turn?: number;
  metadata?: Record<string, unknown>;
}

export interface TelephonyTurnInfo {
  turn: number;
  transcript?: string;
  response?: string;
  llm_duration_ms?: number;
  tts_duration_ms?: number;
  first_audio_ms?: number;
  audio_bytes?: number;
  audio_chunks?: number;
  speech_final_to_utterance_end_ms?: number;
  utterance_end_to_agent_ms?: number;
  speech_final_to_agent_ms?: number;
  release_reason?: string;
  settle_ms?: number;
  // Streaming metrics
  streamed?: boolean;
  first_token_ms?: number;
  first_sentence_ms?: number;
}
