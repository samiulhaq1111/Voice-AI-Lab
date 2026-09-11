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
  configured: boolean;
  default_model?: string;
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
