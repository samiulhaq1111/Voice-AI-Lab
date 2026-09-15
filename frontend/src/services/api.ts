/** API client for the Voice AI Lab backend. */

import type {
  BenchmarkBatchResult,
  BenchmarkRunRequest,
  BenchmarkRunResult,
  BenchmarkScenario,
  ChatRequest,
  ChatResponse,
  HealthStatus,
  ProviderAvailability,
  ToolInfo,
} from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    const msg = detail?.detail || `API error: ${response.status} ${response.statusText}`;
    throw new Error(msg);
  }
  return response.json() as Promise<T>;
}

/** Check backend health. */
export async function getHealth(): Promise<HealthStatus> {
  return fetchJson<HealthStatus>('/health');
}

/** List all registered tools. */
export async function getTools(): Promise<{ tools: ToolInfo[]; count: number }> {
  return fetchJson<{ tools: ToolInfo[]; count: number }>('/api/v1/tools');
}

/** Get available providers. */
export async function getProviders(): Promise<ProviderAvailability> {
  return fetchJson<ProviderAvailability>('/api/v1/providers');
}

/** Send a chat message to the agent. */
export async function sendChat(request: ChatRequest): Promise<ChatResponse> {
  return fetchJson<ChatResponse>('/api/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

// --- Benchmark API (Phase 5B) ---

/** List available benchmark scenarios. */
export async function getBenchmarkScenarios(): Promise<BenchmarkScenario[]> {
  return fetchJson<BenchmarkScenario[]>('/api/v1/benchmarks/scenarios');
}

/** Execute a benchmark scenario. */
export async function runBenchmark(request: BenchmarkRunRequest): Promise<BenchmarkRunResult> {
  return fetchJson<BenchmarkRunResult>('/api/v1/benchmarks/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

/** Execute a benchmark scenario N times with aggregation. */
export async function runBenchmarkBatch(request: BenchmarkRunRequest): Promise<BenchmarkBatchResult> {
  return fetchJson<BenchmarkBatchResult>('/api/v1/benchmarks/run/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}
