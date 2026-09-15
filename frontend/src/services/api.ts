/** API client for the Voice AI Lab backend. */

import type {
  BenchmarkBatchResult,
  BenchmarkComparisonResult,
  BenchmarkConfiguration,
  BenchmarkCostBreakdown,
  BenchmarkCostSummary,
  BenchmarkOverallSummary,
  BenchmarkProviderSummary,
  BenchmarkRecentResult,
  BenchmarkRunRequest,
  BenchmarkRunResult,
  BenchmarkScenario,
  BenchmarkScenarioSummary,
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

// --- Benchmark Analytics API (Phase 5C) ---

/** Get overall benchmark summary. */
export async function getBenchmarkSummary(params?: {
  scenario_id?: string;
  benchmark_mode?: string;
}): Promise<BenchmarkOverallSummary> {
  const qs = new URLSearchParams();
  if (params?.scenario_id) qs.set('scenario_id', params.scenario_id);
  if (params?.benchmark_mode) qs.set('benchmark_mode', params.benchmark_mode);
  const query = qs.toString();
  return fetchJson<BenchmarkOverallSummary>(
    `/api/v1/benchmarks/summary${query ? `?${query}` : ''}`,
  );
}

/** Get per-scenario summaries. */
export async function getBenchmarkScenarioSummaries(params?: {
  benchmark_mode?: string;
}): Promise<BenchmarkScenarioSummary[]> {
  const qs = new URLSearchParams();
  if (params?.benchmark_mode) qs.set('benchmark_mode', params.benchmark_mode);
  const query = qs.toString();
  return fetchJson<BenchmarkScenarioSummary[]>(
    `/api/v1/benchmarks/scenarios/summary${query ? `?${query}` : ''}`,
  );
}

/** Get per-provider/model summaries. */
export async function getBenchmarkProviderSummaries(params?: {
  benchmark_mode?: string;
}): Promise<BenchmarkProviderSummary[]> {
  const qs = new URLSearchParams();
  if (params?.benchmark_mode) qs.set('benchmark_mode', params.benchmark_mode);
  const query = qs.toString();
  return fetchJson<BenchmarkProviderSummary[]>(
    `/api/v1/benchmarks/providers/summary${query ? `?${query}` : ''}`,
  );
}

/** Get recent benchmark results. */
export async function getBenchmarkRecentResults(params?: {
  limit?: number;
  scenario_id?: string;
  benchmark_mode?: string;
}): Promise<BenchmarkRecentResult[]> {
  const qs = new URLSearchParams();
  if (params?.limit != null) qs.set('limit', String(params.limit));
  if (params?.scenario_id) qs.set('scenario_id', params.scenario_id);
  if (params?.benchmark_mode) qs.set('benchmark_mode', params.benchmark_mode);
  const query = qs.toString();
  return fetchJson<BenchmarkRecentResult[]>(
    `/api/v1/benchmarks/results${query ? `?${query}` : ''}`,
  );
}

// --- Benchmark Cost API (Phase 5D) ---

/** Get cost breakdown for a specific benchmark run. */
export async function getBenchmarkRunCost(runId: string): Promise<BenchmarkCostBreakdown> {
  return fetchJson<BenchmarkCostBreakdown>(`/api/v1/benchmarks/${runId}/cost`);
}

/** Get aggregate cost summary across benchmark runs. */
export async function getBenchmarkCostSummary(params?: {
  scenario_id?: string;
  benchmark_mode?: string;
}): Promise<BenchmarkCostSummary> {
  const qs = new URLSearchParams();
  if (params?.scenario_id) qs.set('scenario_id', params.scenario_id);
  if (params?.benchmark_mode) qs.set('benchmark_mode', params.benchmark_mode);
  const query = qs.toString();
  return fetchJson<BenchmarkCostSummary>(
    `/api/v1/benchmarks/cost/summary${query ? `?${query}` : ''}`,
  );
}

// --- Benchmark Comparison API (Phase 5E) ---

/** List available benchmark configurations. */
export async function getBenchmarkConfigurations(): Promise<BenchmarkConfiguration[]> {
  return fetchJson<BenchmarkConfiguration[]>('/api/v1/benchmarks/configurations');
}

/** Run a comparison across multiple configurations. */
export async function runBenchmarkComparison(request: {
  scenario_id: string;
  configuration_ids: string[];
  repetitions: number;
}): Promise<BenchmarkComparisonResult> {
  return fetchJson<BenchmarkComparisonResult>('/api/v1/benchmarks/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

/** Delete all persisted benchmark results (development aid). */
export async function resetBenchmarkData(): Promise<{ deleted: number; status: string }> {
  return fetchJson<{ deleted: number; status: string }>('/api/v1/benchmarks/results', {
    method: 'DELETE',
  });
}
