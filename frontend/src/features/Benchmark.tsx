/** Benchmark — Phase 5B runner + 5C analytics + 5D cost + 5E comparison (UX improved). */

import { useCallback, useEffect, useState } from 'react';
import type {
  BenchmarkComparisonResult,
  BenchmarkConfiguration,
  BenchmarkCostBreakdown,
  BenchmarkCostSummary,
  BenchmarkOverallSummary,
  BenchmarkProviderSummary,
  BenchmarkRecentResult,
  BenchmarkRunResult,
  BenchmarkScenario,
  BenchmarkScenarioSummary,
  ComparisonConfigurationResult,
  LatencyStats,
} from '../types';
import {
  getBenchmarkConfigurations,
  getBenchmarkCostSummary,
  getBenchmarkRecentResults,
  getBenchmarkRunCost,
  getBenchmarkScenarioSummaries,
  getBenchmarkScenarios,
  getBenchmarkSummary,
  getBenchmarkProviderSummaries,
  resetBenchmarkData,
  runBenchmark,
  runBenchmarkComparison,
} from '../services/api';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format ms with unit: <1000 → "123 ms", ≥1000 → "1.97 s". */
const fmtDuration = (v: number | null): string => {
  if (v == null) return 'No data';
  if (v < 1000) return `${Math.round(v)} ms`;
  return `${(v / 1000).toFixed(2)} s`;
};

/** Format percentage with fraction: "100% (3/3)". */
const fmtSuccessRate = (rate: number, success: number, total: number): string => {
  if (total === 0) return 'No runs yet';
  return `${(rate * 100).toFixed(1)}% (${success}/${total})`;
};

const fmtPct = (v: number) => `${(v * 100).toFixed(1)}%`;

const fmtCost = (v: number | null) => {
  if (v == null) return 'Unavailable';
  if (v === 0) return '$0.00';
  if (v < 0.001) return `$${v.toFixed(6)}`;
  if (v < 0.01) return `$${v.toFixed(5)}`;
  return `$${v.toFixed(4)}`;
};

const fmtBytes = (v: number | null) => (v == null ? 'No data' : `${(v / 1024).toFixed(1)} KB`);

/** Classify an error message into a human-friendly category. */
function classifyError(msg: string | null | undefined): {
  label: string;
  color: string;
} {
  if (!msg) return { label: 'Failed', color: 'text-red-400' };
  const lower = msg.toLowerCase();
  if (lower.includes('429') || lower.includes('rate limit') || lower.includes('rate_limit')) {
    return { label: 'Rate limited', color: 'text-yellow-400' };
  }
  if (lower.includes('4') && (lower.includes('http') || lower.includes('status'))) {
    return { label: 'Provider error', color: 'text-orange-400' };
  }
  return { label: 'Failed', color: 'text-red-400' };
}

/** Format a Date or ISO string as HH:MM:SS. */
const fmtTime = (d: Date | string) => {
  const date = typeof d === 'string' ? new Date(d) : d;
  return date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
};

const categoryColor = (cat: string) => {
  switch (cat) {
    case 'baseline': return 'text-blue-400';
    case 'tool': return 'text-purple-400';
    case 'stress': return 'text-orange-400';
    default: return 'text-gray-400';
  }
};

const stageColor = (stage: string) => {
  switch (stage) {
    case 'stt': return 'text-green-400';
    case 'llm': return 'text-blue-400';
    case 'tts': return 'text-purple-400';
    default: return 'text-gray-400';
  }
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function Benchmark() {
  const [scenarios, setScenarios] = useState<BenchmarkScenario[]>([]);
  const [running, setRunning] = useState<string | null>(null);
  const [result, setResult] = useState<BenchmarkRunResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Individual run configuration selector (independent from comparison)
  const [runConfigId, setRunConfigId] = useState<string>('');

  // Comparison state (Phase 5E)
  const [configurations, setConfigurations] = useState<BenchmarkConfiguration[]>([]);
  const [selectedConfigs, setSelectedConfigs] = useState<Set<string>>(new Set());
  const [compareScenario, setCompareScenario] = useState('');
  const [compareReps, setCompareReps] = useState(3);
  const [comparing, setComparing] = useState(false);
  const [comparisonResult, setComparisonResult] = useState<BenchmarkComparisonResult | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);

  // Analytics state
  const [overall, setOverall] = useState<BenchmarkOverallSummary | null>(null);
  const [scenarioSummaries, setScenarioSummaries] = useState<BenchmarkScenarioSummary[]>([]);
  const [providerSummaries, setProviderSummaries] = useState<BenchmarkProviderSummary[]>([]);
  const [recentResults, setRecentResults] = useState<BenchmarkRecentResult[]>([]);
  const [costSummary, setCostSummary] = useState<BenchmarkCostSummary | null>(null);
  const [runCost, setRunCost] = useState<BenchmarkCostBreakdown | null>(null);

  // Refresh state
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // Reset state
  const [showResetDialog, setShowResetDialog] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [resetMessage, setResetMessage] = useState<string | null>(null);

  // Load scenarios + configurations once
  useEffect(() => {
    getBenchmarkScenarios().then(setScenarios).catch(() => {});
    getBenchmarkConfigurations().then((configs) => {
      setConfigurations(configs);
      // Default: first production-eligible PAYG config, else first config
      if (configs.length > 0 && !runConfigId) {
        const paygProd = configs.find((c) => c.production_eligible && c.pricing_type === 'payg');
        setRunConfigId((paygProd ?? configs[0]).configuration_id);
      }
    }).catch(() => {});
    doRefresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Fetch all analytics data and replace state. */
  const doRefresh = useCallback(async () => {
    setRefreshing(true);
    setRefreshError(null);
    try {
      const [summary, scenSummaries, provSummaries, recent, cost] = await Promise.all([
        getBenchmarkSummary(),
        getBenchmarkScenarioSummaries(),
        getBenchmarkProviderSummaries(),
        getBenchmarkRecentResults({ limit: 10 }),
        getBenchmarkCostSummary(),
      ]);
      setOverall(summary);
      setScenarioSummaries(scenSummaries);
      setProviderSummaries(provSummaries);
      setRecentResults(recent);
      setCostSummary(cost);
      setLastUpdated(new Date());
    } catch {
      setRefreshError('Unable to refresh benchmark data.');
    } finally {
      setRefreshing(false);
    }
  }, []);

  /** Debounced auto-refresh after a run completes (cleared after 30 s). */
  const handleRun = useCallback(async (scenarioId: string) => {
    setRunning(scenarioId);
    setResult(null);
    setError(null);
    setRunCost(null);
    try {
      const res = await runBenchmark({ scenario_id: scenarioId, configuration_id: runConfigId || undefined });
      setResult(res);
      if (res.run_id) {
        getBenchmarkRunCost(res.run_id).then(setRunCost).catch(() => {});
      }
      // Refresh analytics after run
      await doRefresh();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Unknown error';
      setError(msg);
    } finally {
      setRunning(null);
    }
  }, [doRefresh, runConfigId]);

  const toggleConfig = useCallback((id: string) => {
    setSelectedConfigs((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const handleCompare = useCallback(async () => {
    if (!compareScenario || selectedConfigs.size < 2) return;
    setComparing(true);
    setCompareError(null);
    setComparisonResult(null);
    try {
      const res = await runBenchmarkComparison({
        scenario_id: compareScenario,
        configuration_ids: Array.from(selectedConfigs),
        repetitions: compareReps,
      });
      setComparisonResult(res);
      doRefresh();
    } catch (e: unknown) {
      setCompareError(e instanceof Error ? e.message : 'Comparison failed');
    } finally {
      setComparing(false);
    }
  }, [compareScenario, selectedConfigs, compareReps, doRefresh]);

  const handleReset = useCallback(async () => {
    setResetting(true);
    setResetMessage(null);
    try {
      const res = await resetBenchmarkData();
      setResetMessage(`Benchmark data reset. ${res.deleted} record${res.deleted !== 1 ? 's' : ''} deleted.`);
      setShowResetDialog(false);
      // Clear frontend state
      setOverall({ total_runs: 0, successful_runs: 0, failed_runs: 0, success_rate: 0,
        latency: { avg_ms: null, median_ms: null, min_ms: null, max_ms: null },
        stt_latency: { avg_ms: null, median_ms: null, min_ms: null, max_ms: null },
        llm_latency: { avg_ms: null, median_ms: null, min_ms: null, max_ms: null },
        tts_latency: { avg_ms: null, median_ms: null, min_ms: null, max_ms: null },
        tool_execution: { avg_ms: null, median_ms: null, min_ms: null, max_ms: null },
        avg_prompt_tokens: null, avg_completion_tokens: null, avg_total_tokens: null,
        avg_tts_characters: null, avg_tts_audio_bytes: null });
      setScenarioSummaries([]);
      setProviderSummaries([]);
      setRecentResults([]);
      setCostSummary(null);
      setComparisonResult(null);
      setResult(null);
      setRunCost(null);
    } catch {
      setResetMessage('Failed to reset benchmark data.');
    } finally {
      setResetting(false);
    }
  }, []);

  // Lookup maps
  const scenarioMap = new Map(scenarios.map((s) => [s.scenario_id, s]));
  const configMap = new Map(configurations.map((c) => [c.configuration_id, c]));
  const compareScenarioObj = scenarioMap.get(compareScenario);
  const selectedRunConfig = configMap.get(runConfigId);

  /** Render latency for the Performance section. */
  const renderLatency = (s: LatencyStats, stage: string) => {
    if (stage === 'stt') {
      return (
        <div>
          <div className="text-gray-500 text-[11px]">Not measured</div>
          <div className="text-gray-600 text-[10px]">Text benchmark</div>
        </div>
      );
    }
    if (s.avg_ms == null) {
      // Distinguish: runs exist but no successful measurement vs no runs at all
      return <span className="text-gray-600">No data</span>;
    }
    return (
      <div>
        <div className="text-gray-200">{fmtDuration(s.avg_ms)}</div>
        <div className="text-gray-500 text-[11px]">Median: {fmtDuration(s.median_ms)}</div>
      </div>
    );
  };

  const totalRuns = selectedConfigs.size * compareReps;
  const hasRuns = overall != null && overall.total_runs > 0;
  const allFailed = hasRuns && overall.successful_runs === 0;

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto w-full">
      {/* ── Header ── */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div>
          <h2 className="text-lg font-semibold text-white">Benchmark</h2>
          <p className="text-xs text-gray-500">
            Controlled testing & comparison of AI configurations
          </p>
        </div>
        <div className="flex items-center gap-2">
          {lastUpdated && (
            <span className="text-[10px] text-gray-600">
              Last updated: {fmtTime(lastUpdated)}
            </span>
          )}
          <button
            onClick={doRefresh}
            disabled={refreshing}
            className="px-3 py-1 bg-gray-700 text-gray-300 rounded text-xs hover:bg-gray-600 disabled:opacity-50"
          >
            {refreshing ? 'Refreshing…' : 'Refresh'}
          </button>
          <button
            onClick={() => setShowResetDialog(true)}
            disabled={refreshing || resetting}
            className="px-3 py-1 bg-red-900/40 text-red-300 rounded text-xs hover:bg-red-900/60 disabled:opacity-50"
          >
            Reset Data
          </button>
        </div>
      </div>
      {refreshError && (
        <div className="mx-4 mt-2 text-xs text-yellow-400 bg-yellow-900/20 rounded px-3 py-2 flex items-center justify-between">
          <span>{refreshError}</span>
          <button onClick={doRefresh} className="underline ml-2">Retry</button>
        </div>
      )}
      {resetMessage && (
        <div className="mx-4 mt-2 text-xs text-green-400 bg-green-900/20 rounded px-3 py-2">
          {resetMessage}
        </div>
      )}

      {/* ── Reset Confirmation Dialog ── */}
      {showResetDialog && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 max-w-sm mx-4 space-y-4">
            <h3 className="text-sm font-semibold text-white">Reset Benchmark Data?</h3>
            <p className="text-xs text-gray-400">
              This will permanently delete recorded benchmark runs, comparison results, and benchmark history.
            </p>
            <p className="text-xs text-gray-500">
              Scenarios and provider configurations will not be deleted.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowResetDialog(false)}
                className="px-3 py-1.5 bg-gray-700 text-gray-300 rounded text-xs hover:bg-gray-600"
              >
                Cancel
              </button>
              <button
                onClick={handleReset}
                disabled={resetting}
                className="px-3 py-1.5 bg-red-600 text-white rounded text-xs hover:bg-red-500 disabled:opacity-50"
              >
                {resetting ? 'Resetting…' : 'Reset Data'}
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-6">
        {/* ── Overview ── */}
        {overall && (
          <section>
            <h3 className="text-sm font-semibold text-gray-300 mb-2">Overview</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-xs text-gray-500">Recorded Runs</div>
                <div className="text-lg font-mono text-white">{overall.total_runs}</div>
                <div className="text-[10px] text-gray-600">Stored benchmark results</div>
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-xs text-gray-500">Success Rate</div>
                {overall.total_runs === 0 ? (
                  <div className="text-lg font-mono text-gray-600">No runs yet</div>
                ) : (
                  <>
                    <div className={`text-lg font-mono ${overall.successful_runs > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {fmtPct(overall.success_rate)}
                    </div>
                    <div className="text-[10px] text-gray-500">
                      {overall.successful_runs}/{overall.total_runs} successful
                    </div>
                  </>
                )}
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-xs text-gray-500">Avg Latency</div>
                <div className="text-lg font-mono text-white">
                  {allFailed ? <span className="text-gray-600 text-sm">No successful measurements</span> : fmtDuration(overall.latency.avg_ms)}
                </div>
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-xs text-gray-500">Median Latency</div>
                <div className="text-lg font-mono text-white">
                  {allFailed ? <span className="text-gray-600 text-sm">No successful measurements</span> : fmtDuration(overall.latency.median_ms)}
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ── Performance ── */}
        {hasRuns && (
          <section>
            <h3 className="text-sm font-semibold text-gray-300 mb-2">Performance</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-gray-500 mb-1">LLM</div>
                {renderLatency(overall!.llm_latency, 'llm')}
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-gray-500 mb-1">TTS</div>
                {renderLatency(overall!.tts_latency, 'tts')}
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-gray-500 mb-1">Tools</div>
                {overall!.successful_runs > 0 && overall!.tool_execution.avg_ms == null ? (
                  <div>
                    <div className="text-gray-500 text-[11px]">Not used</div>
                    <div className="text-gray-600 text-[10px]">0 calls</div>
                  </div>
                ) : renderLatency(overall!.tool_execution, 'tools')}
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-gray-500 mb-1">STT</div>
                {renderLatency(overall!.stt_latency, 'stt')}
              </div>
            </div>
          </section>
        )}

        {/* ── Cost Summary ── */}
        {costSummary && costSummary.total_runs > 0 ? (
          <section>
            <h3 className="text-sm font-semibold text-gray-300 mb-2">Cost Summary</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-xs text-gray-500">Total Cost</div>
                <div className="text-lg font-mono text-white">{fmtCost(costSummary.total_cost)}</div>
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-xs text-gray-500">Avg Cost</div>
                <div className="text-lg font-mono text-white">{fmtCost(costSummary.avg_cost)}</div>
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-xs text-gray-500">Min Cost</div>
                <div className="text-lg font-mono text-gray-300">{fmtCost(costSummary.min_cost)}</div>
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-xs text-gray-500">Max Cost</div>
                <div className="text-lg font-mono text-gray-300">{fmtCost(costSummary.max_cost)}</div>
              </div>
            </div>
            <div className="text-xs text-gray-600 mt-1">
              Cost data available for {costSummary.runs_with_cost} of {costSummary.total_runs} runs • v{costSummary.pricing_version}
            </div>
          </section>
        ) : hasRuns ? (
          <section>
            <h3 className="text-sm font-semibold text-gray-300 mb-2">Cost Summary</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {['Total Cost', 'Avg Cost', 'Min Cost', 'Max Cost'].map((label) => (
                <div key={label} className="bg-gray-800 rounded px-3 py-2">
                  <div className="text-xs text-gray-500">{label}</div>
                  <div className="text-lg font-mono text-gray-600">Unavailable</div>
                </div>
              ))}
            </div>
          </section>
        ) : null}

        {/* ── Scenarios ── */}
        {scenarioSummaries.length > 0 && (
          <section>
            <h3 className="text-sm font-semibold text-gray-300 mb-2">Scenarios</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-700">
                    <th className="text-left py-1 pr-2">Scenario</th>
                    <th className="text-right py-1 px-2">Runs</th>
                    <th className="text-right py-1 px-2">Success</th>
                    <th className="text-right py-1 px-2">Avg Latency</th>
                    <th className="text-right py-1 px-2">Median</th>
                    <th className="text-right py-1 px-2">Tokens</th>
                  </tr>
                </thead>
                <tbody>
                  {scenarioSummaries.map((s) => (
                    <tr key={s.scenario_id} className="border-b border-gray-800">
                      <td className="py-1.5 pr-2 text-white">{s.scenario_id}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">{s.run_count}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-green-400">
                        {fmtSuccessRate(s.success_rate, s.successful_runs, s.run_count)}
                      </td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">{fmtDuration(s.latency.avg_ms)}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">{fmtDuration(s.latency.median_ms)}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">
                        {s.avg_total_tokens != null ? Math.round(s.avg_total_tokens) : 'No data'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* ── Providers / Models ── */}
        {providerSummaries.length > 0 && (
          <section>
            <h3 className="text-sm font-semibold text-gray-300 mb-1">Provider / Model Usage</h3>
            <p className="text-[10px] text-gray-600 mb-2">Usage across recorded benchmark runs, including previous tests.</p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-700">
                    <th className="text-left py-1 pr-2">Stage</th>
                    <th className="text-left py-1 px-2">Provider</th>
                    <th className="text-left py-1 px-2">Model</th>
                    <th className="text-right py-1 px-2">Runs</th>
                    <th className="text-right py-1 px-2">Success</th>
                    <th className="text-right py-1 px-2">Avg Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {providerSummaries.map((p, i) => (
                    <tr key={i} className="border-b border-gray-800">
                      <td className={`py-1.5 pr-2 font-medium ${stageColor(p.stage)}`}>{p.stage}</td>
                      <td className="py-1.5 px-2 text-white">{p.provider ?? 'No data'}</td>
                      <td className="py-1.5 px-2 text-gray-400 font-mono truncate max-w-[200px]">
                        {p.model ?? 'No data'}
                      </td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">{p.run_count}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-green-400">
                        {fmtSuccessRate(p.success_rate, p.successful_runs, p.run_count)}
                      </td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">{fmtDuration(p.latency.avg_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* ── Run a Benchmark ── */}
        <section>
          <h3 className="text-sm font-semibold text-gray-300 mb-1">Run a Benchmark</h3>
          <p className="text-xs text-gray-500 mb-3">
            Run one scenario using one selected AI configuration.
          </p>

          {/* Configuration selector */}
          <div className="mb-3">
            <label className="text-xs text-gray-400 block mb-1">Configuration</label>
            <select
              value={runConfigId}
              onChange={(e) => setRunConfigId(e.target.value)}
              className="bg-gray-900 border border-gray-600 rounded px-2 py-1.5 text-sm text-white w-full max-w-xs"
            >
              {configurations.map((c) => (
                <option key={c.configuration_id} value={c.configuration_id}>{c.name}</option>
              ))}
            </select>
            <p className="text-[10px] text-gray-600 mt-1">Select the AI stack used for an individual benchmark run.</p>
            {selectedRunConfig && (
              <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px]">
                <span className="text-gray-400">LLM: <span className="text-gray-300">{selectedRunConfig.llm_provider} / {selectedRunConfig.llm_model}</span></span>
                <span className="text-gray-400">TTS: <span className="text-gray-300">{selectedRunConfig.tts_provider} / {selectedRunConfig.tts_model}</span></span>
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                  selectedRunConfig.pricing_type === 'free'
                    ? 'bg-yellow-900/50 text-yellow-300'
                    : 'bg-green-900/50 text-green-300'
                }`}>
                  {selectedRunConfig.pricing_type === 'free' ? 'FREE' : 'PAYG'}
                </span>
                {selectedRunConfig.production_eligible ? (
                  <span className="text-[10px] text-green-500">Production eligible</span>
                ) : (
                  <span className="text-[10px] text-yellow-500">
                    {selectedRunConfig.pricing_type === 'free' ? 'Free • Rate limited' : 'Not production eligible'}
                  </span>
                )}
              </div>
            )}
          </div>

          <div className="space-y-2">
            {scenarios.map((s) => (
              <div
                key={s.scenario_id}
                className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 flex items-center justify-between gap-4"
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-white">{s.name}</span>
                    <span className={`text-xs ${categoryColor(s.category)}`}>{s.category}</span>
                    {s.include_tts && <span className="text-xs text-gray-600">+TTS</span>}
                  </div>
                  <p className="text-xs text-gray-400 truncate">{s.description}</p>
                </div>
                <button
                  onClick={() => handleRun(s.scenario_id)}
                  disabled={running !== null}
                  className="px-4 py-1.5 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-500 disabled:opacity-50 whitespace-nowrap"
                >
                  {running === s.scenario_id ? 'Running…' : 'Run'}
                </button>
              </div>
            ))}
          </div>

          {/* Run result */}
          {error && (
            <div className="bg-red-900/30 border border-red-700 rounded-lg px-4 py-3 text-sm text-red-300 mt-3">
              {error}
            </div>
          )}
          {result && (
            <div className="bg-gray-900 border border-gray-700 rounded-lg px-4 py-4 space-y-3 mt-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className={`inline-block w-2 h-2 rounded-full ${result.success ? 'bg-green-400' : 'bg-red-400'}`} />
                  <span className={`text-sm font-medium ${result.success ? 'text-green-400' : 'text-red-400'}`}>
                    {result.success ? 'Success' : 'Failed'}
                  </span>
                </div>
                <span className="text-xs text-gray-500 font-mono">{result.run_id.slice(0, 8)}…</span>
              </div>
              {result.configuration_id && configMap.get(result.configuration_id) && (
                <div className="text-[11px] text-gray-400">
                  Configuration: <span className="text-gray-300">{configMap.get(result.configuration_id)!.name}</span>
                </div>
              )}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                <div className="flex justify-between"><span className="text-gray-500">LLM</span><span className="text-gray-200">{fmtDuration(result.llm_latency_ms)}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">TTS</span><span className="text-gray-200">{fmtDuration(result.tts_latency_ms)}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Total</span><span className="text-blue-300">{fmtDuration(result.total_processing_ms)}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Tools</span><span className={result.tool_call_match ? 'text-green-400' : 'text-red-400'}>{result.actual_tool_calls}/{result.expected_tool_calls}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Tokens</span><span className="text-gray-200">{result.usage.total_tokens ?? 'No data'}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Audio</span><span className="text-gray-200">{fmtBytes(result.tts_audio_bytes)}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Tool Exec</span><span className="text-gray-200">{fmtDuration(result.tool_execution_ms)}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Mode</span><span className="text-gray-200">{result.benchmark_mode}</span></div>
              </div>
              {result.validation_errors.length > 0 && (
                <div className="space-y-1">
                  {result.validation_errors.map((err, i) => (
                    <div key={i} className="text-xs text-red-300 font-mono bg-red-900/20 rounded px-2 py-1">{err}</div>
                  ))}
                </div>
              )}
              {/* Cost Breakdown */}
              {runCost && (
                <div className="border-t border-gray-700 pt-3 mt-3">
                  <div className="text-xs text-gray-400 mb-2">Cost Breakdown</div>
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs font-mono">
                    <div className="flex justify-between">
                      <span className="text-gray-500">STT</span>
                      <span className="text-gray-200">{fmtCost(runCost.stt_cost)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">LLM In</span>
                      <span className="text-gray-200">{fmtCost(runCost.llm_input_cost)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">LLM Out</span>
                      <span className="text-gray-200">{fmtCost(runCost.llm_output_cost)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">TTS</span>
                      <span className="text-gray-200">{fmtCost(runCost.tts_cost)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Total</span>
                      <span className="text-green-400">{fmtCost(runCost.total_cost)}</span>
                    </div>
                  </div>
                  {!runCost.pricing_available && (
                    <div className="text-xs text-yellow-500 mt-1">Some pricing unavailable</div>
                  )}
                </div>
              )}
            </div>
          )}
        </section>

        {/* ── Compare Configurations (Phase 5E) ── */}
        <section>
          <h3 className="text-sm font-semibold text-gray-300 mb-1">Compare Configurations</h3>
          <p className="text-xs text-gray-500 mb-3">
            Run the same scenario multiple times against different AI stacks.
          </p>
          <div className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 space-y-4">
            {/* Scenario selector */}
            <div>
              <label className="text-xs text-gray-400 block mb-1">Scenario</label>
              <select
                value={compareScenario}
                onChange={(e) => setCompareScenario(e.target.value)}
                className="bg-gray-900 border border-gray-600 rounded px-2 py-1.5 text-sm text-white w-full max-w-xs"
              >
                <option value="">Select a scenario…</option>
                {scenarios.map((s) => (
                  <option key={s.scenario_id} value={s.scenario_id}>{s.name}</option>
                ))}
              </select>
              {compareScenarioObj && (
                <p className="text-[11px] text-gray-500 mt-1">{compareScenarioObj.description}</p>
              )}
            </div>

            {/* Configuration checkboxes */}
            <div>
              <label className="text-xs text-gray-400 block mb-1">
                Configurations
                <span className="text-gray-600 ml-1">— Select 2 or more AI stacks to compare.</span>
              </label>
              <div className="space-y-1.5">
                {configurations.map((c) => {
                  const isSelected = selectedConfigs.has(c.configuration_id);
                  return (
                    <label
                      key={c.configuration_id}
                      className={`block rounded border px-3 py-2 cursor-pointer transition-colors ${
                        isSelected
                          ? 'border-blue-500 bg-blue-900/20'
                          : 'border-gray-700 hover:border-gray-600'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggleConfig(c.configuration_id)}
                        className="mr-2 accent-blue-500"
                      />
                      <span className="text-sm text-white">{c.name}</span>
                      <div className="text-[11px] text-gray-500 ml-5 mt-0.5">
                        LLM: {c.llm_provider} / {c.llm_model}
                        <br />
                        TTS: {c.tts_provider} / {c.tts_model}
                      </div>
                      <div className="ml-5 mt-1 flex items-center gap-2">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                          c.pricing_type === 'free'
                            ? 'bg-yellow-900/50 text-yellow-300'
                            : 'bg-green-900/50 text-green-300'
                        }`}>
                          {c.pricing_type === 'free' ? 'FREE' : 'PAYG'}
                        </span>
                        {c.production_eligible ? (
                          <span className="text-[10px] text-green-500">Production eligible</span>
                        ) : (
                          <span className="text-[10px] text-yellow-500">
                            {c.pricing_type === 'free' ? 'Free • Rate limited' : 'Not production eligible'}
                          </span>
                        )}
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Repetitions + Run */}
            <div className="flex flex-wrap items-end gap-4">
              <div>
                <label className="text-xs text-gray-400 block mb-1">Repetitions</label>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={compareReps}
                  onChange={(e) => setCompareReps(Math.max(1, Math.min(10, Number(e.target.value))))}
                  className="bg-gray-900 border border-gray-600 rounded px-2 py-1.5 text-sm text-white w-16"
                />
                <p className="text-[10px] text-gray-600 mt-1">
                  How many times each configuration runs the scenario.
                </p>
              </div>
              <div className="pb-2">
                <button
                  onClick={handleCompare}
                  disabled={comparing || !compareScenario || selectedConfigs.size < 2}
                  className="px-4 py-1.5 bg-indigo-600 text-white rounded text-sm font-medium hover:bg-indigo-500 disabled:opacity-50"
                >
                  {comparing ? 'Running Comparison…' : 'Run Comparison'}
                </button>
              </div>
            </div>
            {selectedConfigs.size >= 1 && (
              <p className="text-[11px] text-gray-500">
                {selectedConfigs.size} configuration{selectedConfigs.size !== 1 ? 's' : ''} × {compareReps} repetition{compareReps !== 1 ? 's' : ''} = {totalRuns} run{totalRuns !== 1 ? 's' : ''}
              </p>
            )}
          </div>

          {/* Comparison error */}
          {compareError && (
            <div className="bg-red-900/30 border border-red-700 rounded-lg px-4 py-3 text-sm text-red-300 mt-3">
              {compareError}
            </div>
          )}

          {/* Comparison result table */}
          {comparisonResult && comparisonResult.configurations.length > 0 && (
            <div className="mt-3">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <h4 className="text-xs font-semibold text-green-400">Comparison complete</h4>
                  <p className="text-[10px] text-gray-500">
                    {comparisonResult.configurations.length} configuration{comparisonResult.configurations.length !== 1 ? 's' : ''} × {comparisonResult.repetitions} repetition{comparisonResult.repetitions !== 1 ? 's' : ''} = {comparisonResult.configurations.length * comparisonResult.repetitions} run{comparisonResult.configurations.length * comparisonResult.repetitions !== 1 ? 's' : ''}
                  </p>
                </div>
                <span className="text-[10px] text-gray-600 font-mono">
                  {comparisonResult.comparison_id.slice(0, 8)}…
                </span>
              </div>
              {comparisonResult.validation_errors.length > 0 && (
                <div className="space-y-1 mb-2">
                  {comparisonResult.validation_errors.map((err, i) => (
                    <div key={i} className="text-xs text-red-300 font-mono bg-red-900/20 rounded px-2 py-1">{err}</div>
                  ))}
                </div>
              )}
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-gray-500 border-b border-gray-700">
                      <th className="text-left py-1 pr-2">Configuration</th>
                      <th className="text-right py-1 px-2">Success</th>
                      <th className="text-right py-1 px-2">Avg Latency</th>
                      <th className="text-right py-1 px-2">Median Latency</th>
                      <th className="text-right py-1 px-2">Tokens</th>
                      <th className="text-right py-1 px-2">Cost</th>
                      <th className="text-center py-1 px-2">Pricing</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparisonResult.configurations.map((c: ComparisonConfigurationResult) => (
                      <tr key={c.configuration_id} className="border-b border-gray-800">
                        <td className="py-1.5 pr-2 text-white">
                          {c.configuration_name}
                        </td>
                        <td className="py-1.5 px-2 text-right font-mono text-green-400">
                          {fmtSuccessRate(c.success_rate, c.successful_runs, c.run_count)}
                        </td>
                        <td className="py-1.5 px-2 text-right font-mono text-gray-300">
                          {c.avg_total_latency_ms != null ? fmtDuration(c.avg_total_latency_ms) : c.failed_runs > 0 ? <span className="text-gray-600">No successful measurements</span> : 'No data'}
                        </td>
                        <td className="py-1.5 px-2 text-right font-mono text-gray-300">
                          {c.median_total_latency_ms != null ? fmtDuration(c.median_total_latency_ms) : c.failed_runs > 0 ? <span className="text-gray-600">No successful measurements</span> : 'No data'}
                        </td>
                        <td className="py-1.5 px-2 text-right font-mono text-gray-300">
                          {c.avg_total_tokens != null ? Math.round(c.avg_total_tokens) : 'No data'}
                        </td>
                        <td className="py-1.5 px-2 text-right font-mono text-gray-300">
                          {c.cost_note === 'no_usage' ? (
                            <span className="text-gray-600">No usage</span>
                          ) : c.cost_note === 'free' ? (
                            '$0.00'
                          ) : c.cost_available ? (
                            fmtCost(c.avg_cost)
                          ) : (
                            'Unavailable'
                          )}
                        </td>
                        <td className="py-1.5 px-2 text-center">
                          {c.pricing_type === 'free' ? (
                            <span className="px-1.5 py-0.5 rounded bg-yellow-900/50 text-yellow-300 text-[10px] font-medium">FREE</span>
                          ) : c.cost_available ? (
                            <span className="px-1.5 py-0.5 rounded bg-green-900/50 text-green-300 text-[10px] font-medium">PAYG</span>
                          ) : (
                            <span className="px-1.5 py-0.5 rounded bg-gray-700 text-gray-400 text-[10px] font-medium">Unavailable</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="text-[10px] text-gray-600 mt-1">
                STT: Not measured (text_input mode)
              </div>
            </div>
          )}
        </section>

        {/* ── Recent Runs ── */}
        <section>
          <h3 className="text-sm font-semibold text-gray-300 mb-1">Recent Runs</h3>
          {recentResults.length > 0 ? (
            <>
              <p className="text-[10px] text-gray-600 mb-2">Showing latest {recentResults.length} runs</p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-700">
                    <th className="text-left py-1 pr-2">Scenario</th>
                    <th className="text-left py-1 px-2">Configuration</th>
                    <th className="text-center py-1 px-2">Status</th>
                    <th className="text-right py-1 px-2">Total</th>
                    <th className="text-right py-1 px-2">LLM</th>
                    <th className="text-right py-1 px-2">Tokens</th>
                    <th className="text-right py-1 px-2">Cost</th>
                    <th className="text-right py-1 px-2">Run ID</th>
                  </tr>
                </thead>
                <tbody>
                  {recentResults.map((r) => {
                    const errInfo = !r.success ? classifyError(r.error_message) : null;
                    return (
                      <tr key={r.id} className="border-b border-gray-800">
                        <td className="py-1.5 pr-2 text-white">{r.scenario_id ?? '—'}</td>
                        <td className="py-1.5 px-2 text-gray-400 text-[11px]">
                          {r.configuration_id && configMap.get(r.configuration_id)
                            ? configMap.get(r.configuration_id)!.name
                            : r.llm_provider
                              ? `${r.llm_provider}/${r.llm_model ?? '?'}`
                              : '—'}
                        </td>
                        <td className="py-1.5 px-2 text-center">
                          {r.success ? (
                            <span className="inline-block w-2 h-2 rounded-full bg-green-400" />
                          ) : (
                            <span className={`text-[10px] font-medium ${errInfo?.color ?? 'text-red-400'}`}>
                              {errInfo?.label ?? 'Failed'}
                            </span>
                          )}
                        </td>
                        <td className="py-1.5 px-2 text-right font-mono text-gray-300">{fmtDuration(r.total_processing_ms)}</td>
                        <td className="py-1.5 px-2 text-right font-mono text-gray-300">{fmtDuration(r.llm_latency_ms)}</td>
                        <td className="py-1.5 px-2 text-right font-mono text-gray-300">{r.token_usage ?? 'No data'}</td>
                        <td className="py-1.5 px-2 text-right font-mono text-gray-300">{fmtCost(r.total_cost)}</td>
                        <td className="py-1.5 px-2 text-right font-mono text-gray-500">{r.run_id ? r.run_id.slice(0, 8) : '—'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            </>
          ) : (
            <p className="text-xs text-gray-600 mt-1">
              No benchmark runs yet. Run a scenario above to create your first benchmark result.
            </p>
          )}
        </section>
      </div>
    </div>
  );
}
