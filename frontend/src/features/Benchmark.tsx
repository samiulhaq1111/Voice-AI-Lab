/** Benchmark — Phase 5B runner + Phase 5C analytics UI. */

import { useCallback, useEffect, useState } from 'react';
import type {
  BenchmarkOverallSummary,
  BenchmarkProviderSummary,
  BenchmarkRecentResult,
  BenchmarkRunResult,
  BenchmarkScenario,
  BenchmarkScenarioSummary,
  LatencyStats,
} from '../types';
import {
  getBenchmarkRecentResults,
  getBenchmarkScenarioSummaries,
  getBenchmarkScenarios,
  getBenchmarkSummary,
  getBenchmarkProviderSummaries,
  runBenchmark,
} from '../services/api';

export default function Benchmark() {
  const [scenarios, setScenarios] = useState<BenchmarkScenario[]>([]);
  const [running, setRunning] = useState<string | null>(null);
  const [result, setResult] = useState<BenchmarkRunResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Analytics state
  const [overall, setOverall] = useState<BenchmarkOverallSummary | null>(null);
  const [scenarioSummaries, setScenarioSummaries] = useState<BenchmarkScenarioSummary[]>([]);
  const [providerSummaries, setProviderSummaries] = useState<BenchmarkProviderSummary[]>([]);
  const [recentResults, setRecentResults] = useState<BenchmarkRecentResult[]>([]);

  useEffect(() => {
    getBenchmarkScenarios().then(setScenarios).catch(() => {});
    loadAnalytics();
  }, []);

  const loadAnalytics = useCallback(() => {
    getBenchmarkSummary().then(setOverall).catch(() => {});
    getBenchmarkScenarioSummaries().then(setScenarioSummaries).catch(() => {});
    getBenchmarkProviderSummaries().then(setProviderSummaries).catch(() => {});
    getBenchmarkRecentResults({ limit: 20 }).then(setRecentResults).catch(() => {});
  }, []);

  const handleRun = useCallback(async (scenarioId: string) => {
    setRunning(scenarioId);
    setResult(null);
    setError(null);
    try {
      const res = await runBenchmark({ scenario_id: scenarioId });
      setResult(res);
      // Refresh analytics after run
      loadAnalytics();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Unknown error';
      setError(msg);
    } finally {
      setRunning(null);
    }
  }, [loadAnalytics]);

  const fmtMs = (v: number | null) => (v == null ? 'N/A' : `${Math.round(v)} ms`);
  const fmtBytes = (v: number | null) => (v == null ? 'N/A' : `${(v / 1024).toFixed(1)} KB`);
  const fmtPct = (v: number) => `${(v * 100).toFixed(1)}%`;

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

  const renderLatency = (s: LatencyStats) => {
    if (s.avg_ms == null) return <span className="text-gray-600">N/A</span>;
    return (
      <span className="text-gray-200">
        {Math.round(s.avg_ms)}ms <span className="text-gray-500">(med {Math.round(s.median_ms ?? 0)}ms)</span>
      </span>
    );
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto w-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div>
          <h2 className="text-lg font-semibold text-white">Benchmark</h2>
          <p className="text-xs text-gray-500">
            Phase 5C — Analytics &amp; Results
          </p>
        </div>
        <button
          onClick={loadAnalytics}
          className="px-3 py-1 bg-gray-700 text-gray-300 rounded text-xs hover:bg-gray-600"
        >
          Refresh
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-6">
        {/* Overall Summary */}
        {overall && overall.total_runs > 0 && (
          <section>
            <h3 className="text-sm font-semibold text-gray-300 mb-2">Overall</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-xs text-gray-500">Total Runs</div>
                <div className="text-lg font-mono text-white">{overall.total_runs}</div>
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-xs text-gray-500">Success Rate</div>
                <div className="text-lg font-mono text-green-400">{fmtPct(overall.success_rate)}</div>
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-xs text-gray-500">Avg Latency</div>
                <div className="text-lg font-mono text-white">{fmtMs(overall.latency.avg_ms)}</div>
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-xs text-gray-500">Median Latency</div>
                <div className="text-lg font-mono text-white">{fmtMs(overall.latency.median_ms)}</div>
              </div>
            </div>
          </section>
        )}

        {/* Latency Breakdown */}
        {overall && overall.total_runs > 0 && (
          <section>
            <h3 className="text-sm font-semibold text-gray-300 mb-2">Latency Breakdown</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-gray-500 mb-1">STT</div>
                {renderLatency(overall.stt_latency)}
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-gray-500 mb-1">LLM</div>
                {renderLatency(overall.llm_latency)}
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-gray-500 mb-1">TTS</div>
                {renderLatency(overall.tts_latency)}
              </div>
              <div className="bg-gray-800 rounded px-3 py-2">
                <div className="text-gray-500 mb-1">Tools</div>
                {renderLatency(overall.tool_execution)}
              </div>
            </div>
          </section>
        )}

        {/* Scenario Summaries */}
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
                    <th className="text-right py-1 px-2">Avg</th>
                    <th className="text-right py-1 px-2">Median</th>
                    <th className="text-right py-1 px-2">Tokens</th>
                  </tr>
                </thead>
                <tbody>
                  {scenarioSummaries.map((s) => (
                    <tr key={s.scenario_id} className="border-b border-gray-800">
                      <td className="py-1.5 pr-2 text-white">{s.scenario_id}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">{s.run_count}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-green-400">{fmtPct(s.success_rate)}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">{fmtMs(s.latency.avg_ms)}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">{fmtMs(s.latency.median_ms)}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">
                        {s.avg_total_tokens != null ? Math.round(s.avg_total_tokens) : 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Provider Summaries */}
        {providerSummaries.length > 0 && (
          <section>
            <h3 className="text-sm font-semibold text-gray-300 mb-2">Providers / Models</h3>
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
                      <td className="py-1.5 px-2 text-white">{p.provider ?? 'N/A'}</td>
                      <td className="py-1.5 px-2 text-gray-400 font-mono truncate max-w-[200px]">
                        {p.model ?? 'N/A'}
                      </td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">{p.run_count}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-green-400">{fmtPct(p.success_rate)}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">{fmtMs(p.latency.avg_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Scenario Runner */}
        <section>
          <h3 className="text-sm font-semibold text-gray-300 mb-2">Run Scenario</h3>
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
                  {running === s.scenario_id ? 'Running...' : 'Run'}
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
                  <span className="text-sm font-medium text-white">{result.success ? 'Success' : 'Failed'}</span>
                </div>
                <span className="text-xs text-gray-500 font-mono">{result.run_id.slice(0, 8)}...</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                <div className="flex justify-between"><span className="text-gray-500">LLM</span><span className="text-gray-200">{fmtMs(result.llm_latency_ms)}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">TTS</span><span className="text-gray-200">{fmtMs(result.tts_latency_ms)}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Total</span><span className="text-blue-300">{fmtMs(result.total_processing_ms)}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Tools</span><span className={result.tool_call_match ? 'text-green-400' : 'text-red-400'}>{result.actual_tool_calls}/{result.expected_tool_calls}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Tokens</span><span className="text-gray-200">{result.usage.total_tokens ?? 'N/A'}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Audio</span><span className="text-gray-200">{fmtBytes(result.tts_audio_bytes)}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Tool Exec</span><span className="text-gray-200">{fmtMs(result.tool_execution_ms)}</span></div>
                <div className="flex justify-between"><span className="text-gray-500">Mode</span><span className="text-gray-200">{result.benchmark_mode}</span></div>
              </div>
              {result.validation_errors.length > 0 && (
                <div className="space-y-1">
                  {result.validation_errors.map((err, i) => (
                    <div key={i} className="text-xs text-red-300 font-mono bg-red-900/20 rounded px-2 py-1">{err}</div>
                  ))}
                </div>
              )}
            </div>
          )}
        </section>

        {/* Recent Results */}
        {recentResults.length > 0 && (
          <section>
            <h3 className="text-sm font-semibold text-gray-300 mb-2">Recent Runs</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-700">
                    <th className="text-left py-1 pr-2">Scenario</th>
                    <th className="text-center py-1 px-2">Status</th>
                    <th className="text-right py-1 px-2">Total ms</th>
                    <th className="text-right py-1 px-2">LLM ms</th>
                    <th className="text-right py-1 px-2">Tokens</th>
                    <th className="text-right py-1 px-2">Run ID</th>
                  </tr>
                </thead>
                <tbody>
                  {recentResults.map((r) => (
                    <tr key={r.id} className="border-b border-gray-800">
                      <td className="py-1.5 pr-2 text-white">{r.scenario_id ?? '—'}</td>
                      <td className="py-1.5 px-2 text-center">
                        <span className={`inline-block w-2 h-2 rounded-full ${r.success ? 'bg-green-400' : 'bg-red-400'}`} />
                      </td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">{fmtMs(r.total_processing_ms)}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">{fmtMs(r.llm_latency_ms)}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-300">{r.token_usage ?? 'N/A'}</td>
                      <td className="py-1.5 px-2 text-right font-mono text-gray-500">{r.run_id ? r.run_id.slice(0, 8) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Empty state */}
        {overall && overall.total_runs === 0 && (
          <div className="text-center text-gray-600 mt-8">
            No benchmark runs yet. Run a scenario above to get started.
          </div>
        )}
      </div>
    </div>
  );
}
