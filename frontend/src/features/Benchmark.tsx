/** Benchmark — Phase 5B deterministic scenario runner UI. */

import { useCallback, useEffect, useState } from 'react';
import type {
  BenchmarkRunResult,
  BenchmarkScenario,
} from '../types';
import { getBenchmarkScenarios, runBenchmark } from '../services/api';

export default function Benchmark() {
  const [scenarios, setScenarios] = useState<BenchmarkScenario[]>([]);
  const [running, setRunning] = useState<string | null>(null);
  const [result, setResult] = useState<BenchmarkRunResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getBenchmarkScenarios().then(setScenarios).catch(() => {});
  }, []);

  const handleRun = useCallback(async (scenarioId: string) => {
    setRunning(scenarioId);
    setResult(null);
    setError(null);
    try {
      const res = await runBenchmark({ scenario_id: scenarioId });
      setResult(res);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Unknown error';
      setError(msg);
    } finally {
      setRunning(null);
    }
  }, []);

  const fmtMs = (v: number | null) => (v == null ? 'N/A' : `${Math.round(v)} ms`);
  const fmtBytes = (v: number | null) => (v == null ? 'N/A' : `${(v / 1024).toFixed(1)} KB`);

  const categoryColor = (cat: string) => {
    switch (cat) {
      case 'baseline':
        return 'text-blue-400';
      case 'tool':
        return 'text-purple-400';
      case 'stress':
        return 'text-orange-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto w-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div>
          <h2 className="text-lg font-semibold text-white">Benchmark</h2>
          <p className="text-xs text-gray-500">
            Phase 5B — Deterministic text_input scenarios (no STT)
          </p>
        </div>
      </div>

      {/* Scenario list */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
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
                  {s.include_tts && (
                    <span className="text-xs text-gray-600">+TTS</span>
                  )}
                </div>
                <p className="text-xs text-gray-400 truncate">{s.description}</p>
                <div className="text-xs text-gray-500 mt-1">
                  Expected tools: {s.expected_tool_calls}
                </div>
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
          {scenarios.length === 0 && (
            <div className="text-center text-gray-600 mt-8">
              Loading scenarios...
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-900/30 border border-red-700 rounded-lg px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="bg-gray-900 border border-gray-700 rounded-lg px-4 py-4 space-y-3">
            {/* Status header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span
                  className={`inline-block w-2 h-2 rounded-full ${
                    result.success ? 'bg-green-400' : 'bg-red-400'
                  }`}
                />
                <span className="text-sm font-medium text-white">
                  {result.success ? 'Success' : 'Failed'}
                </span>
              </div>
              <span className="text-xs text-gray-500 font-mono">
                {result.run_id.slice(0, 8)}...
              </span>
            </div>

            {/* Scenario info */}
            <div className="text-xs text-gray-400">
              Scenario: <span className="text-gray-200">{result.scenario_id}</span>
              <span className="mx-2 text-gray-600">|</span>
              Mode: <span className="text-gray-200">{result.benchmark_mode}</span>
            </div>

            {/* Metrics grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-gray-500">LLM</span>
                <span className="text-gray-200">{fmtMs(result.llm_latency_ms)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">TTS</span>
                <span className="text-gray-200">{fmtMs(result.tts_latency_ms)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Total</span>
                <span className="text-blue-300">{fmtMs(result.total_processing_ms)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Tools</span>
                <span className={result.tool_call_match ? 'text-green-400' : 'text-red-400'}>
                  {result.actual_tool_calls}/{result.expected_tool_calls}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Tokens</span>
                <span className="text-gray-200">
                  {result.usage.total_tokens ?? 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Audio</span>
                <span className="text-gray-200">{fmtBytes(result.tts_audio_bytes)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Iterations</span>
                <span className="text-gray-200">{result.iterations}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Providers</span>
                <span className="text-gray-200">
                  {result.llm_provider || '?'}{result.tts_provider ? ` / ${result.tts_provider}` : ''}
                </span>
              </div>
            </div>

            {/* Validation errors */}
            {result.validation_errors.length > 0 && (
              <div className="space-y-1">
                <div className="text-xs text-red-400 font-medium">Validation Errors:</div>
                {result.validation_errors.map((err, i) => (
                  <div key={i} className="text-xs text-red-300 font-mono bg-red-900/20 rounded px-2 py-1">
                    {err}
                  </div>
                ))}
              </div>
            )}

            {/* Tool calls */}
            {result.tool_calls.length > 0 && (
              <div className="space-y-1">
                <div className="text-xs text-gray-500 font-medium">Tool Calls:</div>
                {result.tool_calls.map((tc, i) => (
                  <div
                    key={i}
                    className="text-xs text-purple-300 font-mono bg-purple-900/20 rounded px-2 py-1"
                  >
                    {tc.function.name}({tc.function.arguments})
                  </div>
                ))}
              </div>
            )}

            {/* Response preview */}
            {result.response_text && (
              <div className="space-y-1">
                <div className="text-xs text-gray-500 font-medium">Response:</div>
                <div className="text-xs text-gray-300 bg-gray-800 rounded px-3 py-2 max-h-32 overflow-y-auto break-words">
                  {result.response_text.length > 500
                    ? result.response_text.slice(0, 500) + '...'
                    : result.response_text}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
