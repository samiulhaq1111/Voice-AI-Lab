/** MCP panel for the Voice Chat screen (runtime enable/disable + details).
 *
 * Compact by design: one control row (switch, ACTUAL session badge, server
 * label, tool count, details toggle) that sits above the conversation
 * without disturbing the provider selectors. The switch reflects the
 * backend's desired state; the badge reflects the live session. Nothing is
 * claimed optimistically: every toggle response replaces the local status
 * with the backend payload, and a failed enable shows "Startup failed"
 * with the sanitized backend error.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { MCPState, MCPStatus } from '../types';
import { disableMCP, enableMCP, getMCPStatus } from '../services/api';
import {
  isMCPActive,
  isToggleBusy,
  nextToggleAction,
  phaseLabel,
  toggleTitle,
  type MCPTogglePhase,
} from './mcpToggle';

const STATE_LABEL: Record<MCPState, string> = {
  disabled: 'Disabled',
  connected: 'Connected',
  failed: 'Startup failed',
  not_started: 'Enabled — no live session',
};

const STATE_CLASS: Record<MCPState, string> = {
  disabled: 'bg-gray-800 border-gray-700 text-gray-400',
  connected: 'bg-green-900/30 border-green-700 text-green-300',
  failed: 'bg-red-900/30 border-red-700 text-red-300',
  not_started: 'bg-yellow-900/30 border-yellow-700 text-yellow-300',
};

export default function MCPStatus() {
  const [status, setStatus] = useState<MCPStatus | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [phase, setPhase] = useState<MCPTogglePhase>('idle');
  const [detailsOpen, setDetailsOpen] = useState(false);
  const aliveRef = useRef(true);

  // Load on mount (Voice Chat opens) — same fetch-on-mount convention as
  // the other panels; no polling loop is needed for a local toggle.
  useEffect(() => {
    aliveRef.current = true;
    getMCPStatus()
      .then((next) => {
        if (!aliveRef.current) return;
        setStatus(next);
        setLoadError(null);
      })
      .catch((err: Error) => {
        if (aliveRef.current) setLoadError(err.message);
      });
    return () => {
      aliveRef.current = false;
    };
  }, []);

  /** Re-read the backend state (used to reconcile after a failed toggle). */
  const refresh = useCallback(async () => {
    try {
      const next = await getMCPStatus();
      if (!aliveRef.current) return;
      setStatus(next);
      setLoadError(null);
    } catch (err) {
      if (aliveRef.current) setLoadError((err as Error).message);
    }
  }, []);

  const handleToggle = useCallback(async () => {
    // Serialize requests: ignore clicks while one is in flight.
    if (!status || isToggleBusy(phase)) return;
    const action = nextToggleAction(status);
    setPhase(action === 'enable' ? 'enabling' : 'disabling');
    setActionError(null);
    try {
      // The response IS the backend's actual resulting state.
      const next = action === 'enable' ? await enableMCP() : await disableMCP();
      if (aliveRef.current) setStatus(next);
    } catch (err) {
      if (aliveRef.current) setActionError((err as Error).message);
      await refresh();
    } finally {
      if (aliveRef.current) setPhase('idle');
    }
  }, [phase, status, refresh]);

  const busy = isToggleBusy(phase);
  const active = isMCPActive(status);

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs">
        <span className="text-gray-400">MCP:</span>
        <button
          type="button"
          role="switch"
          aria-checked={status?.enabled ?? false}
          aria-label={toggleTitle(status)}
          title={toggleTitle(status)}
          onClick={() => void handleToggle()}
          disabled={!status || busy}
          className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors disabled:opacity-50 ${
            status?.enabled ? 'bg-blue-600 border-blue-500' : 'bg-gray-800 border-gray-700'
          }`}
        >
          <span
            className={`inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform ${
              status?.enabled ? 'translate-x-[1.2rem]' : 'translate-x-[0.2rem]'
            }`}
          />
        </button>
        {status ? (
          <>
            <span
              aria-live="polite"
              title={
                active ? 'Live MCP session held by the backend' : 'No live MCP session'
              }
              className={`px-2 py-0.5 rounded border ${STATE_CLASS[status.state]}`}
            >
              {STATE_LABEL[status.state]}
            </span>
            {status.enabled && (
              <span className="text-gray-500">server: {status.server_label}</span>
            )}
            <span className="text-gray-500">
              {status.mcp_tools.length} MCP tool
              {status.mcp_tools.length === 1 ? '' : 's'}
            </span>
            <button
              type="button"
              onClick={() => setDetailsOpen((open) => !open)}
              aria-expanded={detailsOpen}
              className="ml-auto px-2 py-0.5 rounded border border-gray-700 text-gray-300 hover:bg-gray-800"
            >
              {detailsOpen ? 'Hide details' : 'Details'}
            </button>
          </>
        ) : (
          <span className="text-gray-500">
            {loadError ? `MCP status unavailable: ${loadError}` : 'Loading MCP status…'}
          </span>
        )}
        {phaseLabel(phase) && (
          <span className="text-yellow-300 animate-pulse">{phaseLabel(phase)}</span>
        )}
      </div>

      {actionError && (
        <div className="bg-red-900/30 border border-red-700 rounded px-2 py-1 text-xs text-red-300">
          MCP toggle failed: {actionError}
        </div>
      )}

      {detailsOpen && status && (
        <div className="bg-gray-950/60 border border-gray-800 rounded-lg p-2.5 space-y-2">
          {status.error && (
            <div className="bg-red-900/30 border border-red-700 rounded px-2 py-1 text-xs text-red-300">
              {status.error}
            </div>
          )}

          {status.mcp_tools.length === 0 ? (
            <p className="text-xs text-gray-500">
              No MCP tools registered{status.enabled ? '' : ' (MCP is disabled)'}.
            </p>
          ) : (
            <ul className="space-y-1.5">
              {status.mcp_tools.map((tool) => (
                <li key={tool.name} className="text-xs">
                  <span className="font-mono text-blue-400">{tool.name}</span>
                  <span className="text-gray-500"> (MCP tool: {tool.mcp_tool})</span>
                  <p className="text-gray-400">{tool.description}</p>
                </li>
              ))}
            </ul>
          )}

          <p className="text-[11px] text-gray-500">
            Native tools ({status.native_tools.length}):{' '}
            {status.native_tools.join(', ') || 'none'}
          </p>
        </div>
      )}
    </div>
  );
}
