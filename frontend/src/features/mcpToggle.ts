/** Pure MCP toggle/status helpers for the Voice Chat MCP panel.
 *
 * Kept free of React so the shipped logic is exercised directly by
 * tests/mcpToggle.test.mjs (the project's node:test + tsc convention).
 * Everything derives from the backend status payload — nothing here
 * assumes an enable attempt succeeded.
 */

import type { MCPStatus } from '../types';

export type MCPToggleAction = 'enable' | 'disable';
export type MCPTogglePhase = 'idle' | 'enabling' | 'disabling';

/** True only when the backend reports BOTH enabled and a live session. */
export function isMCPActive(status: MCPStatus | null): boolean {
  return status !== null && status.enabled && status.connected;
}

/** The switch shows the desired state; a failed enable stays ON so it can
 * be switched back off. */
export function nextToggleAction(status: MCPStatus | null): MCPToggleAction {
  return status?.enabled ? 'disable' : 'enable';
}

/** Requests are serialized: the toggle is inert while one is in flight. */
export function isToggleBusy(phase: MCPTogglePhase): boolean {
  return phase !== 'idle';
}

/** Short label for the in-flight state ('' when idle). */
export function phaseLabel(phase: MCPTogglePhase): string {
  if (phase === 'enabling') return 'Enabling…';
  if (phase === 'disabling') return 'Disabling…';
  return '';
}

/** Accessible switch description derived from the backend-reported state. */
export function toggleTitle(status: MCPStatus | null): string {
  if (!status) return 'MCP status unavailable';
  if (status.enabled) return 'Disable MCP tools for new agent calls';
  return 'Enable MCP tools from the configured local server';
}
