/**
 * MCP toggle/status helpers (Voice Chat panel).
 *
 * The project has no React test runner, so the pure module
 * (src/features/mcpToggle.ts) is compiled with the installed TypeScript
 * compiler and exercised directly — the real shipped logic, not a copy:
 *
 *     cd frontend && node --test tests/mcpToggle.test.mjs
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = join(HERE, '..');
const SRC_DIR = join(FRONTEND_ROOT, 'src');
const SOURCE = join(SRC_DIR, 'features', 'mcpToggle.ts');
const BUILD_DIR = join(HERE, '.build');
const TSC = join(FRONTEND_ROOT, 'node_modules', 'typescript', 'bin', 'tsc');

execFileSync(
  process.execPath,
  [
    TSC,
    '--module',
    'esnext',
    '--moduleResolution',
    'bundler',
    '--target',
    'es2022',
    '--skipLibCheck',
    '--rootDir',
    SRC_DIR,
    '--outDir',
    BUILD_DIR,
    SOURCE,
  ],
  { cwd: FRONTEND_ROOT, stdio: 'pipe' },
);

const { isMCPActive, nextToggleAction, isToggleBusy, phaseLabel, toggleTitle } =
  await import(pathToFileURL(join(BUILD_DIR, 'features', 'mcpToggle.js')).href);

/** Backend status payload factory (disabled with no tools by default). */
function status(overrides = {}) {
  return {
    enabled: false,
    connected: false,
    state: 'disabled',
    server_label: 'local',
    error: null,
    mcp_tools: [],
    native_tools: ['get_weather', 'get_employee', 'get_leave_balance'],
    ...overrides,
  };
}

test('1. active only when the backend reports enabled AND a live session', () => {
  assert.equal(isMCPActive(null), false);
  assert.equal(isMCPActive(status()), false);
  // Enabled but failed startup: the switch is ON, the session is NOT live.
  assert.equal(
    isMCPActive(
      status({
        enabled: true,
        connected: false,
        state: 'failed',
        error: 'MCP startup failed (RuntimeError)',
      }),
    ),
    false,
  );
  assert.equal(isMCPActive(status({ enabled: true, connected: true, state: 'connected' })), true);
});

test('2. the switch reflects the desired state, not the connection', () => {
  assert.equal(nextToggleAction(null), 'enable');
  assert.equal(nextToggleAction(status()), 'enable');
  // A failed enable stays ON so the user can switch it back off.
  assert.equal(nextToggleAction(status({ enabled: true, state: 'failed' })), 'disable');
  assert.equal(nextToggleAction(status({ enabled: true, connected: true, state: 'connected' })), 'disable');
});

test('3. requests are serialized while one is in flight', () => {
  assert.equal(isToggleBusy('idle'), false);
  assert.equal(isToggleBusy('enabling'), true);
  assert.equal(isToggleBusy('disabling'), true);
});

test('4. in-flight labels match the phase', () => {
  assert.equal(phaseLabel('idle'), '');
  assert.equal(phaseLabel('enabling'), 'Enabling…');
  assert.equal(phaseLabel('disabling'), 'Disabling…');
});

test('5. switch description follows backend state', () => {
  assert.equal(toggleTitle(null), 'MCP status unavailable');
  assert.match(toggleTitle(status()), /Enable MCP tools/);
  assert.match(toggleTitle(status({ enabled: true })), /Disable MCP tools/);
});
