/**
 * T11 (Phase 6E): the realtime conversation UI keeps ONE evolving current
 * user turn — interim partials replace in place and are never appended as
 * conversation messages. Exactly one user message is committed per dispatched
 * turn, immediately followed by the assistant reply.
 *
 * The project has no TS test runner (no vitest/jest), so the pure model
 * module (src/features/realtimeTurnModel.ts) is compiled once with the
 * already-installed TypeScript compiler and the emitted JS is exercised
 * directly — the real shipped reducer, not a copy:
 *
 *     cd frontend && node --test tests/realtimeTurnModel.test.mjs
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = join(HERE, '..');
const FEATURES_DIR = join(FRONTEND_ROOT, 'src', 'features');
const SOURCE = join(FEATURES_DIR, 'realtimeTurnModel.ts');
const BUILD_DIR = join(HERE, '.build');
const TSC = join(FRONTEND_ROOT, 'node_modules', 'typescript', 'bin', 'tsc');

execFileSync(
  process.execPath,
  [
    TSC,
    '--module',
    'es2022',
    '--target',
    'es2022',
    '--skipLibCheck',
    '--rootDir',
    FEATURES_DIR,
    '--outDir',
    BUILD_DIR,
    SOURCE,
  ],
  { cwd: FRONTEND_ROOT, stdio: 'pipe' },
);

const { conversationReducer, createConversationState, currentTurnText } =
  await import(pathToFileURL(join(BUILD_DIR, 'realtimeTurnModel.js')).href);

function apply(state, actions) {
  return actions.reduce(conversationReducer, state);
}

test('1. partials replace in place — one evolving turn, no committed entries', () => {
  const s = apply(createConversationState(), [
    { type: 'partial', text: 'Tell me the' },
    { type: 'partial', text: 'Tell me the details of' },
    { type: 'partial', text: 'Tell me the details of employee ID' },
  ]);
  assert.equal(s.entries.length, 0, 'partials must never commit messages');
  assert.equal(currentTurnText(s), 'Tell me the details of employee ID');
});

test('2. multiple final segments accumulate into ONE current turn', () => {
  const s = apply(createConversationState(), [
    { type: 'final', text: 'Tell me the details of employee ID.' },
    { type: 'final', text: 'E zero zero one.' },
  ]);
  assert.equal(s.entries.length, 0, 'finals alone must not commit messages');
  assert.equal(
    currentTurnText(s),
    'Tell me the details of employee ID. E zero zero one.',
  );
});

test('3. dispatch commits exactly ONE user message, then the assistant reply', () => {
  const s = apply(createConversationState(), [
    { type: 'partial', text: 'Tell me the' },
    { type: 'partial', text: 'Tell me the details of employee ID' },
    { type: 'final', text: 'Tell me the details of employee ID.' },
    { type: 'partial', text: 'E zero zero' },
    { type: 'partial', text: 'E zero zero one' },
    { type: 'final', text: 'E zero zero one.' },
    { type: 'agent_processing' },
    { type: 'agent_response', text: 'Here you go.', toolCalls: 1, iterations: 2 },
  ]);
  assert.equal(s.entries.length, 2);
  assert.deepEqual(s.entries[0], {
    role: 'user',
    text: 'Tell me the details of employee ID. E zero zero one.',
  });
  assert.deepEqual(s.entries[1], {
    role: 'assistant',
    text: 'Here you go.',
    toolCalls: 1,
    iterations: 2,
  });
  assert.equal(s.agentProcessing, false);
  assert.equal(currentTurnText(s), '', 'the live turn is consumed at dispatch');
});

test('4. dispatch with no transcript does not create an empty message', () => {
  const s = apply(createConversationState(), [{ type: 'agent_processing' }]);
  assert.equal(s.entries.length, 0);
  assert.equal(s.agentProcessing, true);
});

test('5. a second turn appends after the first — turns never merge', () => {
  const s = apply(createConversationState(), [
    { type: 'final', text: 'First question.' },
    { type: 'agent_processing' },
    { type: 'agent_response', text: 'First answer.' },
    { type: 'partial', text: 'Second' },
    { type: 'final', text: 'Second question.' },
    { type: 'agent_processing' },
    { type: 'agent_response', text: 'Second answer.' },
  ]);
  assert.deepEqual(
    s.entries.map((entry) => `${entry.role}:${entry.text}`),
    [
      'user:First question.',
      'assistant:First answer.',
      'user:Second question.',
      'assistant:Second answer.',
    ],
  );
});

test('6. agent failure stops the in-flight indicator, keeps history', () => {
  const s = apply(createConversationState(), [
    { type: 'final', text: 'Will fail.' },
    { type: 'agent_processing' },
    { type: 'agent_failed' },
  ]);
  assert.equal(s.agentProcessing, false);
  assert.equal(s.entries.length, 1);
  assert.equal(s.entries[0].text, 'Will fail.');
});

test('7. reset clears the conversation and the live turn', () => {
  const s = apply(createConversationState(), [
    { type: 'final', text: 'partial turn' },
    { type: 'partial', text: 'more' },
    { type: 'reset' },
  ]);
  assert.deepEqual(s, createConversationState());
});

test('8. empty final text does not pollute the turn', () => {
  const s = apply(createConversationState(), [{ type: 'final', text: '' }]);
  assert.equal(s.pendingFinals.length, 0);
  assert.equal(currentTurnText(s), '');
});
