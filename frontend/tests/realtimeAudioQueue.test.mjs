/**
 * Phase 6H: the realtime browser path now receives ONE audio segment per
 * completed LLM sentence and must play them strictly in sentence order.
 *
 * The project has no TS test runner (no vitest/jest), so the pure queue
 * module (src/features/realtimeAudioQueue.ts) is compiled once with the
 * already-installed TypeScript compiler and the emitted JS is exercised
 * directly — the real shipped queue, not a copy:
 *
 *     cd frontend && node --test tests/realtimeAudioQueue.test.mjs
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = join(HERE, '..');
const FEATURES_DIR = join(FRONTEND_ROOT, 'src', 'features');
const SOURCE = join(FEATURES_DIR, 'realtimeAudioQueue.ts');
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

const {
  beginNextSegment,
  clearAudioQueue,
  createAudioQueueState,
  endCurrentSegment,
  enqueueSegment,
  markTurnReported,
} = await import(pathToFileURL(join(BUILD_DIR, 'realtimeAudioQueue.js')).href);

function segment(url, turn = 1) {
  return { url, turn, wsTransitMs: 3, receivedAt: 100 };
}

test('1. segments play strictly in arrival (sentence) order', () => {
  let state = createAudioQueueState();
  state = enqueueSegment(state, segment('blob:turn1-s1'));
  state = enqueueSegment(state, segment('blob:turn1-s2'));
  state = enqueueSegment(state, segment('blob:turn1-s3'));

  const first = beginNextSegment(state);
  assert.equal(first.next.url, 'blob:turn1-s1');
  state = endCurrentSegment(first.state).state;

  const second = beginNextSegment(state);
  assert.equal(second.next.url, 'blob:turn1-s2');
  state = endCurrentSegment(second.state).state;

  const third = beginNextSegment(state);
  assert.equal(third.next.url, 'blob:turn1-s3');
  state = endCurrentSegment(third.state).state;

  assert.equal(beginNextSegment(state).next, null, 'queue drained');
});

test('2. a later segment never starts while one is playing', () => {
  let state = createAudioQueueState();
  state = enqueueSegment(state, segment('blob:s1'));
  const first = beginNextSegment(state);
  state = first.state; // now playing

  state = enqueueSegment(state, segment('blob:s2'));
  const blocked = beginNextSegment(state);
  assert.equal(blocked.next, null, 'segment 2 must wait for segment 1');
  assert.equal(blocked.state.pending.length, 1);
});

test('3. ending a segment hands back its object URL for revocation', () => {
  let state = createAudioQueueState();
  state = enqueueSegment(state, segment('blob:revoke-me'));
  state = beginNextSegment(state).state;

  const finished = endCurrentSegment(state);
  assert.equal(finished.revokeUrl, 'blob:revoke-me');
  assert.equal(finished.state.current, null);
});

test('4. the first segment starts immediately on an idle queue', () => {
  const state = enqueueSegment(createAudioQueueState(), segment('blob:first'));
  const promoted = beginNextSegment(state);
  assert.equal(promoted.next.url, 'blob:first');
  assert.equal(promoted.state.current.url, 'blob:first');
  assert.equal(promoted.state.pending.length, 0);
});

test('5. first-playback report is claimed exactly once per turn', () => {
  let state = createAudioQueueState();
  const first = markTurnReported(state, 1);
  assert.equal(first.first, true, 'first segment of the turn reports');
  state = first.state;

  const second = markTurnReported(state, 1);
  assert.equal(second.first, false, 'later sentences must not re-report');

  const nextTurn = markTurnReported(state, 2);
  assert.equal(nextTurn.first, true, 'every turn reports its own first audio');
});

test('6. operations are pure — previous states are never mutated', () => {
  const base = createAudioQueueState();
  const withSegment = enqueueSegment(base, segment('blob:s1'));
  const playing = beginNextSegment(withSegment).state;
  endCurrentSegment(playing);

  assert.deepEqual(base, createAudioQueueState(), 'base state unchanged');
  assert.equal(withSegment.pending.length, 1, 'enqueue state unchanged');
  assert.equal(playing.current !== null, true, 'promoted state unchanged by end');
});

test('7. clearing returns every owned URL and resets to a fresh state', () => {
  let state = createAudioQueueState();
  state = enqueueSegment(state, segment('blob:current'));
  state = beginNextSegment(state).state;
  state = enqueueSegment(state, segment('blob:pending-1'));
  state = enqueueSegment(state, segment('blob:pending-2'));

  const cleared = clearAudioQueue(state);
  assert.deepEqual(
    [...cleared.revokeUrls].sort(),
    ['blob:current', 'blob:pending-1', 'blob:pending-2'],
  );
  assert.deepEqual(cleared.state, createAudioQueueState());
  assert.equal(cleared.state.pending.length, 0, 'stale audio cannot survive');
});

test('8. ending an idle queue is a safe no-op', () => {
  const state = createAudioQueueState();
  const finished = endCurrentSegment(state);
  assert.equal(finished.revokeUrl, null);
  assert.equal(finished.state, state);
});
