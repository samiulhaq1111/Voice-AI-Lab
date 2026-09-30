/**
 * Focused tests for the RealtimeStt AudioWorklet PCM batching (~50 ms frames).
 *
 * The project has no frontend test runner configured (no vitest/jest), so
 * these tests use Node's built-in test runner and zero new dependencies:
 *
 *     cd frontend && node --test tests/pcm-batching.test.mjs
 *
 * The worklet processor ships as a template literal inside
 * src/features/useRealtimeVoice.ts (loaded at runtime via a Blob URL). This test
 * extracts that exact source and evaluates it against minimal stubs of
 * AudioWorkletGlobalScope, so the real shipped batching code is exercised.
 *
 * Limitation: the main-thread side (handler detach + ws.send guard in
 * useRealtimeVoice.ts) is DOM-dependent and not covered here; only the worklet
 * batching/accumulator logic is unit-tested.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const SOURCE_PATH = join(HERE, '..', 'src', 'features', 'useRealtimeVoice.ts');

const FRAME_SAMPLES = 2400; // 50 ms @ 48 kHz
const FRAME_BYTES = 4800; // 2400 samples x Int16

function extractWorkletSource() {
  const tsx = readFileSync(SOURCE_PATH, 'utf8');
  const match = tsx.match(/const PCM_WORKLET_CODE = `([\s\S]*?)`;/);
  assert.ok(match, 'PCM_WORKLET_CODE template literal not found in useRealtimeVoice.ts');
  return match[1];
}

const WORKLET_SOURCE = extractWorkletSource();

/** Evaluate the worklet source against minimal AudioWorkletGlobalScope stubs. */
function loadProcessor() {
  const posted = [];

  globalThis.sampleRate = 48000;
  globalThis.AudioWorkletProcessor = class {
    constructor() {
      this.port = {
        postMessage: (msg) => posted.push(msg),
        onmessage: null,
        close: () => {},
      };
    }
  };
  let ProcessorClass = null;
  globalThis.registerProcessor = (name, cls) => {
    if (name === 'pcm-processor') ProcessorClass = cls;
  };

  new Function(WORKLET_SOURCE)();
  assert.ok(ProcessorClass, 'registerProcessor did not register pcm-processor');

  return { processor: new ProcessorClass(), posted };
}

/** Mirror of the worklet Float32 -> Int16 conversion (same expression). */
function toInt16(values) {
  const out = new Int16Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const s = Math.max(-1, Math.min(1, values[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

/** Deterministic signal; values are exact binary fractions (no clamping). */
function signal(start, length) {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = (((start + i) % 16384) - 4096) / 32768;
  }
  return out;
}

/** Feed `total` samples as render quanta of `quantumSize` (default 128). */
function feedQuanta(processor, start, total, quantumSize = 128) {
  let fed = 0;
  while (fed < total) {
    const size = Math.min(quantumSize, total - fed);
    processor.process([[signal(start + fed, size)]]);
    fed += size;
  }
}

function frameSamplesAt(posted, index) {
  return Array.from(new Int16Array(posted[index].buffer));
}

test('1. 128-sample render quanta accumulate without emitting a frame', () => {
  const { processor, posted } = loadProcessor();
  for (let q = 0; q < 18; q++) {
    processor.process([[signal(q * 128, 128)]]); // 18 x 128 = 2304 samples
  }
  assert.equal(posted.length, 0, 'no frame must be posted below 2400 samples');
});

test('2. exactly 2400 samples produce one 4800-byte frame', () => {
  const { processor, posted } = loadProcessor();
  feedQuanta(processor, 0, FRAME_SAMPLES); // 18 quanta + 96-sample remainder

  assert.equal(posted.length, 1);
  const msg = posted[0];
  assert.equal(msg.type, 'pcm_frame');
  assert.equal(msg.samples, FRAME_SAMPLES);
  assert.equal(msg.bytes, FRAME_BYTES);
  assert.equal(msg.durationMs, 50);
  assert.equal(msg.buffer.byteLength, FRAME_BYTES);
  assert.equal(msg.frame, 1);
  assert.equal(msg.quanta, 19); // pcm_quanta_received counter
  assert.deepEqual(frameSamplesAt(posted, 0), Array.from(toInt16(signal(0, FRAME_SAMPLES))));
});

test('3. remainder samples are preserved across process() callbacks', () => {
  const { processor, posted } = loadProcessor();
  const firstBatch = 2500; // one frame (2400) + 100-sample remainder
  feedQuanta(processor, 0, firstBatch);
  assert.equal(posted.length, 1, 'only one full frame emitted');

  feedQuanta(processor, firstBatch, FRAME_SAMPLES - 100); // 100 + 2300 = 2400
  assert.equal(posted.length, 2, 'preserved remainder completes the next frame');
  assert.deepEqual(frameSamplesAt(posted, 1), Array.from(toInt16(signal(2400, FRAME_SAMPLES))));
});

test('4/5. no samples are duplicated or lost across many quanta', () => {
  const { processor, posted } = loadProcessor();
  const total = 10000;
  feedQuanta(processor, 0, total);

  const fullFrames = Math.floor(total / FRAME_SAMPLES); // 4
  assert.equal(posted.length, fullFrames);
  for (const msg of posted) {
    assert.equal(msg.samples, FRAME_SAMPLES);
    assert.equal(msg.bytes, FRAME_BYTES);
  }

  const flattened = posted.flatMap((m) => Array.from(new Int16Array(m.buffer)));
  assert.equal(flattened.length, fullFrames * FRAME_SAMPLES);
  assert.deepEqual(flattened, Array.from(toInt16(signal(0, fullFrames * FRAME_SAMPLES))));

  // The remaining 400 samples complete the next frame exactly.
  feedQuanta(processor, total, 2000);
  assert.equal(posted.length, fullFrames + 1);
  assert.deepEqual(
    frameSamplesAt(posted, fullFrames),
    Array.from(toInt16(signal(9600, FRAME_SAMPLES))),
  );
});

test('6. STOP reset clears the accumulator — no pre-STOP samples survive', () => {
  const { processor, posted } = loadProcessor();
  feedQuanta(processor, 0, 1000); // partial frame, under 2400
  assert.equal(posted.length, 0);

  processor.port.onmessage({ data: { type: 'reset' } }); // STOP path

  // 1000 (pre-reset) + 1400 would complete 2400 if the reset failed.
  feedQuanta(processor, 1000, 1400);
  assert.equal(posted.length, 0, 'pre-reset samples must not survive the reset');

  // From a cleared accumulator the next 2400 samples form exactly the frame
  // [1000, 3400) — proving clear + no duplication + no loss around reset.
  feedQuanta(processor, 2400, 1000);
  assert.equal(posted.length, 1);
  assert.deepEqual(frameSamplesAt(posted, 0), Array.from(toInt16(signal(1000, FRAME_SAMPLES))));
});

test('7. no partial frame is ever emitted after STOP', () => {
  const { processor, posted } = loadProcessor();
  processor.port.onmessage({ data: { type: 'reset' } });
  feedQuanta(processor, 0, FRAME_SAMPLES - 1); // 2399 samples — one short
  assert.equal(posted.length, 0, 'partial remainder must never be flushed');
});
