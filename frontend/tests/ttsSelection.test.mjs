/**
 * Unified TTS dropdown mapping (Voice Chat).
 *
 * The project has no React test runner, so the pure mapping module
 * (src/features/ttsSelection.ts) is compiled with the installed TypeScript
 * compiler and exercised directly — the real shipped module, not a copy:
 *
 *     cd frontend && node --test tests/ttsSelection.test.mjs
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = join(HERE, '..');
const SRC_DIR = join(FRONTEND_ROOT, 'src');
const SOURCE = join(SRC_DIR, 'features', 'ttsSelection.ts');
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

const { TTS_SELECTION_OPTIONS, ttsSelectionOf, ttsStartFields } = await import(
  pathToFileURL(join(BUILD_DIR, 'features', 'ttsSelection.js')).href
);

test('1. dropdown lists ElevenLabs, Azure Speech, Browser TTS in order', () => {
  assert.deepEqual(
    TTS_SELECTION_OPTIONS.map((o) => [o.value, o.label]),
    [
      ['elevenlabs', 'ElevenLabs'],
      ['azure', 'Azure Speech'],
      ['browser', 'Browser TTS'],
    ],
  );
  // Existing behavior preserved: ElevenLabs stays the default (first option).
  assert.equal(TTS_SELECTION_OPTIONS[0].value, 'elevenlabs');
});

test('2. server selections map to server transport + provider', () => {
  assert.deepEqual(ttsStartFields('elevenlabs'), {
    tts_mode: 'elevenlabs',
    tts_provider: 'elevenlabs',
  });
  assert.deepEqual(ttsStartFields('azure'), {
    tts_mode: 'elevenlabs',
    tts_provider: 'azure',
  });
});

test('3. browser selection is client-side and sends no server provider', () => {
  const fields = ttsStartFields('browser');
  assert.equal(fields.tts_mode, 'browser');
  assert.equal('tts_provider' in fields, false);
});

test('4. dropdown value derives from engine mode + provider state', () => {
  assert.equal(ttsSelectionOf('browser', 'azure'), 'browser');
  assert.equal(ttsSelectionOf('elevenlabs', 'azure'), 'azure');
  assert.equal(ttsSelectionOf('elevenlabs', 'elevenlabs'), 'elevenlabs');
});
