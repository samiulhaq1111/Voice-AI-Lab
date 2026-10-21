/** Unified TTS dropdown: selection ↔ realtime protocol mapping.
 *
 * The realtime protocol separates the TTS *transport* (`tts_mode`) from the
 * server-side *provider* (`tts_provider`). `tts_mode: "elevenlabs"` means
 * "synthesize on the server"; the provider field names which adapter. Browser
 * TTS is purely client-side and therefore never sends a provider.
 */

import type {
  RealtimeTtsMode,
  RealtimeTtsProvider,
  RealtimeTtsSelection,
} from '../types';

export interface TtsSelectionOption {
  value: RealtimeTtsSelection;
  label: string;
}

/** Dropdown order: existing default first, then Azure, then browser. */
export const TTS_SELECTION_OPTIONS: readonly TtsSelectionOption[] = [
  { value: 'elevenlabs', label: 'ElevenLabs' },
  { value: 'azure', label: 'Azure Speech' },
  { value: 'browser', label: 'Browser TTS' },
];

/** START-message fields for one dropdown selection.
 *
 * Browser TTS deliberately returns no `tts_provider`: the backend must not
 * create or route through a server adapter for browser speech.
 */
export function ttsStartFields(selection: RealtimeTtsSelection): {
  tts_mode: RealtimeTtsMode;
  tts_provider?: RealtimeTtsProvider;
} {
  if (selection === 'browser') {
    return { tts_mode: 'browser' };
  }
  return { tts_mode: 'elevenlabs', tts_provider: selection };
}

/** Derive the dropdown value from the engine's mode + provider state. */
export function ttsSelectionOf(
  mode: RealtimeTtsMode,
  provider: RealtimeTtsProvider,
): RealtimeTtsSelection {
  return mode === 'browser' ? 'browser' : provider;
}
