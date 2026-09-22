/** Voice AI Lab - Main Application */

import { useEffect, useState } from 'react';
import type { HealthStatus } from './types';
import { getHealth } from './services/api';
import TextChat from './features/TextChat';
import VoiceTest from './features/VoiceTest';
import Benchmark from './features/Benchmark';
import TelephoneObservability from './features/TelephoneObservability';

type Tab = 'status' | 'chat' | 'voice' | 'telephone' | 'benchmark';

function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('chat');

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((err: Error) => setError(err.message));
  }, []);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      <header className="border-b border-gray-800 px-6 py-3 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">Voice AI Lab</h1>
          <p className="text-xs text-gray-500">
            {health ? `Backend: ${health.status} (v${health.version})` : 'Phase 5E — Controlled Provider/Model Comparison'}
          </p>
        </div>
        <nav className="flex gap-1">
          <button
            onClick={() => setTab('chat')}
            className={`px-3 py-1.5 rounded text-sm ${
              tab === 'chat'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-gray-200'
            }`}
          >
            Chat
          </button>
          <button
            onClick={() => setTab('voice')}
            className={`px-3 py-1.5 rounded text-sm ${
              tab === 'voice'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-gray-200'
            }`}
          >
            Voice
          </button>
          <button
            onClick={() => setTab('telephone')}
            className={`px-3 py-1.5 rounded text-sm ${
              tab === 'telephone'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-gray-200'
            }`}
          >
            Telephone
          </button>
          <button
            onClick={() => setTab('benchmark')}
            className={`px-3 py-1.5 rounded text-sm ${
              tab === 'benchmark'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-gray-200'
            }`}
          >
            Benchmark
          </button>
          <button
            onClick={() => setTab('status')}
            className={`px-3 py-1.5 rounded text-sm ${
              tab === 'status'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-gray-200'
            }`}
          >
            Status
          </button>
        </nav>
      </header>

      <main className="flex-1 flex flex-col">
        {tab === 'chat' && <TextChat />}
        {tab === 'voice' && <VoiceTest />}
        {tab === 'telephone' && <TelephoneObservability />}
        {tab === 'benchmark' && <Benchmark />}

        {tab === 'status' && (
          <div className="max-w-4xl mx-auto px-6 py-12 w-full">
            <section className="mb-8">
              <h2 className="text-xl font-semibold mb-4">System Status</h2>
              {error && (
                <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 text-red-300">
                  Backend connection error: {error}
                </div>
              )}
              {health && (
                <div className="bg-green-900/30 border border-green-700 rounded-lg p-4 text-green-300">
                  Backend: {health.status} (v{health.version})
                </div>
              )}
            </section>

            <section className="mb-8">
              <h2 className="text-xl font-semibold mb-4">Architecture</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <h3 className="font-medium text-blue-400 mb-2">STT</h3>
                  <p className="text-sm text-gray-400">Deepgram (implemented)</p>
                </div>
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <h3 className="font-medium text-purple-400 mb-2">LLM</h3>
                  <p className="text-sm text-gray-400">OpenRouter (implemented)</p>
                </div>
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <h3 className="font-medium text-green-400 mb-2">TTS</h3>
                  <p className="text-sm text-gray-400">ElevenLabs (implemented)</p>
                </div>
              </div>
            </section>
          </div>
        )}
      </main>

      <footer className="border-t border-gray-800 px-6 py-2 text-center text-xs text-gray-600">
        Voice AI Lab v0.2.0 &mdash; Phase 5E: Controlled Provider/Model Comparison
      </footer>
    </div>
  );
}

export default App;
