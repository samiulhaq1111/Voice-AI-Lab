/** Text Chat — minimal development/testing UI for the agent loop. */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { ChatResponse, ProviderAvailability, ToolCallInfo } from '../types';
import { getProviders, sendChat } from '../services/api';

interface DisplayMessage {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  toolCalls?: ToolCallInfo[];
  usage?: Record<string, number>;
  iterations?: number;
}

export default function TextChat() {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [providers, setProviders] = useState<ProviderAvailability | null>(null);
  const [selectedProvider, setSelectedProvider] = useState('openrouter');
  const [model, setModel] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getProviders()
      .then(setProviders)
      .catch(() => {});
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput('');
    setError(null);
    setLoading(true);
    setMessages((prev) => [...prev, { role: 'user', content: text }]);

    try {
      const res: ChatResponse = await sendChat({
        message: text,
        session_id: sessionId,
        provider: selectedProvider || null,
        model: model || null,
      });

      setSessionId(res.session_id);

      const assistantMsg: DisplayMessage = {
        role: 'assistant',
        content: res.response,
        toolCalls: res.tool_calls,
        usage: res.usage,
        iterations: res.iterations,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [input, loading, sessionId, selectedProvider, model]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleReset = () => {
    setMessages([]);
    setSessionId(null);
    setError(null);
  };

  // Split the shared catalogue into free and paid groups for the dropdown.
  const llmProv = providers?.llm.find((p) => p.provider === selectedProvider);
  const paidModels = llmProv?.paid_models ?? [];
  const freeModels = (llmProv?.models ?? []).filter((m) => !paidModels.includes(m));

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h2 className="text-lg font-semibold text-white">Text Chat (Dev UI)</h2>
        <div className="flex items-center gap-3">
          {sessionId && (
            <span className="text-xs text-gray-500 font-mono">
              Session: {sessionId.slice(0, 8)}...
            </span>
          )}
          <button
            onClick={handleReset}
            className="text-xs px-2 py-1 bg-gray-800 text-gray-300 rounded hover:bg-gray-700"
          >
            New Session
          </button>
        </div>
      </div>

      {/* Provider/Model Selection */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-gray-800 bg-gray-900/50">
        <label className="text-xs text-gray-400">LLM:</label>
        <select
          value={selectedProvider}
          onChange={(e) => setSelectedProvider(e.target.value)}
          className="text-xs bg-gray-800 text-gray-200 rounded px-2 py-1 border border-gray-700"
        >
          {providers?.llm.map((p) => (
            <option key={p.provider} value={p.provider}>
              {p.provider} {p.configured ? '(configured)' : '(not configured)'}
            </option>
          )) || <option value="openrouter">openrouter</option>}
        </select>
        <label className="text-xs text-gray-400">Model:</label>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="text-xs bg-gray-800 text-gray-200 rounded px-2 py-1 border border-gray-700"
        >
          <option value="">
            default ({llmProv?.default_model || 'nvidia/nemotron-3.5-lightning:free'})
          </option>
          {freeModels.length > 0 && (
            <optgroup label="FREE / EXPERIMENTAL">
              {freeModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </optgroup>
          )}
          {paidModels.length > 0 && (
            <optgroup label="PAID / PAYG">
              {paidModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </optgroup>
          )}
        </select>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-600 mt-12">
            Send a message to start the agent loop.
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i}>
            {/* Main message bubble */}
            <div
              className={`rounded-lg px-4 py-2 max-w-[85%] ${
                msg.role === 'user'
                  ? 'ml-auto bg-blue-900/40 border border-blue-800 text-blue-100'
                  : msg.role === 'tool'
                    ? 'bg-yellow-900/30 border border-yellow-800 text-yellow-200 text-sm'
                    : 'bg-gray-800 border border-gray-700 text-gray-100'
              }`}
            >
              <div className="text-xs text-gray-500 mb-1 font-medium uppercase">
                {msg.role}
              </div>
              <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
            </div>

            {/* Tool calls display */}
            {msg.toolCalls && msg.toolCalls.length > 0 && (
              <div className="mt-1 ml-2 space-y-1">
                {msg.toolCalls.map((tc, j) => (
                  <div
                    key={j}
                    className="text-xs bg-orange-900/20 border border-orange-800/50 rounded px-2 py-1 text-orange-300 font-mono"
                  >
                    Tool: {tc.function.name}({tc.function.arguments})
                  </div>
                ))}
              </div>
            )}

            {/* Usage/iterations for assistant messages */}
            {msg.role === 'assistant' && msg.iterations !== undefined && (
              <div className="mt-1 text-xs text-gray-600">
                {msg.iterations} iteration{msg.iterations !== 1 ? 's' : ''}
                {msg.usage?.total_tokens ? ` · ${msg.usage.total_tokens} tokens` : ''}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 max-w-[85%]">
            <div className="text-xs text-gray-500 mb-1 font-medium uppercase">assistant</div>
            <div className="text-sm text-gray-400 animate-pulse">Thinking...</div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Error */}
      {error && (
        <div className="mx-4 mb-2 bg-red-900/30 border border-red-700 rounded-lg px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Input */}
      <div className="px-4 py-3 border-t border-gray-800">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message... (Enter to send)"
            rows={1}
            className="flex-1 bg-gray-800 text-gray-100 rounded-lg px-3 py-2 text-sm border border-gray-700 focus:border-blue-600 focus:outline-none resize-none"
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
