# Voice AI Lab - Architecture

## Overview

Voice AI Lab is a browser-based voice AI platform designed to be provider-independent. The architecture uses abstraction interfaces and adapters so that no part of the system directly depends on a specific AI provider.

## System Architecture

```
Browser (React + TypeScript)
    |
    | WebRTC / HTTP
    v
FastAPI Voice Gateway
    |
    v
STT Adapter (STTInterface)
    |
    v
Agent Runtime
    |
    v
LLM Adapter (LLMInterface)
    |
    v
Tool Calling (if required)
    |
    v
Tool Registry / Tool Executor
    |
    v
Backend functions / external APIs
    |
    v
Tool Result -> Agent Runtime -> LLM response
    |
    v
TTS Adapter (TTSInterface)
    |
    v
Browser audio playback
```

## Provider Abstraction

All AI providers implement abstract interfaces:

```
STTInterface
    +-- DeepgramAdapter (implemented)
    +-- OpenAIAdapter (future)
    +-- GoogleAdapter (future)

LLMInterface
    +-- OpenRouterAdapter (implemented)
    +-- OpenAIAdapter (future)
    +-- GeminiAdapter (future)
    +-- AnthropicAdapter (future)

TTSInterface
    +-- ElevenLabsAdapter (implemented)
    +-- OpenAIAdapter (future)
    +-- GoogleAdapter (future)
```

The Agent Runtime depends ONLY on interfaces, never on concrete providers.

## Provider Factory

The provider factory (`backend/app/providers/factory.py`) is the ONLY module that knows about concrete adapters. It:

- Instantiates adapters based on configuration
- Loads API keys from environment variables
- Validates provider names
- Returns interface types to callers

Usage:
```python
from app.providers.factory import get_stt_provider, get_llm_provider, get_tts_provider

stt = get_stt_provider("deepgram")
llm = get_llm_provider("openrouter")
tts = get_tts_provider("elevenlabs")
```

The factory uses lazy imports to avoid loading all adapters at startup.

## Agent Runtime

The Agent Runtime is the core orchestration loop. It:

- Maintains conversation state and message history
- Invokes the LLM with current history + tool schemas
- Detects tool calls in LLM responses
- Executes tool calls via ToolExecutor
- Feeds tool results back to the LLM
- Returns the final text response via `AgentResult`
- Handles errors and enforces iteration limits

### Agent Loop

```
User Message
    |
    v
LLM (via LLMInterface)
    |
    v
Tool calls requested?
    |-- NO  --> Final response (AgentResult)
    |
    |-- YES --> Validate tool (ToolRegistry)
                    |
                    v
               Execute tool (ToolExecutor)
                    |
                    v
               Append tool result to history
                    |
                    v
               Call LLM again
                    |
                    v
               Repeat (up to MAX_AGENT_ITERATIONS)
```

The loop stops when:
- LLM returns a final text response (no tool calls)
- Maximum iteration count is reached
- An unrecoverable provider error occurs

### Provider Independence

AgentRuntime depends ONLY on `LLMInterface`, `ToolRegistry`, and `ToolExecutor`. It never imports concrete adapters (OpenRouterAdapter, DeepgramAdapter, ElevenLabsAdapter). This is verified by the `TestFakeLLMArchitecture` test which uses a `FakeLLM` implementation.

## Tool Architecture

```
Tool (definition)
    - name, description, input schema
    - handler function
    - timeout, enabled/disabled state

ToolRegistry
    - Central registry of all available tools
    - Lookup by name
    - Enable/disable tools

ToolExecutor
    - Executes tool calls with timeout
    - Error handling
    - Structured results
```

Tools are NEVER given direct access to databases, API keys, or external services.

## Database

- **SQLite** for MVP (easily upgradeable to PostgreSQL)
- **SQLAlchemy 2.x** ORM with typed models
- **Alembic** for schema migrations

### Entities

| Entity | Purpose |
|--------|---------|
| VoiceSession | A single voice conversation session |
| Message | Individual messages in a session |
| ToolCall | Records of tool invocations |
| BenchmarkResult | Provider comparison metrics |
| UsageRecord | API usage and cost tracking |
| ProviderConfiguration | Provider settings |

## Technology Stack

### Frontend
- React 19 + TypeScript
- Vite (build tool)
- Tailwind CSS (styling)

### Backend
- Python 3.11+
- FastAPI (web framework)
- Pydantic v2 (validation)
- SQLAlchemy 2.x (ORM)
- Alembic (migrations)

### AI Providers (MVP)
- STT: Deepgram (adapter implemented)
- LLM: OpenRouter (adapter implemented)
- TTS: ElevenLabs (adapter implemented)

## Provider API Endpoint

`GET /api/v1/providers` returns safe metadata about available providers (no API keys):

```json
{
  "stt": [
    {
      "provider": "deepgram",
      "models": ["nova-3"],
      "configured": true
    }
  ],
  "llm": [
    {
      "provider": "openrouter",
      "configured": true
    }
  ],
  "tts": [
    {
      "provider": "elevenlabs",
      "models": [],
      "configured": true
    }
  ]
}
```

## Project Structure

```
voice-ai-lab/
+-- backend/
|   +-- app/
|   |   +-- api/           # FastAPI route handlers
|   |   +-- core/          # Config, database, logging
|   |   +-- agents/        # Agent Runtime (orchestration)
|   |   +-- providers/     # Provider interfaces + adapters
|   |   |   +-- stt/       # Speech-to-text
|   |   |   +-- llm/       # Large language models
|   |   |   +-- tts/       # Text-to-speech
|   |   +-- tools/         # Tool definitions, registry, executor
|   |   +-- models/        # SQLAlchemy models
|   |   +-- schemas/       # Pydantic schemas
|   |   +-- services/      # Business logic services
|   |   +-- main.py        # FastAPI application entry point
|   +-- tests/
|   +-- alembic/           # Database migrations
|
+-- frontend/
|   +-- src/
|   |   +-- components/    # Reusable UI components
|   |   +-- pages/         # Page-level components
|   |   +-- features/      # Feature modules
|   |   +-- hooks/         # Custom React hooks
|   |   +-- services/      # API client
|   |   +-- types/         # TypeScript type definitions
|
+-- docs/                  # Documentation
+-- docker-compose.yml
```

## Design Principles

- **Provider Independence**: No direct provider dependencies in business logic
- **Dependency Inversion**: High-level modules depend on abstractions
- **SOLID Principles**: Clean separation of concerns
- **Async-first**: FastAPI async handlers for I/O-bound operations
- **Type Safety**: TypeScript frontend, typed Python backend
- **Testability**: All components are independently testable
