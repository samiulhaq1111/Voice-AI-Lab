# Voice AI Lab

A browser-based, provider-independent voice AI platform for real-time voice conversations with AI agents.

## Purpose

Voice AI Lab allows you to:

- Have real-time voice conversations from a web browser
- Convert speech to text using selectable STT providers
- Send text to selectable LLM providers
- Allow the LLM to call backend tools/functions
- Convert responses to speech using selectable TTS providers
- Store conversation/session/benchmark data
- Compare different STT + LLM + TTS provider combinations

The goal is to avoid vendor lock-in to any single voice AI provider (Vapi, Retell, Bland, etc.).

## Architecture

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
Agent Runtime (orchestration loop)
    |
    v
LLM Adapter (LLMInterface)
    |
    v
Tool Calling -> Tool Registry -> Tool Executor
    |
    v
TTS Adapter (TTSInterface)
    |
    v
Browser audio playback
```

All AI providers implement abstract interfaces. The Agent Runtime depends on interfaces, NOT concrete providers.

See [docs/architecture.md](docs/architecture.md) for full details.

## Technology Stack

### Frontend
- React + TypeScript
- Vite
- Tailwind CSS

### Backend
- Python 3.11+
- FastAPI
- Pydantic v2
- SQLAlchemy 2.x
- Alembic

### Database
- SQLite (MVP)

### AI Providers (MVP)
- **STT**: Deepgram
- **LLM**: OpenRouter
- **TTS**: ElevenLabs

## Project Structure

```
voice-ai-lab/
+-- backend/           # FastAPI backend
|   +-- app/
|   |   +-- api/       # Route handlers
|   |   +-- core/      # Config, database, logging
|   |   +-- agents/    # Agent Runtime
|   |   +-- providers/ # STT/LLM/TTS interfaces + adapters
|   |   +-- tools/     # Tool system
|   |   +-- models/    # SQLAlchemy models
|   |   +-- schemas/   # Pydantic schemas
|   |   +-- services/  # Business logic
|   +-- tests/
|   +-- alembic/       # Migrations
|
+-- frontend/          # React frontend
|   +-- src/
|
+-- docs/              # Documentation
+-- docker-compose.yml
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys
uvicorn app.main:app --reload
```

Backend: http://localhost:8000
API Docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

### Docker

```bash
docker compose up
```

## Provider Abstraction

All AI providers implement abstract interfaces (STTInterface, LLMInterface, TTSInterface). Concrete adapters are instantiated via a provider factory:

```python
from app.providers.factory import get_stt_provider, get_llm_provider, get_tts_provider

stt = get_stt_provider("deepgram")
llm = get_llm_provider("openrouter")
tts = get_tts_provider("elevenlabs")
```

The Agent Runtime depends ONLY on interfaces, never on concrete providers.

### Current Adapters

| Type | Provider | Status |
|------|----------|--------|
| STT | Deepgram | Implemented |
| LLM | OpenRouter | Implemented |
| TTS | ElevenLabs | Implemented |

### Provider API

`GET /api/v1/providers` returns available provider metadata (no API keys exposed).

## Current MVP Scope

- Project foundation and architecture
- Provider abstraction interfaces (STT, LLM, TTS)
- Provider adapters (Deepgram, OpenRouter, ElevenLabs)
- Provider factory with environment-driven configuration
- Tool system (registry, executor, demo tools)
- Agent Runtime with bounded tool-call loop
- Text chat endpoint (`POST /api/v1/chat`)
- Session, message, tool call, and usage persistence
- SQLAlchemy models and Alembic migrations
- FastAPI health, tools, providers, and chat endpoints
- React frontend with text chat testing UI
- Database schema for sessions, messages, tool calls, benchmarks

## Text Chat API

`POST /api/v1/chat` sends a message through the agent loop:

```json
// Request
{ "message": "What is the weather in Islamabad?", "session_id": null, "provider": "openrouter", "model": null }

// Response
{ "response": "...", "session_id": "...", "tool_calls": [...], "usage": {...}, "iterations": 2 }
```

The agent loop: User message -> LLM -> Tool call? -> Execute -> LLM again -> Final response. Maximum iterations configurable via `MAX_AGENT_ITERATIONS` (default: 5).

## Explicitly Excluded from MVP

- WebRTC voice streaming
- Complete UI implementation
- Benchmarking implementation
- PSTN/telephony (Twilio)
- Third-party voice platforms (Vapi, Retell, Bland)
- LangChain/LangGraph
- Celery/Redis/Kafka/RabbitMQ
- Vector databases / RAG
- Kubernetes / cloud infrastructure
- MCP

## License

Private - All rights reserved.
