# Development Guide

## Prerequisites

- Python 3.11+
- Node.js 20+
- npm or pnpm

## Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload
```

Backend runs at: http://localhost:8000
API docs at: http://localhost:8000/docs

## Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend runs at: http://localhost:5173

## Running Tests

### Backend

Tests use mocked HTTP calls — no real API keys are required.

```bash
cd backend
pytest tests/ -v
```

The test suite includes:
- Configuration loading and validation
- Provider factory (instantiation, unsupported providers, missing keys)
- Deepgram, OpenRouter, and ElevenLabs adapters (mocked HTTP)
- AgentRuntime with FakeLLM (provider independence test)
- Tool call handling (single, multiple, multi-step, unknown, invalid args)
- Chat API endpoint (session creation, reuse, error handling)
- API key redaction / safe logging
- API endpoint tests (health, tools, providers, chat)

Lint and format:
```bash
cd backend
ruff check app/
ruff format --check app/
```

### Frontend
```bash
cd frontend
npm run build  # Type-check and build
```

## Database Migrations

```bash
cd backend

# Create a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1
```

## Docker

```bash
# Start both services
docker compose up

# Start in background
docker compose up -d

# Stop services
docker compose down
```

## Testing the Chat Endpoint

With the backend running and an `OPENROUTER_API_KEY` configured:

```bash
# Simple chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# Chat with tool calling
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Islamabad?"}'

# Continue an existing session
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Thanks!", "session_id": "<session-id-from-above>"}'
```

Or use the Text Chat tab in the frontend UI at http://localhost:5173.
