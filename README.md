# JalurAI

Smart logistics risk prediction system.

## Structure

- `backend/` — FastAPI API and prediction pipeline boundary
- `frontend/` — Next.js operational dashboard
- `docker-compose.yml` — local development services

## Run

```bash
docker compose up --build
```

API docs: `http://localhost:8000/docs`
Dashboard: `http://localhost:3000`

## Optional Local Resolver Agent

JalurAI uses Ollama with the `llama3.2` model when it is available. If
Ollama is offline or the model is missing, the API automatically uses the
deterministic resolver fallback.

```bash
ollama pull llama3.2
ollama serve
```

Ollama must be reachable at `http://localhost:11434`.
