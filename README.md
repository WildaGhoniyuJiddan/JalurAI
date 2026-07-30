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

### Install Ollama on Windows

The Python `ollama` package is only the client library. The Ollama Windows
application is still required to provide the CLI and local model server.

1. Download and run `OllamaSetup.exe` from the
   [official Ollama download page](https://ollama.com/download/windows).
2. Close all PowerShell windows, then open a new PowerShell session so the
   updated `PATH` is loaded.
3. Verify the installation and download the model:

```powershell
ollama --version
ollama pull llama3.2
```

The Windows application normally starts the Ollama server automatically in
the background. Verify that it is reachable at `http://localhost:11434`:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

If `ollama` is still not recognized after installation, check the default
installation path and add it to the current PowerShell session:

```powershell
Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" --version
$env:Path += ";$env:LOCALAPPDATA\Programs\Ollama"
ollama pull llama3.2
```

Alternatively, Ollama provides an official PowerShell installer:

```powershell
irm https://ollama.com/install.ps1 | iex
```

After Ollama and `llama3.2` are available, start the JalurAI backend as usual.
If Ollama is unavailable, prediction still works through the deterministic
resolver fallback.
