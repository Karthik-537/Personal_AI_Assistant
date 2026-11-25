# Personal AI Assistant

Modular personal AI assistant built around Google Gemini and the Model Context
Protocol (MCP). Each capability (weather, PC automation, phone integration)
is exposed as an independent MCP server; a central agent discovers tools
dynamically and orchestrates them via Gemini function calling.

## Architecture

```
User → Python app (MCP host) → Gemini API → MCP client → MCP servers → external APIs
```

- `backend/` — host application: Gemini agent + MCP client
- `mcp_servers/` — independent tool servers (weather, …)
- `tests/` — unit/integration tests

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env       # then fill in GEMINI_API_KEY
```

## Status

Module 1 (Weather Agent) in progress — see project plan for steps.
