# Company Intelligence Agent

An AI-powered multi-agent system that researches, analyzes, and generates structured intelligence reports on any company. Built with **CrewAI**, **FastAPI**, and **Gradio**.

## Architecture

```
User Input (company name)
        │
        ▼
┌─────────────────────────────────────────────┐
│              CrewAI Pipeline                │
│                                             │
│  [Researcher] ──► [Analyst] ──► [Writer]      │
│  (DuckDuckGo)   (GPT-4o-mini) (GPT-4o-mini)  │
└─────────────────────────────────────────────┘
        │
        ▼
  Markdown Report
```

**Three sequential agents:**
1. **Research Specialist** — searches the web using DuckDuckGo to gather company data
2. **Business Analyst** — analyzes the research and extracts insights
3. **Report Writer** — produces a structured Markdown intelligence report

## Stack

| Component | Technology |
|---|---|
| Agent framework | CrewAI |
| LLM | OpenAI GPT-4o-mini |
| Search tool | DuckDuckGo (no API key) |
| API layer | FastAPI |
| Frontend | Gradio (mounted on FastAPI) |
| Observability | AgentOps |
| Hosting | Render.com |

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo>
cd crewai_project
uv sync
```

> Install UV: `pip install uv` or see [docs.astral.sh/uv](https://docs.astral.sh/uv)

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required keys:
- `OPENAI_API_KEY` — [platform.openai.com](https://platform.openai.com)

Optional:
- `AGENTOPS_API_KEY` — [agentops.ai](https://agentops.ai) (free tier for tracing)

### 3. Run locally

```bash
uv run uvicorn main:app --reload
```

| URL | Description |
|---|---|
| `http://localhost:8000/` | Gradio UI |
| `http://localhost:8000/docs` | FastAPI Swagger UI |
| `http://localhost:8000/health` | Health check |

## API Usage

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"company_name": "Tesla"}'
```

## Deploy to Render

1. Push to a GitHub repository
2. Connect the repo in [Render Dashboard](https://dashboard.render.com)
3. Render will auto-detect `render.yaml`
4. Add your API keys in the Render environment variables dashboard
5. Deploy

> **Note:** The Render free tier spins down after 15 minutes of inactivity. The first request after a cold start may take ~30 seconds.

## Observability

- **AgentOps dashboard** — tracks agent runs, tool calls, LLM token usage, and latency
- **OpenAI platform** — `platform.openai.com/traces` shows raw API call history
- **FastAPI logs** — visible in Render's log stream
