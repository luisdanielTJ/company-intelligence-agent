import asyncio
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

load_dotenv()

# Import after env vars are loaded
from src.crew import CompanyIntelligenceCrew

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _executor.shutdown(wait=False)


fastapi_app = FastAPI(
    title="Company Intelligence Agent",
    description=(
        "An AI-powered multi-agent system built with CrewAI that researches, "
        "analyzes, and reports on any company. Powered by OpenAI + DuckDuckGo."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class AnalysisRequest(BaseModel):
    company_name: str

    @field_validator("company_name")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("company_name must not be empty")
        return v.strip()


class AnalysisResponse(BaseModel):
    company: str
    report: str


@fastapi_app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint — used by Render to verify the service is running."""
    return {"status": "ok", "service": "Company Intelligence Agent"}


@fastapi_app.post("/api/analyze", response_model=AnalysisResponse, tags=["Agent"])
async def analyze_company(request: AnalysisRequest):
    """
    Run the multi-agent research pipeline on the given company.

    This kicks off three sequential agents:
    1. **Researcher** — searches the web for company data
    2. **Analyst** — extracts business insights
    3. **Report Writer** — produces a structured Markdown report

    ⚠️ This endpoint is synchronous and may take 30–90 seconds.
    """
    logger.info("Starting analysis for: %s", request.company_name)
    try:
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(
            _executor,
            lambda: CompanyIntelligenceCrew().run(request.company_name),
        )
        return AnalysisResponse(company=request.company_name, report=report)
    except Exception as exc:
        logger.error("Analysis failed for %s: %s", request.company_name, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Gradio UI — mounted on FastAPI at "/"
# ---------------------------------------------------------------------------

_AGENT_STAGES = [
    ("Researcher", "Searching the web and gathering company data..."),
    ("Analyst",    "Analyzing research and extracting business insights..."),
    ("Writer",     "Composing the intelligence report..."),
]


def _run_analysis(company_name: str):
    """Generator — yields live status updates then the final report."""
    company_name = company_name.strip()
    if not company_name:
        yield "Please enter a company name."
        return

    updates: queue.Queue = queue.Queue()
    completed_tasks = [0]

    def on_task_complete(_task_output):
        completed_tasks[0] += 1
        idx = completed_tasks[0]
        if idx < len(_AGENT_STAGES):
            agent, action = _AGENT_STAGES[idx]
            updates.put(("status", agent, action))

    def run_crew():
        try:
            result = CompanyIntelligenceCrew().run(company_name, on_task_complete=on_task_complete)
            updates.put(("done", result, ""))
        except Exception as exc:
            logger.error("Crew error: %s", exc)
            updates.put(("error", str(exc), ""))

    thread = threading.Thread(target=run_crew, daemon=True)
    thread.start()

    def status_md(agent: str, action: str) -> str:
        lines = []
        for i, (a, _) in enumerate(_AGENT_STAGES):
            if a == agent:
                lines.append(f"- ⚙️ **{a}** — {action}")
            elif i < _AGENT_STAGES.index((agent, action)):
                lines.append(f"- ✅ **{a}** — done")
            else:
                lines.append(f"- ⏳ **{a}**")
        return (
            f"## Analyzing **{company_name}**...\n\n"
            + "\n".join(lines)
            + "\n\n_This takes 30–90 seconds._"
        )

    first_agent, first_action = _AGENT_STAGES[0]
    yield status_md(first_agent, first_action)

    while True:
        try:
            msg_type, payload, extra = updates.get(timeout=120)
            if msg_type == "status":
                yield status_md(payload, extra)
            elif msg_type == "done":
                yield payload
                break
            elif msg_type == "error":
                yield f"**Error:** {payload}\n\nPlease check your API keys and try again."
                break
        except queue.Empty:
            yield "**Timeout:** The analysis took too long. Please try again."
            break


with gr.Blocks(
    title="Company Intelligence Agent",
    theme=gr.themes.Soft(),
    css=".output-markdown { font-size: 0.95rem; line-height: 1.6; }",
) as gradio_ui:

    gr.Markdown(
        """
        # Company Intelligence Agent
        > Powered by **CrewAI** · **OpenAI** · **DuckDuckGo**

        Enter any company name and get a comprehensive AI-generated intelligence report
        covering market position, recent news, financials, risks, and opportunities.
        """
    )

    with gr.Row():
        company_input = gr.Textbox(
            label="Company Name",
            placeholder="e.g., Tesla, Stripe, Anthropic, Nvidia...",
            scale=4,
        )
        submit_btn = gr.Button("Analyze", variant="primary", scale=1)

    output = gr.Markdown(
        label="Intelligence Report",
        elem_classes=["output-markdown"],
    )

    gr.Examples(
        examples=[["Tesla"], ["Stripe"], ["Anthropic"], ["Shopify"]],
        inputs=company_input,
    )

    gr.Markdown(
        """
        ---
        **API:** [`/docs`](/docs) · [`/health`](/health) · [`POST /api/analyze`](/docs#/Agent/analyze_company_api_analyze_post)
        """
    )

    submit_btn.click(fn=_run_analysis, inputs=company_input, outputs=output)
    company_input.submit(fn=_run_analysis, inputs=company_input, outputs=output)


# Mount Gradio onto FastAPI — Gradio serves at "/" and Swagger at "/docs"
app = gr.mount_gradio_app(fastapi_app, gradio_ui, path="/")
