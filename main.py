import asyncio
import logging
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

def _run_analysis(company_name: str):
    """Generator — shows a progress message, then yields the final report + button state."""
    company_name = company_name.strip()
    if not company_name:
        yield "Please enter a company name.", gr.update(visible=False)
        return

    yield (
        f"## Analyzing **{company_name}**...\n\n"
        "- ⚙️ **Researcher** → Analyst → Writer\n\n"
        "_This takes 30–90 seconds._",
        gr.update(visible=False),
    )

    try:
        report = CompanyIntelligenceCrew().run(company_name)
        yield report, gr.update(visible=True)
    except Exception as exc:
        logger.error("Crew error: %s", exc)
        yield f"**Error:** {exc}\n\nPlease check your API keys and try again.", gr.update(visible=True)


def _reset():
    return "", "", gr.update(visible=False)


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

    new_search_btn = gr.Button("New Search", visible=False, variant="secondary")

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

    # concurrency_limit=2 — max 2 analyses run simultaneously; extras queue
    submit_btn.click(
        fn=_run_analysis, inputs=company_input, outputs=[output, new_search_btn], concurrency_limit=2
    )
    company_input.submit(
        fn=_run_analysis, inputs=company_input, outputs=[output, new_search_btn], concurrency_limit=2
    )
    new_search_btn.click(
        fn=_reset, outputs=[company_input, output, new_search_btn]
    )


# Mount Gradio onto FastAPI — Gradio serves at "/" and Swagger at "/docs"
app = gr.mount_gradio_app(fastapi_app, gradio_ui, path="/")
