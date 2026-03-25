import os
import logging

from crewai import Crew, Process

from .agents import create_researcher, create_analyst, create_writer
from .tasks import create_research_task, create_analysis_task, create_report_task

logger = logging.getLogger(__name__)


def _init_agentops() -> None:
    api_key = os.getenv("AGENTOPS_API_KEY")
    if not api_key:
        return
    try:
        import agentops
        agentops.init(api_key, default_tags=["crewai", "company-intelligence"])
        logger.info("AgentOps tracing initialized.")
    except Exception as exc:
        logger.warning("AgentOps init failed (tracing disabled): %s", exc)


_init_agentops()


class CompanyIntelligenceCrew:
    """Orchestrates the multi-agent company research pipeline."""

    def run(self, company: str) -> str:
        researcher = create_researcher()
        analyst = create_analyst()
        writer = create_writer()

        tasks = [
            create_research_task(company, researcher),
            create_analysis_task(company, analyst),
            create_report_task(company, writer),
        ]

        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff(inputs={"company": company})
        return str(result)
