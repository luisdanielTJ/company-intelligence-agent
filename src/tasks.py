from crewai import Task
from crewai import Agent


def create_research_task(company: str, researcher: Agent) -> Task:
    return Task(
        description=(
            f"Research {company} thoroughly using multiple search queries. Collect:\n"
            "- Company overview: founding year, headquarters, size, industry sector\n"
            "- Core products and services\n"
            "- Recent news and press releases (last 6 months)\n"
            "- Key leadership team (CEO, CTO, and other C-suite)\n"
            "- Financials: revenue, funding rounds, valuation (if available)\n"
            "- Main competitors and market share\n"
            "- Technology stack or infrastructure (if publicly known)\n\n"
            "Perform 2 targeted searches. Be concise."
        ),
        expected_output=(
            "A bullet-point research summary under 350 words, organized by category. "
            "Key facts only — no filler."
        ),
        agent=researcher,
    )


def create_analysis_task(company: str, analyst: Agent) -> Task:
    return Task(
        description=(
            f"Using the research gathered on {company}, perform a business intelligence analysis covering:\n"
            "- Competitive positioning: what makes them stand out or fall behind\n"
            "- Growth trajectory: are they scaling, stagnating, or declining\n"
            "- Key risks: market, operational, regulatory, or reputational\n"
            "- Opportunities: untapped markets, partnerships, product expansion\n"
            "- Technology strengths or gaps\n"
            "- Business model assessment: is it sustainable and defensible\n\n"
            "Base your analysis strictly on the research provided. Be objective."
        ),
        expected_output=(
            "A concise analysis under 250 words, one short paragraph per category. "
            "Only the most critical insights."
        ),
        agent=analyst,
    )


def create_report_task(company: str, writer: Agent) -> Task:
    return Task(
        description=(
            f"Write a professional intelligence report about {company} in Markdown. "
            "Use the research and analysis provided. Follow this exact structure:\n\n"
            f"# {company} — Intelligence Report\n\n"
            "## Executive Summary\n"
            "## Company Overview\n"
            "## Products & Services\n"
            "## Market Position & Competitors\n"
            "## Recent Developments\n"
            "## Technology & Innovation\n"
            "## Financial Overview\n"
            "## Key Risks & Opportunities\n"
            "## Conclusion\n\n"
            "Keep each section to 3–5 bullet points maximum. Total report under 600 words."
        ),
        expected_output=(
            f"A complete, well-formatted Markdown intelligence report about {company} "
            "following the specified structure, ready to be rendered and presented."
        ),
        agent=writer,
    )
