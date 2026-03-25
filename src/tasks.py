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
            "Perform at least 3 different searches to ensure comprehensive coverage."
        ),
        expected_output=(
            "A detailed research summary organized by category, with source references "
            "and key facts clearly stated for each section."
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
            "A structured analysis with clear findings and reasoning for each category, "
            "highlighting the most critical insights for decision-makers."
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
            "Keep each section concise but informative. Use bullet points where appropriate."
        ),
        expected_output=(
            f"A complete, well-formatted Markdown intelligence report about {company} "
            "following the specified structure, ready to be rendered and presented."
        ),
        agent=writer,
    )
