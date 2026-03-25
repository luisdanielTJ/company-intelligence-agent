import os
from crewai import Agent, LLM
from .tools import search_tool


def get_llm() -> LLM:
    return LLM(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


def create_researcher() -> Agent:
    return Agent(
        role="Company Research Specialist",
        goal="Gather comprehensive and accurate information about {company}",
        backstory=(
            "You are an expert business researcher with years of experience analyzing companies. "
            "You excel at finding accurate, up-to-date information from diverse sources including "
            "news articles, press releases, financial reports, and industry publications. "
            "You always verify information by searching multiple times with different queries."
        ),
        tools=[search_tool],
        llm=get_llm(),
        verbose=True,
        max_iter=3,
    )


def create_analyst() -> Agent:
    return Agent(
        role="Business Intelligence Analyst",
        goal="Analyze research data about {company} and extract key business insights",
        backstory=(
            "You are a seasoned business analyst who transforms raw research into actionable insights. "
            "You identify market trends, competitive advantages, potential risks, and growth opportunities. "
            "Your analysis is data-driven, objective, and focused on what matters most to stakeholders."
        ),
        llm=get_llm(),
        verbose=True,
    )


def create_writer() -> Agent:
    return Agent(
        role="Intelligence Report Writer",
        goal="Produce a professional, well-structured intelligence report about {company}",
        backstory=(
            "You are a professional business writer specializing in corporate intelligence reports. "
            "You present complex information clearly and concisely, using structured Markdown formatting. "
            "Your reports are used by executives and investors to make informed decisions."
        ),
        llm=get_llm(),
        verbose=True,
    )
