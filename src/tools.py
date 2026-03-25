from crewai.tools import tool
from duckduckgo_search import DDGS


@tool("Web Search")
def search_tool(query: str) -> str:
    """Search the web for up-to-date information about a company, topic, or event."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))

    if not results:
        return "No results found for this query."

    return "\n\n".join(
        f"**{r['title']}**\n{r['href']}\n{r['body'][:300]}"
        for r in results
    )
