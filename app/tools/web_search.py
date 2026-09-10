"""
Real tool call, not a hallucinated one. This is what makes the
Researcher agent "agentic" instead of just another prompt.
"""
from ddgs import DDGS


def search_company(company_name: str, max_results: int = 5) -> list[dict]:
    """Returns raw snippets about the company, each with its source
    URL, so downstream agents can cite claims back to a real source
    instead of asserting facts with no way to verify them."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(f"{company_name} company news recent", max_results=max_results):
            snippet = r.get("body") or r.get("title", "")
            url = r.get("href", "")
            if snippet:
                results.append({"text": snippet, "url": url})
    return results
