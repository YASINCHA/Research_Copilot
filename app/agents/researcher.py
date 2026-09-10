"""
Researcher agent: uses a REAL tool (web search) to ground its output
in facts instead of letting the LLM invent details.
Refuses (low confidence) rather than hallucinating - borrowed from
the "typed agentic RAG" refusal pattern.
"""
import asyncio
from app.llm import get_llm, safe_ainvoke
from app.state import CopilotState, ResearchFindings
from app.tools.web_search import search_company

SYSTEM_PROMPT = """You are a researcher. You will be given raw search
results, each with a snippet and its source URL. Summarize only what
the snippets actually support - key facts and recent developments.
For EACH fact you keep, attribute it to the source_url of the
specific search result it came from. If the snippets are thin,
contradictory, or off-topic, set confidence to "low" and say so. Do
not fill gaps with invented facts, and never invent a source_url -
leave it empty if genuinely unclear which result it came from."""


async def run_researcher(state: CopilotState) -> CopilotState:
    scope = state["scope"]
    query_terms = f"{state['topic']} " + " ".join(scope.subtopics[:2])

    # search_company does blocking network I/O (the ddgs library has
    # no async client) - run it in a thread so it doesn't stall the
    # event loop while other requests are being served.
    results = await asyncio.to_thread(search_company, query_terms)
    numbered_snippets = "\n---\n".join(
        f"[{r['url']}]\n{r['text']}" for r in results
    )

    llm = get_llm().with_structured_output(ResearchFindings)
    messages = [
        ("system", SYSTEM_PROMPT),
        (
            "user",
            f"Topic: {state['topic']}\n"
            f"Key questions to address: {scope.key_questions}\n\n"
            f"Search results (URL in brackets, snippet below it):\n{numbered_snippets}",
        ),
    ]
    result: ResearchFindings = await safe_ainvoke(llm, messages)
    state["findings"] = result
    return state
