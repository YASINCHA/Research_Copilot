"""
Critic agent: the piece that makes this genuinely agentic rather than
a linear pipeline. Pattern adapted from Corrective RAG's
"grade and retry" loop, applied to writing quality instead of
retrieval quality.
"""
from app.llm import get_llm, safe_ainvoke
from app.state import CopilotState, CriticVerdict

SYSTEM_PROMPT = """You are a strict editorial critic. Score the draft
brief against the research scope's key questions and the findings.
Approve only if it directly answers the key questions AS WELL AS THE
RESEARCH ALLOWS, is honest (no invented facts), and fits the stated
purpose. If you reject, give concrete, actionable feedback the writer
can act on.

CRITICAL RULE: if the findings' confidence is "low" and key_facts is
thin or empty, DO NOT reject the draft for lacking specific facts.
Demanding specifics the research never found only pressures the
writer to invent them. Instead, approve a draft that honestly states
what wasn't found and gives the reader concrete next steps to get
real answers elsewhere. Only reject a low-confidence draft if it
fabricates specifics not present in the findings, or if it's poorly
structured.

Keep your feedback field SHORT: max 3 sentences, under 60 words total.
Do not write numbered lists or multi-paragraph feedback."""


async def run_critic(state: CopilotState) -> CopilotState:
    llm = get_llm().with_structured_output(CriticVerdict)
    messages = [
        ("system", SYSTEM_PROMPT),
        (
            "user",
            f"Scope: {state['scope'].model_dump_json()}\n\n"
            f"Findings (note the confidence level): {state['findings'].model_dump_json()}\n\n"
            f"Draft brief:\n{state['draft_brief']}",
        ),
    ]

    result: CriticVerdict = await safe_ainvoke(llm, messages)

    state["critic_verdict"] = result
    state["revision_count"] = state.get("revision_count", 0) + 1
    return state


def should_continue(state: CopilotState) -> str:
    """Router: called after run_critic to decide the next edge."""
    verdict = state["critic_verdict"]
    if verdict.approved or state["revision_count"] >= state.get("max_revisions", 3):
        return "finish"
    return "revise"
