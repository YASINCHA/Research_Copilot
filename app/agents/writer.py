"""
Writer agent: drafts the final research brief.
Kept deliberately simple - the engineering effort goes into the
Critic loop, not this prompt.
"""
from app.llm import get_llm, safe_ainvoke
from app.state import CopilotState

SYSTEM_PROMPT = """You are a briefing writer. Using the research scope
and findings, write a concise, well-organized brief (under 350 words)
that directly answers the key questions and serves the stated
purpose.

Each fact in the findings has a source_url. When you use a fact in
the brief, cite it inline right after the claim, like this:
"EY invested $1B in tech and talent [Source: reuters.com]." Use the
domain only (not the full URL) to keep it readable. If a fact has no
source_url, don't fabricate one - state it without a citation.

Never invent facts beyond what you were given. If revision feedback
is provided, address it directly."""


async def run_writer(state: CopilotState) -> CopilotState:
    feedback = ""
    if state.get("critic_verdict") and not state["critic_verdict"].approved:
        feedback = f"\n\nPrevious draft was rejected. Feedback to address:\n{state['critic_verdict'].feedback}"

    llm = get_llm(temperature=0.5)
    messages = [
        ("system", SYSTEM_PROMPT),
        (
            "user",
            f"Topic: {state['topic']}\nPurpose: {state['purpose']}\n\n"
            f"Scope: {state['scope'].model_dump_json()}\n\n"
            f"Findings: {state['findings'].model_dump_json()}"
            f"{feedback}",
        ),
    ]
    response = await safe_ainvoke(llm, messages)
    state["draft_brief"] = response.content
    return state
