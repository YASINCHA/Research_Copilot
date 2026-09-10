"""
Scope agent: given a topic and purpose, decides what actually needs
to be researched. This is step 1 - get this alone working end-to-end
before touching any other agent.
"""
from app.llm import get_llm, safe_ainvoke
from app.state import CopilotState, ResearchScope

SYSTEM_PROMPT = """You are a research planner. Given a topic and the
purpose of the research (e.g. a meeting, a decision, personal
curiosity), decide what specifically needs to be found out.

STRICT LIMITS - do not exceed these under any circumstances:
- Exactly 3 to 5 key_questions, each a single sentence, no duplicates.
- Exactly 3 to 5 subtopics, each 2-4 words, no duplicates.

Do not repeat similar questions with slightly different wording.
Do not do the research yourself - just scope it."""


async def run_scope_agent(state: CopilotState) -> CopilotState:
    llm = get_llm().with_structured_output(ResearchScope)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", f"Topic: {state['topic']}\n\nPurpose: {state['purpose']}"),
    ]

    result: ResearchScope = await safe_ainvoke(llm, messages)

    # Hard safety net even if the model ignores the limit
    result.key_questions = result.key_questions[:5]
    result.subtopics = result.subtopics[:5]

    state["scope"] = result
    return state
