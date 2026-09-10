"""
Automated evaluation of the agent pipeline, not just manual spot-checks.

Run with: pytest tests/test_pipeline_evals.py -v

Test 1 (Faithfulness): does the final brief only state things the
Researcher actually found? This is the measurable version of
"the agent doesn't hallucinate."

Test 2 (Critic improves the draft): when the Critic rejects a draft
and sends feedback, does the rewritten draft actually address that
feedback? This is the measurable version of "the revision loop works."

NOTE ON ASYNC: the graph and agents are async (see app/agents/*.py),
but these test functions are deliberately kept SYNCHRONOUS. DeepEval's
assert_test() manages its own event loop internally via nest_asyncio;
running it from inside a pytest-asyncio-managed async test corrupts
both libraries' event loop state. Instead, each async call is wrapped
in its own self-contained asyncio.run(), which fully starts and tears
down before DeepEval's internal loop ever gets involved.
"""
import time
import asyncio
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import FaithfulnessMetric, GEval
import pytest

from app.graph import build_graph
from app.agents.writer import run_writer
from app.state import CriticVerdict, ResearchScope, ResearchFindings, SourcedFact
from app.eval.groq_judge import GroqJudge

judge = GroqJudge()

# Groq's free tier has a low tokens-per-minute limit and DeepEval's
# FaithfulnessMetric makes several verbose calls per case. Keep this
# to 1 topic by default so the suite fits the free tier; add more
# once you have a paid tier or a higher-limit provider.
RESEARCH_TOPICS = [
    ("Python programming language", "personal research"),
]


@pytest.mark.parametrize("topic,purpose", RESEARCH_TOPICS)
def test_faithfulness_no_hallucination(topic, purpose):
    """The brief should only assert things the Researcher's findings support."""
    graph = build_graph()
    result = asyncio.run(
        graph.ainvoke(
            {"topic": topic, "purpose": purpose, "revision_count": 0, "max_revisions": 3}
        )
    )

    findings = result["findings"]
    retrieval_context = [f.claim for f in findings.key_facts + findings.recent_developments]

    test_case = LLMTestCase(
        input=f"Research brief on {topic} for {purpose}",
        actual_output=result["final_output"],
        retrieval_context=retrieval_context or ["No facts were found - low confidence expected."],
    )

    metric = FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True)
    assert_test(test_case, [metric])


def test_critic_improves_draft_on_revision():
    """When the Critic rejects with specific feedback, the rewritten
    draft should actually address that feedback - not just be a
    cosmetic reshuffle."""
    # Free-tier pacing: let the previous test's token usage clear
    # the rate-limit window before making more heavy judge calls.
    time.sleep(20)

    fake_feedback = (
        "The draft never mentions pricing or cost, which the reader "
        "explicitly needs to know before making a decision."
    )

    base_state = {
        "topic": "Notion vs Obsidian for note-taking",
        "purpose": "choosing a tool to buy",
        "scope": ResearchScope(
            key_questions=["Which is cheaper?"], subtopics=["Pricing"]
        ),
        "findings": ResearchFindings(
            key_facts=[
                SourcedFact(claim="Notion has a free tier and paid plans from $8/month.", source_url="notion.so"),
                SourcedFact(claim="Obsidian's core app is free; sync costs $8/month.", source_url="obsidian.md"),
            ],
            recent_developments=[],
            confidence="medium",
        ),
        "revision_count": 1,
        "max_revisions": 3,
    }

    # First draft, no feedback yet
    draft_v1_state = asyncio.run(run_writer({**base_state, "critic_verdict": None}))

    # Simulate a rejection with concrete feedback, then rewrite
    rejected = CriticVerdict(approved=False, feedback=fake_feedback, score=3)
    draft_v2_state = asyncio.run(run_writer({**base_state, "critic_verdict": rejected}))

    test_case = LLMTestCase(
        input=fake_feedback,
        actual_output=draft_v2_state["draft_brief"],
    )

    metric = GEval(
        name="AddressesFeedback",
        criteria=(
            "Does the actual_output clearly address the specific "
            "feedback given in 'input' (i.e. does it now mention "
            "pricing/cost)?"
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.6,
    )
    assert_test(test_case, [metric])

    # Sanity check the revision actually changed something
    assert draft_v1_state["draft_brief"] != draft_v2_state["draft_brief"]
