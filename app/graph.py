"""
Wires the 4 agents into a state graph:

  scope_agent -> researcher -> writer -> critic --(approved)--> finish
                                   ^                  |
                                   └----(revise)-------┘

This file is the "architecture diagram" of the project - the part
worth screenshotting for your README.
"""
from langgraph.graph import StateGraph, END
from app.state import CopilotState
from app.agents.scope_agent import run_scope_agent
from app.agents.researcher import run_researcher
from app.agents.writer import run_writer
from app.agents.critic import run_critic, should_continue


def finish(state: CopilotState) -> CopilotState:
    state["final_output"] = state["draft_brief"]
    return state


def build_graph():
    graph = StateGraph(CopilotState)

    graph.add_node("scope_agent", run_scope_agent)
    graph.add_node("researcher", run_researcher)
    graph.add_node("writer", run_writer)
    graph.add_node("critic", run_critic)
    graph.add_node("finish", finish)

    graph.set_entry_point("scope_agent")
    graph.add_edge("scope_agent", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")

    graph.add_conditional_edges(
        "critic",
        should_continue,
        {"finish": "finish", "revise": "writer"},
    )
    graph.add_edge("finish", END)

    return graph.compile()
