"""
Run with: uvicorn app.api:app --reload

Exposes the agent pipeline as a real API instead of only a Streamlit
script. The /brief endpoint streams progress in real time (Server-
Sent Events) so a client doesn't have to wait 15-60s staring at a
spinner with zero visibility into what's happening.
"""
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from app.graph import build_graph

load_dotenv()

app = FastAPI(title="AI Research & Meeting Co-Pilot API")

# Allow the demo.html client (or any local frontend) to call the API
# from the browser. Restrict origins in a real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class BriefRequest(BaseModel):
    topic: str
    purpose: str


def _serialize_node_update(node_name: str, node_state: dict) -> dict:
    """Pull out just the new, interesting piece of state each node
    produced, instead of dumping the whole (growing) state each time."""
    payload = {"node": node_name}

    if node_name == "scope_agent" and node_state.get("scope"):
        payload["scope"] = node_state["scope"].model_dump()
    elif node_name == "researcher" and node_state.get("findings"):
        payload["findings"] = node_state["findings"].model_dump()
    elif node_name == "writer" and node_state.get("draft_brief"):
        payload["draft"] = node_state["draft_brief"]
    elif node_name == "critic" and node_state.get("critic_verdict"):
        payload["verdict"] = node_state["critic_verdict"].model_dump()
    elif node_name == "finish" and node_state.get("final_output"):
        payload["final_output"] = node_state["final_output"]

    return payload


@app.post("/brief")
async def generate_brief_streaming(req: BriefRequest):
    """Streams one Server-Sent Event per agent step as the graph runs,
    so the client can show live progress instead of a blank wait."""
    graph = build_graph()

    async def event_generator():
        async for chunk in graph.astream(
            {
                "topic": req.topic,
                "purpose": req.purpose,
                "revision_count": 0,
                "max_revisions": 3,
            }
        ):
            node_name = next(iter(chunk))
            node_state = chunk[node_name]
            payload = _serialize_node_update(node_name, node_state)
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/brief/sync")
async def generate_brief_sync(req: BriefRequest):
    """Simple non-streaming version for clients that just want the
    final result (e.g. curl, quick scripts, tests)."""
    graph = build_graph()
    result = await graph.ainvoke(
        {
            "topic": req.topic,
            "purpose": req.purpose,
            "revision_count": 0,
            "max_revisions": 3,
        }
    )
    return {
        "scope": result["scope"].model_dump(),
        "findings": result["findings"].model_dump(),
        "final_brief": result["final_output"],
        "revisions": result["revision_count"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
