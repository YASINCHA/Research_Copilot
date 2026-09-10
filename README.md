# 🧭 AI Research & Meeting Co-Pilot

[![CI](https://github.com/YASINCHA/Research_Copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/YASINCHA/Research_Copilot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A multi-agent system that turns a raw topic — a company, a technology, a meeting to prepare for — into a sourced, self-revising research brief. Built as a full demonstration of **LLMOps practices**, not just a prototype: observability, automated evaluation, a production-style async API, and CI/CD are all part of the project, not an afterthought.

## Architecture

```
scope_agent -> researcher -> writer -> critic --(approved)--> finish
                                 ^                  |
                                 +----(revise)------+
```

- **Scope agent** — turns a vague topic + purpose into 3-5 concrete key questions and subtopics
- **Researcher** — real web-search tool call (not memory); explicitly reports low confidence instead of inventing facts
- **Writer** — drafts the brief from the scope and findings
- **Critic** — grades the draft, approves or sends it back with concrete feedback (capped at 3 revisions). Adapted from the Corrective RAG (CRAG) "grade and retry" pattern, applied here to writing quality instead of retrieval quality.

**A real bug this caught:** early on, the Critic rejected honest "no data found" drafts for lacking specifics, which pressured the Writer into inventing plausible-sounding facts to get approved — a textbook reward-hacking failure mode. Fixed by teaching the Critic to accept transparency over fabricated specificity when the Researcher's confidence is genuinely low.

## What makes this "production-grade," not just a prototype

**Observability (LangSmith)** — every agent call is traced: latency, tokens, inputs/outputs, and the revision loop when it triggers.

**Automated evaluation (DeepEval + pytest)** — two test suites *measure* quality instead of asserting it:
- *Faithfulness*: does the final brief only state things the Researcher actually found?
- *Revision effectiveness*: when the Critic rejects with specific feedback, does the rewrite actually address it?

**Async API (FastAPI)** — the agent pipeline is exposed as a real API with two endpoints: `/brief` streams each agent's progress in real time via Server-Sent Events, `/brief/sync` returns the final result directly. All four agents use `ainvoke()` (not `invoke()`) so the event loop stays non-blocking under concurrent requests — a genuine correctness fix, not just an API wrapper around blocking calls.

**CI/CD (GitHub Actions + Docker)** — every push runs the full automated eval suite, then verifies the Docker image builds cleanly.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY (free tier) and LANGCHAIN_API_KEY (optional, for tracing)
```

**Run the Streamlit UI:**
```bash
streamlit run app/main.py
```

**Run the API:**
```bash
uvicorn app.api:app --reload
# then open demo.html in a browser to see the live agent stream
```

**Run everything with Docker:**
```bash
docker compose up --build
# API on :8000, Streamlit on :8501
```

**Run the automated evals:**
```bash
pytest tests/ -v
```

## Stack

Python, LangGraph (multi-agent orchestration + conditional revision routing), Groq (`openai/gpt-oss-20b`, free tier), Pydantic (structured, validated agent outputs), DuckDuckGo Search (`ddgs`, real tool use), FastAPI + Server-Sent Events (async streaming API), Streamlit (UI), LangSmith (tracing), DeepEval + pytest (automated evals), Docker + GitHub Actions (CI/CD).

## Known limitations

- Free-tier web search (`ddgs`) is sometimes thin on recent results — a production version would use Tavily or Serper.
- No persistent state/checkpointing yet — a crash mid-run loses progress. LangGraph supports checkpointing (`MemorySaver`/`SqliteSaver`); on the roadmap.
- The small judge/agent model (20B, free tier) occasionally needs a retry on malformed structured output — handled, but a larger model would need it less often.

## Roadmap

- [ ] LangGraph checkpointing for resumable runs
- [ ] Swap in a local model (Ollama) for a fully offline mode
- [ ] Upgrade web search to a paid-tier provider for richer results
