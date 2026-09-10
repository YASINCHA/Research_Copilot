"""
Shared state that flows through every agent node in the graph.
This is the single most important file to get right early -
every agent reads from and writes to this object.
"""
from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field


class ResearchScope(BaseModel):
    key_questions: List[str] = Field(default_factory=list)
    subtopics: List[str] = Field(default_factory=list)
    context_notes: str = ""  # why this research matters / what it's for


class SourcedFact(BaseModel):
    claim: str
    source_url: str = ""  # empty if the model couldn't attribute it to a specific result


class ResearchFindings(BaseModel):
    summary: str = ""
    key_facts: List[SourcedFact] = Field(default_factory=list)
    recent_developments: List[SourcedFact] = Field(default_factory=list)
    confidence: str = "low"  # "low" | "medium" | "high" -> refuse to use if low


class CriticVerdict(BaseModel):
    approved: bool
    feedback: str = ""
    score: int = 0  # 0-10


class CopilotState(TypedDict, total=False):
    # inputs
    topic: str  # e.g. a company name, a meeting subject, a technology
    purpose: str  # e.g. "job interview prep", "sales call", "personal research"

    # agent outputs, filled in as the graph runs
    scope: Optional[ResearchScope]
    findings: Optional[ResearchFindings]
    draft_brief: Optional[str]
    critic_verdict: Optional[CriticVerdict]

    # control
    revision_count: int
    max_revisions: int
    final_output: Optional[str]
