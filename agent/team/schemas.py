"""Pydantic schemas for team graph I/O — clear contract at the API boundary."""

from enum import Enum
from pydantic import BaseModel, Field


class WorkerName(str, Enum):
    RAG = "rag_analyst"
    CODE = "code_explorer"
    DB = "db_analyst"
    WEB = "web_researcher"


class TeamInput(BaseModel):
    """Input to the team graph."""

    user_query: str = Field(..., min_length=1, description="User's question or request")
    user_id: str = Field(default="", description="Slack user ID")
    project: str = Field(default="", description="Active project name (e.g. 'fpt-dwh-reconcile-svc')")


class WorkerStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    ERROR = "error"


class WorkerResult(BaseModel):
    """Structured output from a single worker."""

    worker: WorkerName
    status: WorkerStatus = WorkerStatus.OK
    content: str = Field(default="", description="Worker's answer/analysis")
    sources: list[str] = Field(default_factory=list, description="File paths or URLs referenced")
    chars: int = Field(default=0, description="Length of content")


class PlannerInput(BaseModel):
    """Input to the ensemble planner node."""

    query: str = Field(description="User's question or request")
    project: str = Field(default="", description="Active project name")


class PlanResult(BaseModel):
    """Output from the ensemble planner."""

    workers: list[WorkerName] = Field(default_factory=list)
    tasks: dict[str, str] = Field(default_factory=dict, description="worker_name -> task instruction")
    votes: dict[str, str] = Field(default_factory=dict, description="llm_name -> raw plan text")


class DiagramType(str, Enum):
    FLOWCHART = "flowchart"
    SEQUENCE = "sequence"
    CLASS = "class"
    ER = "er"
    ACTIVITY = "activity"
    USE_CASE = "use_case"
    C4 = "c4"
    NONE = "none"


class TeamOutput(BaseModel):
    """Final output from the team graph."""

    synthesis: str = Field(..., description="Combined answer for the user")
    plan: str = Field(default="", description="Raw plan text")
    workers: list[WorkerResult] = Field(default_factory=list, description="Individual worker results")
    diagram: str = Field(default="", description="Mermaid diagram source (if generated)")
    diagram_type: DiagramType = Field(default=DiagramType.NONE)
    total_chars: int = Field(default=0, description="Total output size")

    def to_slack_text(self) -> str:
        """Format output for Slack (max 4000 chars)."""
        parts = [self.synthesis]
        if self.diagram:
            parts.append(f"\n\n```mermaid\n{self.diagram}\n```")
        text = "\n".join(parts)
        if len(text) > 4000:
            text = text[:3997] + "..."
        return text
