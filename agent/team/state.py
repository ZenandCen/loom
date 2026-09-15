"""Team graph state definitions."""

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class TeamState(TypedDict, total=False):
    """Shared state for the multi-agent team graph."""

    # Input
    user_query: str
    user_id: str
    project: str

    # Planning
    plan: str
    plan_votes: dict[str, str]

    # Worker results
    rag_result: str
    rag_sources: list[str]
    code_result: str
    code_sources: list[str]
    db_result: str
    db_sources: list[str]
    web_result: str
    web_sources: list[str]

    # Final output
    synthesis: str
    diagram: str
    messages: Annotated[list, add_messages]


class WorkerTask(TypedDict):
    """Task dispatched to a single worker via Send API."""

    task: str
    user_query: str
    project: str
