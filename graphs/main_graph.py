"""Optional StateGraph wrapper — use ONLY when complex routing is needed.

Do NOT use this if a single deep agent with subagents handles everything.
Use when you need: triage → route → process → verify → respond
"""

from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent.build import build_agent
from agent.config import checkpointer, store
from graphs.router import triage_node


class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    classification: str
    urgency: str
    triage_reasoning: str


def build_supervisor_graph():
    """Build the optional supervisor graph with triage routing."""
    agent = build_agent()

    graph = StateGraph(SupervisorState)
    graph.add_node("triage", triage_node)
    graph.add_node("main_agent", agent)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        lambda s: s["classification"],
        {
            "route_to_research": "main_agent",
            "route_to_email": "main_agent",
            "route_to_general": "main_agent",
            "end": END,
        },
    )
    graph.add_edge("main_agent", END)

    return graph.compile(checkpointer=checkpointer, store=store)
