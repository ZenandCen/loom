"""Report pipeline — deterministic workflow as a subagent."""

from typing import TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient

from agent.prompt import PIPELINE_ANALYZE_PROMPT, PIPELINE_FORMAT_PROMPT
from utils.models import LLM, get_llm

model = get_llm(LLM.OPENAI)

tavily_client = TavilyClient()


class ReportWorkflowState(TypedDict):
    messages: list
    topic: str
    data: str
    report: str


def _gather(state: ReportWorkflowState) -> dict:
    """Step 1: Gather raw data from web search."""
    topic = state["messages"][-1].content
    results = tavily_client.search(topic, max_results=5)
    raw = "\n\n".join(
        f"Source: {r['url']}\n{r['content']}" for r in results["results"]
    )
    return {"topic": topic, "data": raw}


def _analyze(state: ReportWorkflowState) -> dict:
    """Step 2: Analyze gathered data with LLM."""
    response = model.invoke([
        {"role": "system", "content": PIPELINE_ANALYZE_PROMPT},
        {
            "role": "user",
            "content": f"Topic: {state['topic']}\n\nData:\n{state['data']}",
        },
    ])
    return {"report": response.content}


def _format(state: ReportWorkflowState) -> dict:
    """Step 3: Format analysis into structured report."""
    response = model.invoke([
        {"role": "system", "content": PIPELINE_FORMAT_PROMPT},
        {"role": "user", "content": state["report"]},
    ])
    return {"messages": [AIMessage(content=response.content)]}


# --- Build pipeline graph ---
_report_graph = StateGraph(ReportWorkflowState)
_report_graph.add_node("gather", _gather)
_report_graph.add_node("analyze", _analyze)
_report_graph.add_node("format", _format)
_report_graph.add_edge(START, "gather")
_report_graph.add_edge("gather", "analyze")
_report_graph.add_edge("analyze", "format")
_report_graph.add_edge("format", END)

report_pipeline = _report_graph.compile()

# --- Expose as subagent ---
report_subagent = {
    "name": "report-pipeline",
    "description": (
        "Generate a structured research report with sources. "
        "Use for detailed analysis requests requiring a formal report."
    ),
    "runnable": report_pipeline,
}
