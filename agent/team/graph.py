"""Team graph assembly — LangGraph StateGraph with Send API for parallel execution.

Flow:
    START → planner (ensemble vote) → [Send to workers in parallel] → synthesize → END
"""

import logging
import re

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agent.team.planner import plan_node, _parse_workers
from agent.team.state import TeamState
from agent.team.synthesizer import synthesize_node
from agent.team.workers import code_worker, db_worker, rag_worker, web_worker

logger = logging.getLogger(__name__)

# Map worker names to (graph_node_name, task_key)
WORKER_MAP = {
    "rag_analyst": ("rag_worker", "rag_result"),
    "code_explorer": ("code_worker", "code_result"),
    "db_analyst": ("db_worker", "db_result"),
    "web_researcher": ("web_worker", "web_result"),
}


def _extract_tasks(plan: str) -> dict[str, str]:
    """Parse task instructions for each worker from the plan."""
    tasks: dict[str, str] = {}
    current_worker = None
    for line in plan.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("WORKERS:"):
            continue
        if stripped.upper().startswith("TASKS:"):
            continue
        # Match "- worker_name: task_description"
        match = re.match(r"^-\s*(\w+):\s*(.+)", stripped)
        if match:
            worker = match.group(1)
            task = match.group(2)
            if worker in WORKER_MAP:
                tasks[worker] = task
    return tasks


def route_after_plan(state: TeamState) -> list[Send]:
    """After planning, dispatch to workers in parallel via Send API."""
    plan = state.get("plan", "")
    workers = _parse_workers(plan)
    tasks = _extract_tasks(plan)

    if not workers:
        # No workers needed — go directly to synthesize
        return [Send("synthesize", {})]

    sends = []
    for worker_name in workers:
        if worker_name not in WORKER_MAP:
            continue
        node_name, _ = WORKER_MAP[worker_name]
        task_text = tasks.get(worker_name, state.get("user_query", ""))
        sends.append(Send(node_name, {
            "task": task_text,
            "user_query": state.get("user_query", ""),
            "project": state.get("project", ""),
        }))

    if not sends:
        return [Send("synthesize", {})]

    logger.info(f"Dispatching {len(sends)} workers in parallel: {workers}")
    return sends


def build_team_graph(checkpointer=None):
    """Build and compile the team StateGraph.

    Args:
        checkpointer: Optional checkpointer for state persistence.

    Returns:
        Compiled StateGraph ready for invoke/stream.
    """
    graph = StateGraph(TeamState)

    # Nodes
    graph.add_node("planner", plan_node)
    graph.add_node("rag_worker", rag_worker)
    graph.add_node("code_worker", code_worker)
    graph.add_node("db_worker", db_worker)
    graph.add_node("web_worker", web_worker)
    graph.add_node("synthesize", synthesize_node)

    # Edges
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_after_plan, [
        "rag_worker", "code_worker", "db_worker", "web_worker", "synthesize",
    ])

    # All workers converge to synthesize
    graph.add_edge("rag_worker", "synthesize")
    graph.add_edge("code_worker", "synthesize")
    graph.add_edge("db_worker", "synthesize")
    graph.add_edge("web_worker", "synthesize")
    graph.add_edge("synthesize", END)

    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)
