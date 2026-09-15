"""Ensemble Planner — 3 LLMs vote on the execution plan.

Voting mechanism:
- All 3 LLMs generate a plan in parallel
- Compare plans: if 2+ agree on the same workers → use that
- If tie or all different → fallback to Qwen (LLM1)
"""

import asyncio
import logging
from typing import Any

from agent.team.state import TeamState
from utils.models import LLM, get_llm

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """\
You are a planning supervisor. Given the user's request, create an execution plan.

Available workers:
- rag_analyst: Search indexed documents (RAG). Use for .xlsx, .pdf, .md file queries, document analysis.
- code_explorer: Read and analyze source code. Use for code structure, implementation details.
- db_analyst: Query PostgreSQL database. Use for table schema, data queries, SQL.
- web_researcher: Search the web. Use for external knowledge, documentation lookup.

Respond with EXACTLY this format (no extra text):
WORKERS: <comma-separated list from: rag_analyst, code_explorer, db_analyst, web_researcher>
TASKS:
- rag_analyst: <specific task instruction>
- code_explorer: <specific task instruction>
- db_analyst: <specific task instruction>

Only include workers that are actually needed. If only 1 worker is needed, list only that one.
If the request is simple (greeting, set_project, remember), respond:
WORKERS: none
"""


async def _generate_plan(llm_name: LLM, query: str, project: str) -> str:
    """Generate a plan using a specific LLM."""
    llm = get_llm(llm_name)
    prompt = (
        f"Active project: {project or 'none'}\n\n"
        f"User request: {query}\n\n"
        f"Create the execution plan:"
    )
    response = await asyncio.to_thread(llm.invoke, [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    content = response.content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        content = " ".join(parts)
    return content.strip()


def _parse_workers(plan: str) -> set[str]:
    """Extract worker names from a plan string."""
    workers_line = ""
    for line in plan.split("\n"):
        if line.strip().upper().startswith("WORKERS:"):
            workers_line = line.split(":", 1)[1].strip()
            break
    if not workers_line or workers_line.lower() == "none":
        return set()
    known = {"rag_analyst", "code_explorer", "db_analyst", "web_researcher"}
    return {w.strip() for w in workers_line.split(",") if w.strip() in known}


def _vote(plans: dict[str, str]) -> str:
    """Vote on plans. Majority wins. Tie → fallback to Qwen (LLM1)."""
    worker_sets = {name: _parse_workers(plan) for name, plan in plans.items()}

    # Count which worker sets appear most
    from collections import Counter

    set_counts: Counter = Counter()
    for ws in worker_sets.values():
        set_counts[frozenset(ws)] += 1

    if not set_counts:
        return plans.get("qwen", plans.get("gemini", ""))

    top_count, top_sets = set_counts.most_common(1)[0][1], [s for s, c in set_counts.items() if c == set_counts.most_common(1)[0]]

    if len(top_sets) == 1:
        # Clear majority
        for name, ws in worker_sets.items():
            if frozenset(ws) in top_sets:
                return plans[name]

    # Tie or all different → fallback to Qwen
    logger.info("Plan vote: tie/fallback to Qwen")
    return plans.get("qwen", list(plans.values())[0])


async def plan_node(state: TeamState) -> dict[str, Any]:
    """Supervisor planning node — ensemble of 3 LLMs vote on the plan."""
    query = state.get("user_query", "")
    project = state.get("project", "")

    # Generate plans in parallel from 3 LLMs
    plans: dict[str, str] = {}
    tasks = [
        _generate_plan(LLM.OPENAI, query, project),
        _generate_plan(LLM.GEMINI, query, project),
        _generate_plan(LLM.OLLAMA, query, project),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    names = ["qwen", "gemini", "ollama"]
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            logger.warning(f"Planner {name} failed: {result}")
            plans[name] = ""
        else:
            plans[name] = result
            logger.info(f"Plan from {name}: {result[:200]}")

    # Vote
    valid_plans = {k: v for k, v in plans.items() if v}
    if not valid_plans:
        return {"plan": "WORKERS: none", "plan_votes": plans}

    winner = _vote(valid_plans)
    logger.info(f"Planner VOTE winner: {list(valid_plans.keys())[list(valid_plans.values()).index(winner)]}")

    return {"plan": winner, "plan_votes": plans}
