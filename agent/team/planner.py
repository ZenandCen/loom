"""Planner — QWEN routes the worker, then 3 LLMs vote on the best task approach.

Design (LOOM_PLANNER_ENSEMBLE=1, default):
  Phase 1: QWEN alone picks which worker(s) to dispatch (routing) + a task draft.
           Safety-net: if qwen says "none" but it's a real question → web_researcher.
  Phase 2: GEMINI + OLLAMA each propose the best task instruction for the chosen worker(s).
  Phase 3: A JUDGE LLM (default gemini, unbiased) picks the best of the 3 task plans.

Fast mode (LOOM_PLANNER_ENSEMBLE=0):
  QWEN only — its routing + task are used directly, no vote.
"""

import asyncio
import logging
import os
import re
from typing import Any

from agent.team.state import TeamState
from utils.models import LLM, get_llm, get_backbone_llm

logger = logging.getLogger(__name__)

_ENSEMBLE = os.getenv("LOOM_PLANNER_ENSEMBLE", "1") == "1"
_JUDGE = os.getenv("LOOM_PLANNER_JUDGE", "gemini").lower()

KNOWN_WORKERS = {"rag_analyst", "code_explorer", "db_analyst", "web_researcher"}

ROUTER_SYSTEM = """\
You are a planning supervisor. Given the user's request, decide which worker(s) should handle it and draft a task.

Available workers:
- rag_analyst: Search indexed documents (RAG). Use for .xlsx, .pdf, .md file queries, document analysis, project knowledge.
- code_explorer: Read and analyze source code. Use for code structure, implementation details, architecture, "how does X work".
- db_analyst: Query PostgreSQL database. Use for table schema, data queries, SQL, data relationships.
- web_researcher: Search the web. Use for external knowledge, general knowledge, health, science, facts, definitions, current events, technology best practices.

Respond with EXACTLY this format (no extra text):
WORKERS: <comma-separated list from: rag_analyst, code_explorer, db_analyst, web_researcher, or "none">
TASKS:
- <worker>: <specific task instruction>

Worker selection rules:
- Question about the ACTIVE project's source code / architecture / implementation → code_explorer
- Question about the project's database/tables/data/relationships → db_analyst
- Question about indexed project documents/files / learned knowledge → rag_analyst
- GENERAL-knowledge question NOT tied to the project (health, science, how-things-work, facts, current events, translations, definitions) → web_researcher
- "How to build X" / "Best practice for X" / "What technology should I use" → code_explorer (existing patterns) + web_researcher (modern solutions)
- A genuine question that asks for information is NEVER "none" — always pick at least one worker.
- "WORKERS: none" is ONLY for pure greetings ("hello", "hi", "thank you"), pleasantries, or meta-commands (set_project, remember, reset).
- If only 1 worker is needed, list only that one.

Task specificity rules:
- Name EXACT entities: file names, function names, table names, column names, module paths.
- If the user's question is ambiguous, make a reasonable interpretation and state it in the task (e.g., "Interpreting 'the API' as the REST endpoints in src/routes/").
- For architecture questions: specify what aspect (data flow, component interaction, entry points, layer structure).
- For "how does X work": specify the starting point (entry point, API endpoint, function) and what level of detail (high-level flow vs line-by-line).
- Do NOT write vague tasks like "analyze the code" or "find relevant information". Be surgical.
"""

TASK_SYSTEM = """\
You are a planning supervisor. The worker(s) to be used have ALREADY been decided: {workers}.
Your ONLY job is to write the best, most specific task instruction(s) for that worker to answer the user's request.
Do NOT change or add workers. Do NOT answer the question yourself.

Respond with EXACTLY this format (no extra text):
TASKS:
- {workers}: <specific, well-scoped task instruction>

Task writing rules:
- Name EXACT entities: file names, function names, table names, column names, module paths, API endpoints.
- Be surgical: "Find the authentication middleware in src/middleware/ and trace how it validates JWT tokens" NOT "analyze the code".
- If the question is ambiguous, state your interpretation: "Interpreting 'the data flow' as the ETL pipeline from source tables to warehouse."
- For code_explorer: specify what to trace (call chain, data flow, error handling) and the starting point.
- For db_analyst: specify which tables to examine and what relationships to trace.
- For rag_analyst: specify what topic/entity to find in the documents.
- For web_researcher: specify the exact search query with relevant technical terms.
- Include context from the conversation if relevant (e.g., "The user previously asked about X, now they want to know Y related to X").
"""

JUDGE_SYSTEM = """\
You are a judge selecting the best task plan. Three analysts each proposed a task plan for the same request.
Pick the ONE most likely to yield the best, most complete answer (most specific, most likely to retrieve the right information).
If two are close, you may merge their strongest parts into one.

Respond with EXACTLY the chosen plan in this format (no extra text):
WORKERS: {workers}
TASKS:
- {workers}: <task instruction>
"""

# Question markers for the routing safety-net (real question vs greeting)
_QUESTION_MARKERS = (
    "thế nào", "thếnào", "như thế nào", "ra sao", "làm sao", "sao", "gì", "vì sao",
    "bao nhiêu", "có phải", "có không", "khi nào", "ở đâu", "ai",
    "how", "what", "why", "when", "where", "which", "who", "explain", "describe",
)


def _is_real_question(query: str) -> bool:
    """Heuristic: is this a genuine information-seeking question (not a greeting)?"""
    q = query.strip()
    ql = q.lower()
    if "?" in q or len(q) > 25:
        return True
    return any(m in ql for m in _QUESTION_MARKERS)


def _mode_block(project: str) -> str:
    """Mode-specific routing guidance appended to the planner prompts.

    project mode  -> prioritize the project's own code / DB / docs.
    free mode (rag_kb) -> learned knowledge (RAG) for internal context;
    for solution / how-to / architecture questions, combine RAG + web research.
    """
    if project:
        return (
            f"\n\nBỐI CẢNH: Đang làm việc với project \"{project}\".\n"
            "- Ưu tiên dữ liệu nội bộ của project: code → code_explorer, DB → db_analyst, tài liệu → rag_analyst.\n"
            "- Chỉ dùng web_researcher khi cần tri thức bên ngoài project.\n"
        )
    return (
        "\n\nBỐI CẢNH: KHÔNG có project — chế độ chung (kho tri thức rag_kb).\n"
        "- rag_analyst: tra cứu ngữ cảnh đã học. Trong free mode nó TRUY CẬP XÉO tất cả project đã học (rag_kb = trang chủ) và sẽ YÊU CẦU USER XÁC NHẬN project trước khi fetch chi tiết.\n"
        "- Dùng rag_analyst cho: 'đã học project nào', 'liệt kê / so sánh / mối liên hệ giữa các project', 'tìm trong kho tri thức'.\n"
        "- web_researcher: tra cứu giải pháp hiện đại, best practice, tài liệu công nghệ.\n"
        "- Câu hỏi về CÁCH LÀM / KIẾN TRÚC / PHƯƠNG ÁN / công nghệ / so sánh: dispatch CẢ rag_analyst (ngữ cảnh liên quan) VÀ web_researcher (giải pháp tốt nhất).\n"
        "- Câu hỏi tri thức chung (sức khỏe, khoa học, sự kiện, định nghĩa, dịch thuật) → web_researcher.\n"
        "- Định hướng: đề xuất giải pháp HIỆN ĐẠI, TỐI GIẢN, ÍT đập code, an toàn; KHÔNG tự ý thay đổi cấu trúc DB.\n"
    )


def _extract_text(content) -> str:
    """Normalize LLM response content (str or list of parts) to a plain string."""
    if isinstance(content, list):
        return " ".join(
            part if isinstance(part, str) else part.get("text", "") for part in content
        ).strip()
    return (content or "").strip()


def _parse_workers(plan: str) -> set[str]:
    """Extract worker names from a plan string."""
    workers_line = ""
    for line in plan.split("\n"):
        if line.strip().upper().startswith("WORKERS:"):
            workers_line = line.split(":", 1)[1].strip()
            break
    if not workers_line or workers_line.lower() == "none":
        return set()
    return {w.strip() for w in workers_line.split(",") if w.strip() in KNOWN_WORKERS}


def _extract_tasks(plan: str) -> dict[str, str]:
    """Extract {worker: task} from a plan string (only for known workers)."""
    tasks: dict[str, str] = {}
    for line in plan.split("\n"):
        m = re.match(r"^\s*-\s*(\w+):\s*(.+)", line.strip())
        if m:
            worker, task = m.group(1), m.group(2)
            if worker in KNOWN_WORKERS:
                tasks[worker] = task
    return tasks


def _force_worker(plan: str, worker: str) -> str:
    """Rewrite a plan so its WORKERS header is a single worker (keeps a matching task line)."""
    task = None
    for line in plan.split("\n"):
        m = re.match(r"^\s*-\s*(\w+):\s*(.+)", line.strip())
        if m and m.group(1) == worker:
            task = m.group(2)
            break
    if not task:
        task = "Answer the user's request."
    return f"WORKERS: {worker}\nTASKS:\n- {worker}: {task}"


def _normalize_final(workers_str: str, task_plan: str) -> str:
    """Build the final plan: assigned workers in the header + voted task body (only assigned workers)."""
    assigned = [w.strip() for w in workers_str.split(",") if w.strip()]
    tasks = {w: t for w, t in _extract_tasks(task_plan).items() if w in assigned}
    lines = [f"WORKERS: {workers_str}", "TASKS:"]
    for w in assigned:
        lines.append(f"- {w}: {tasks.get(w, 'Answer the user\'s request.')}")
    return "\n".join(lines)


def _judge_llm() -> LLM:
    if _JUDGE == "qwen":
        return LLM.OPENAI
    if _JUDGE == "ollama":
        return LLM.OLLAMA
    return LLM.GEMINI  # default: gemini (unbiased — qwen already authored one candidate)


async def _generate_plan(model, query: str, project: str, history: str = "") -> str:
    """Phase 1 — routing plan (worker + task draft). `model` may be a FallbackChatModel."""
    history_block = f"\n{history}\n\n" if history else ""
    prompt = (
        f"Active project: {project or 'none'}\n"
        f"{history_block}\n"
        f"User request: {query}\n\n"
        f"Create the execution plan:"
    )
    response = await asyncio.to_thread(model.invoke, [
        {"role": "system", "content": ROUTER_SYSTEM + _mode_block(project)},
        {"role": "user", "content": prompt},
    ])
    return _extract_text(response.content)


async def _generate_task(llm_name: LLM, workers: str, query: str, project: str, history: str = "") -> str:
    """Phase 2 — task instruction for an already-decided worker set."""
    llm = get_llm(llm_name)
    history_block = f"\n{history}\n\n" if history else ""
    prompt = (
        f"Active project: {project or 'none'}\n"
        f"{history_block}\n"
        f"User request: {query}\n\n"
        f"Write the best task instruction for the assigned worker(s):"
    )
    response = await asyncio.to_thread(llm.invoke, [
        {"role": "system", "content": TASK_SYSTEM.format(workers=workers) + _mode_block(project)},
        {"role": "user", "content": prompt},
    ])
    return _extract_text(response.content)


async def _judge_best_task(llm_name: LLM, workers: str, query: str, candidates: dict[str, str]) -> str:
    """Phase 3 — judge picks the best of the candidate task plans."""
    llm = get_llm(llm_name)
    cand_block = "\n\n".join(f"--- Candidate {i + 1} ---\n{c}" for i, c in enumerate(candidates.values()))
    prompt = (
        f"User request: {query}\n\n"
        f"Assigned worker(s): {workers}\n\n"
        f"Candidate task plans:\n\n{cand_block}\n\n"
        f"Select the best plan:"
    )
    response = await asyncio.to_thread(llm.invoke, [
        {"role": "system", "content": JUDGE_SYSTEM.format(workers=workers)},
        {"role": "user", "content": prompt},
    ])
    return _extract_text(response.content)


async def plan_node(state: TeamState) -> dict[str, Any]:
    """Planner: qwen routes the worker, then (optionally) 3 LLMs vote on the best task."""
    query = state.get("user_query", "")
    project = state.get("project", "")
    history = state.get("history", "")

    # ── Phase 1: route the worker (qwen -> gemini -> gemma auto-fallback) ──
    qwen_plan = await _generate_plan(get_backbone_llm(), query, project, history)
    workers = _parse_workers(qwen_plan)

    # Safety-net fallback: qwen said "none" but it's a genuine question → web_researcher
    if not workers and _is_real_question(query):
        workers = {"web_researcher"}
        qwen_plan = _force_worker(qwen_plan, "web_researcher")
        logger.info("[PLANNER] Routing fallback: qwen='none' but real question → web_researcher")

    logger.info(f"[PLANNER] qwen routed → {sorted(workers) if workers else 'none'}")

    if not workers:
        return {"plan": "WORKERS: none", "plan_votes": {"qwen": qwen_plan}, "routed": []}

    workers_str = ", ".join(sorted(workers))

    if not _ENSEMBLE:
        # Fast mode: use qwen's task directly, no vote.
        final = _normalize_final(workers_str, qwen_plan)
        logger.info(f"[PLANNER] fast mode (no ensemble), final: {final[:200]}")
        return {"plan": final, "plan_votes": {"qwen": qwen_plan}, "routed": sorted(workers)}

    # ── Phase 2: GEMINI + OLLAMA propose tasks for the chosen worker(s) ──
    gem_task, ollama_task = await asyncio.gather(
        _generate_task(LLM.GEMINI, workers_str, query, project, history),
        _generate_task(LLM.OLLAMA, workers_str, query, project, history),
        return_exceptions=True,
    )
    candidates = {
        "qwen": qwen_plan,
        "gemini": gem_task if isinstance(gem_task, str) and gem_task else qwen_plan,
        "ollama": ollama_task if isinstance(ollama_task, str) and ollama_task else qwen_plan,
    }
    for name, val in candidates.items():
        logger.info(f"[PLANNER] task candidate {name}: {val[:160]}")

    # ── Phase 3: JUDGE picks the best task plan ──
    try:
        best = await _judge_best_task(_judge_llm(), workers_str, query, candidates)
        logger.info(f"[PLANNER] judge ({_JUDGE}) picked: {best[:160]}")
    except Exception as e:
        logger.warning(f"[PLANNER] judge failed ({e}); using qwen task")
        best = qwen_plan

    final = _normalize_final(workers_str, best)
    logger.info(f"[PLANNER] FINAL plan: {final[:200]}")
    return {"plan": final, "plan_votes": candidates, "routed": sorted(workers)}
