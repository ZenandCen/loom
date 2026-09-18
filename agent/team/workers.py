"""Worker nodes — specialized retrieval + single LLM call.

Each worker does:
1. Direct retrieval (RAG with parent expansion / file read / DB query / web search)
2. LLM call (single or iterative map-reduce for large contexts)

This avoids context bloat from multi-turn tool calling agents.
"""

import logging
import os
import re
from pathlib import Path

from utils.models import LLM, get_llm, get_backbone_llm

logger = logging.getLogger(__name__)

# Backbone answer model with automatic fallback: qwen -> gemini -> gemma(local).
_worker_model = get_backbone_llm()

# Iterative summarization config
CTX_THRESHOLD = int(os.getenv("LOOM_CTX_THRESHOLD", "30000"))
CTX_WINDOW = int(os.getenv("LOOM_CTX_WINDOW", "8000"))
CTX_MAX_WINDOWS = int(os.getenv("LOOM_CTX_MAX_WINDOWS", "5"))


def _llm_summarize(system: str, context: str, task: str) -> str:
    """LLM call: single-shot if small, iterative map-reduce if context > threshold."""
    if len(context) <= CTX_THRESHOLD:
        return _single_llm_call(system, context, task)
    return _iterative_llm_call(system, context, task)


def _single_llm_call(system: str, context: str, task: str) -> str:
    """Single LLM call: system + context + task → answer."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Context:\n{context}\n\nTask: {task}\n\nAnswer comprehensively with specific references. Include all relevant details, names, and specifics from the context. Do NOT be brief."),
    ]
    response = _worker_model.invoke(messages)
    content = response.content
    if isinstance(content, list):
        content = " ".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    return content.strip()


def _split_windows(context: str, window_size: int) -> list[str]:
    """Split context into windows of ~window_size chars (on paragraph boundaries)."""
    windows = []
    while len(context) > window_size:
        # Try to split at a paragraph boundary near the target
        target = window_size
        split_at = context.rfind("\n\n", 0, target + 2000)
        if split_at < window_size // 2:
            split_at = target
        windows.append(context[:split_at])
        context = context[split_at:].lstrip("\n")
    if context.strip():
        windows.append(context)
    # Cap at max windows
    if len(windows) > CTX_MAX_WINDOWS:
        windows = windows[:CTX_MAX_WINDOWS]
    return windows


def _iterative_llm_call(system: str, context: str, task: str) -> str:
    """Iterative map-reduce: summarize each window, then answer from summaries.

    Used when context exceeds CTX_THRESHOLD to avoid exceeding LLM token limits.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    windows = _split_windows(context, CTX_WINDOW)
    logger.info(f"[ITERATIVE] Context {len(context)} chars → {len(windows)} windows")

    # Map: summarize each window
    summaries = []
    for i, window in enumerate(windows):
        resp = _worker_model.invoke([
            SystemMessage(content="You are a technical summarizer. Extract all key details, function names, code patterns, data structures, relationships, and facts. Be thorough — this summary will be used to answer a question later."),
            HumanMessage(content=f"Section {i+1}/{len(windows)}:\n{window}"),
        ])
        content = resp.content
        if isinstance(content, list):
            content = " ".join(p if isinstance(p, str) else p.get("text", "") for p in content)
        summaries.append(content.strip())

    # Reduce: answer from summaries
    combined = "\n\n---\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(summaries))
    response = _worker_model.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Summarized context:\n{combined}\n\nTask: {task}\n\nAnswer comprehensively with specific references. Include all relevant details, names, and specifics. Do NOT be brief."),
    ])
    content = response.content
    if isinstance(content, list):
        content = " ".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    return content.strip()


def _basic_retrieve(task: str, collection: str, sources: list[str]) -> str:
    """Single-collection retrieval with parent expansion (+ basic fallback)."""
    from rag.retrieval import retrieve_with_parents
    try:
        result = retrieve_with_parents(task, collection_name=collection, k=15)
        context_blocks = result.context_blocks
        for s in result.sources:
            if s not in sources:
                sources.append(s)
        return "\n\n---\n\n".join(context_blocks) if context_blocks else "(no results found)"
    except Exception as e:
        logger.warning(f"Parent retrieval failed, falling back to basic: {e}")
        from rag.indexing import get_vectorstore
        vs = get_vectorstore(collection)
        results = vs.similarity_search(task, k=15)
        context_parts = []
        for i, doc in enumerate(results, 1):
            src = doc.metadata.get("source", "?")
            fname = Path(src).name
            context_parts.append(f"[{i}] {fname}:\n{doc.page_content[:800]}")
            if fname not in sources:
                sources.append(fname)
        return "\n\n---\n\n".join(context_parts) if context_parts else "(no results found)"


_CONFIRM_WORDS = {
    "ok", "okay", "yes", "y", "yep", "confirm", "proceed", "go", "run",
    "đồng ý", "dong y", "dongyi", "tiến hành", "tien hanh", "duyệt", "duyet", "a", "1",
}
_ALL_WORDS = {
    "all", "all projects", "every project", "everything", "tất cả", "tat ca", "tatca",
    "tất cả project", "tat ca project", "mọi project", "moi project", "toàn bộ", "toan bo",
}
_DECLINE_WORDS = {
    "no", "n", "nope", "cancel", "stop", "hủy", "huy", "không", "khong", "bỏ qua", "bo qua",
}


def _parse_scope_decision(decision, candidate_names: list[str], all_project_names: list[str]) -> list[str]:
    """Map the user's HITL reply to a confirmed list of collections.

    - confirm (ok/yes/đồng ý...)   -> proposed candidates
    - all (all/tất cả/...)          -> every learned project
    - decline (no/hủy/cancel...)    -> [] (skip cross-search)
    - explicit names                -> those matching known projects (exact or fuzzy)
    - anything else                 -> proposed candidates (safe default)
    """
    if decision is None:
        return list(candidate_names)
    s = str(decision).strip().lower()
    if not s:
        return list(candidate_names)
    if s in _CONFIRM_WORDS:
        return list(candidate_names)
    if s in _ALL_WORDS:
        return list(all_project_names)
    if s in _DECLINE_WORDS:
        return []

    tokens = [t for t in re.split(r"[,\s]+", s) if t]
    exact = {t for t in tokens if t in {n.lower() for n in all_project_names}}
    if exact:
        return [n for n in all_project_names if n.lower() in exact]
    fuzzy = [
        n for n in all_project_names
        if any(tok in n.lower() or n.lower() in tok for tok in tokens)
    ]
    if fuzzy:
        return fuzzy
    return list(candidate_names)


def rag_worker(state: dict) -> dict:
    """RAG worker: retrieve (single collection, or cross-collection in free mode) → LLM summarizes.

    In free mode (rag_kb "homepage"), retrieval is query-driven across all learned
    projects with a human-in-the-loop confirmation of which projects to fetch.
    In project mode, it only searches that project's own collection (unchanged).
    """
    task = state.get("task", "")
    logger.info(f"[RAG WORKER] {task[:100]}")

    from agent.tools import get_active_collection
    from rag.config import get_rag_settings

    collection = get_active_collection()
    default_collection = get_rag_settings().default_collection
    free_mode = (collection == default_collection)

    sources: list[str] = []
    scope_note = ""
    context = ""

    if free_mode:
        from rag.indexing import list_collections
        learned = list_collections(min_count=1)
        learned_names = [name for name, _ in learned]
        if not learned_names:
            context = _basic_retrieve(task, collection, sources)
        else:
            from rag.retrieval import discover_projects, retrieve_across_collections
            candidates = discover_projects(task, learned_names)
            candidate_names = [name for name, _ in candidates]
            if not candidate_names:
                context = "(no relevant projects found in the knowledge base)"
            else:
                # Human-in-the-loop: confirm which projects to fetch.
                # NOTE: interrupt() must NOT sit inside a try/except — it pauses the graph.
                from langgraph.types import interrupt
                decision = interrupt({
                    "type": "rag_scope_confirm",
                    "candidates": candidate_names,
                    "all_projects": learned_names,
                    "query": task,
                })
                confirmed = _parse_scope_decision(decision, candidate_names, learned_names)
                if not confirmed:
                    context = "(User declined the cross-project search.)"
                else:
                    try:
                        scope_note = "Projects searched: " + ", ".join(f"`{c}`" for c in confirmed) + "."
                        res = retrieve_across_collections(task, confirmed, k_per_collection=8, max_total=40)
                        context = "\n\n".join(res.context_blocks) if res.context_blocks else "(no matching content found in the selected projects)"
                        for s in res.sources:
                            if s not in sources:
                                sources.append(s)
                    except Exception as e:
                        logger.warning(f"Cross-collection retrieval failed: {e}")
                        context = f"(cross-project retrieval error: {e})"
    else:
        context = _basic_retrieve(task, collection, sources)

    if scope_note:
        context = scope_note + "\n\n" + context

    result = _llm_summarize(
        system=(
            "You are a knowledge base analyst. Answer based ONLY on the provided context. Cite sources as [1], [2]. "
            "In free mode the evidence is grouped by project under '## Project: <name>' headers — attribute facts to "
            "their project, and when asked to list/compare projects, describe each and analyse relationships "
            "(shared tech, dependencies, same domain, complementary).\n"
            "CRITICAL RULES:\n"
            "- Be Exhaustive: Include ALL relevant details from the context. Do not summarize away specifics.\n"
            "- Be Specific: Name exact file names, function names, table names, config keys, technology choices.\n"
            "- Holistic View: Show how pieces connect. When describing a project's architecture, show the data flow between components.\n"
            "- Architecture Awareness: Respect and reference the EXISTING patterns in the documents. Do not suggest patterns that contradict what's documented.\n"
            "- If comparing projects: use a table with columns for each dimension (tech stack, architecture, data model, integrations, purpose).\n"
            "- If info is missing, say exactly WHAT is missing (not just 'info not found').\n"
            "- End with '## Key Takeaways' (3-5 bullets) and '## Suggested Next Steps' (2-3 items)."
        ),
        context=context,
        task=task,
    )
    logger.info(f"[RAG WORKER] Done: {len(result)} chars, {len(sources)} sources")
    return {"rag_result": result, "rag_sources": sources}


def code_worker(state: dict) -> dict:
    """Code worker: ReAct agent that explores codebase iteratively."""
    task = state.get("task", "")
    project = state.get("project", "")
    logger.info(f"[CODE WORKER] project={project} task={task[:100]}")

    from agent.tools import (
        list_directory, read_file, search_code, set_project,
        CODE_BASE_DIR, _get_project_dir,
    )
    from rag.tool import rag_query

    if project:
        project_dir = CODE_BASE_DIR / project
        if project_dir.exists():
            # The file tools resolve against the process-global active project, so
            # sync it to the project this worker was given (avoids reading the wrong tree).
            set_project.invoke({"path": project})
        else:
            project_dir = _get_project_dir()
    else:
        project_dir = _get_project_dir()

    from langchain.agents import create_agent

    agent = create_agent(
        model=_worker_model,
        tools=[list_directory, read_file, search_code, rag_query],
        system_prompt=(
            f"You are a senior code architect analyzing the project: {project_dir.name}\n\n"
            "Workflow:\n"
            "1. Start with `list_directory` to understand the project structure and architecture.\n"
            "2. Use `search_code` to find relevant functions, classes, or patterns.\n"
            "3. Use `read_file` to examine specific files (use start_line/end_line for large files).\n"
            "4. Use `rag_query` for semantic search when you don't know which file to look at.\n"
            "5. Follow imports and dependencies: if a file references another module, read that too.\n"
            "6. Trace callers: when you find a function/class, search for who uses it.\n\n"
            "CRITICAL RULES:\n"
            "- Architecture First: Understand the EXISTING architecture before answering. Identify the project's design patterns, layer structure, and data flow. Do NOT introduce new flow types or patterns that don't exist in the codebase.\n"
            "- Reuse Before Creating: Always search for existing utilities, helpers, and patterns before suggesting new code. If a similar function exists, reference it. A small duplicate beats a wrong abstraction.\n"
            "- Holistic View: Show how the code integrates with the broader system. Trace dependencies upstream and downstream. Don't just describe isolated files — explain how they connect.\n"
            "- Follow Existing Conventions: Match the project's naming, structure, and coding style. If the project uses a specific pattern (e.g., repository pattern, service layer), follow it.\n"
            "- Be Specific: Name exact files, functions, classes, line numbers. Never use vague references like 'the service layer' without naming the actual file.\n"
            "- Maximum 8 tool calls. Prioritize the most important files.\n"
            "- Always reference file:line in your answer.\n"
            "- For architecture questions: map entry points → key abstractions → data flow → external dependencies.\n"
            "- For 'how does X work' questions: trace the code path step by step with file:line references.\n"
            "- Be structured: use headings (##, ###), bullet points, code snippets, and tables where appropriate.\n"
            "- End with a '## Key Takeaways' section (3-5 bullet points) and '## Suggested Next Steps' (2-3 items).\n"
            "- Answer in the same language as the user's question."
        ),
    )

    try:
        result = agent.invoke({"messages": [("user", task)]}, config={"recursion_limit": 15})
        content = result["messages"][-1].content
    except Exception as e:
        logger.error(f"[CODE WORKER] Agent failed: {e}")
        content = f"Error: {e}"
        result = {}

    # Extract sources from the agent's tool calls
    sources: list[str] = []
    for msg in result.get("messages", []):
        if hasattr(msg, "content") and msg.type == "tool":
            if hasattr(msg, "name") and msg.name in ("read_file", "search_code"):
                pass  # sources are embedded in the answer already

    logger.info(f"[CODE WORKER] Done: {len(content)} chars")
    return {"code_result": content, "code_sources": sources}


def db_worker(state: dict) -> dict:
    """DB worker: ReAct agent with list_tables + query_database tools."""
    task = state.get("task", "")
    logger.info(f"[DB WORKER] {task[:100]}")

    from agent.tools import list_tables, describe_table, query_database, _get_query_dsn

    dsn = _get_query_dsn()
    if not dsn:
        return {"db_result": "Error: No project database configured. Use `db <dsn>` in Slack to set the database connection."}

    from langchain.agents import create_agent

    agent = create_agent(
        model=_worker_model,
        tools=[list_tables, describe_table, query_database],
        system_prompt=(
            "You are a senior database analyst for a PostgreSQL database.\n"
            "Workflow:\n"
            "1. Start by calling `list_tables` to see all table names.\n"
            "2. Use `describe_table('<name>')` to see full columns of specific tables you need.\n"
            "3. Based on the schema, construct SELECT queries using `query_database` to gather data.\n"
            "4. ONLY use SELECT queries. Never use INSERT, UPDATE, DELETE, DROP, ALTER, or any data modification.\n"
            "5. Always add LIMIT 50 to queries that might return many rows.\n"
            "6. For relationship questions, use JOIN queries and explain the relationship (1:1, 1:N, M:N).\n"
            "7. For count/aggregation questions, use COUNT, SUM, AVG, etc.\n"
            "8. If a query fails, adjust and retry once. If it still fails, explain the error.\n\n"
            "CRITICAL RULES:\n"
            "- Architecture First: Understand the SCHEMA before answering. Identify entity relationships, normalization patterns, and data flow.\n"
            "- Holistic View: Show how tables CONNECT. When describing a table, mention its foreign keys and what they reference. Don't describe tables in isolation.\n"
            "- Be Exhaustive: Include ALL relevant tables, columns, constraints, and relationships. If there are 5 related tables, describe ALL 5.\n"
            "- Be Specific: Name exact column types, constraints (PK, FK, UNIQUE, NOT NULL), and indexes. Never say 'several columns' — list them.\n"
            "- Data Quality: Note any NULL values, default values, or unusual data patterns you observe.\n"
            "- Use markdown tables for schema descriptions. Show actual data samples (first 3-5 rows) when relevant.\n"
            "- End with '## Key Takeaways' (3-5 bullets) and '## Suggested Next Steps' (2-3 items).\n"
            "- Answer in the same language as the user's question.\n"
            "- If the question is ambiguous, make a reasonable assumption and state it explicitly.\n"
            "- Maximum 7 tool calls. Prioritize the most important tables."
        ),
    )

    try:
        result = agent.invoke({"messages": [("user", task)]}, config={"recursion_limit": 15})
        content = result["messages"][-1].content
    except Exception as e:
        logger.error(f"[DB WORKER] Agent failed: {e}")
        content = f"Error: {e}"

    logger.info(f"[DB WORKER] Done: {len(content)} chars")
    return {"db_result": content}


def web_worker(state: dict) -> dict:
    """Web worker: search web → LLM summarizes."""
    task = state.get("task", "")
    logger.info(f"[WEB WORKER] {task[:100]}")

    from agent.tools import tavily_search
    try:
        search_results = tavily_search.invoke({"query": task})
    except Exception as e:
        search_results = f"Search error: {e}"

    result = _llm_summarize(
        system="You are a web research analyst. Summarize findings comprehensively with source URLs. Include all relevant details, specific facts, numbers, and names. Do NOT be brief.",
        context=str(search_results)[:15000],
        task=task,
    )
    import re
    sources = re.findall(r'https?://\S+', str(search_results))[:5]
    logger.info(f"[WEB WORKER] Done: {len(result)} chars, {len(sources)} sources")
    return {"web_result": result, "web_sources": sources}
