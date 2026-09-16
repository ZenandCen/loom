"""Worker nodes — specialized retrieval + single LLM call.

Each worker does:
1. Direct retrieval (RAG with parent expansion / file read / DB query / web search)
2. LLM call (single or iterative map-reduce for large contexts)

This avoids context bloat from multi-turn tool calling agents.
"""

import logging
import os
from pathlib import Path

from utils.models import LLM, get_llm

logger = logging.getLogger(__name__)

_worker_model = get_llm(LLM.OPENAI)

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
        HumanMessage(content=f"Context:\n{context}\n\nTask: {task}\n\nAnswer concisely with references."),
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
        HumanMessage(content=f"Summarized context:\n{combined}\n\nTask: {task}\n\nAnswer concisely with references."),
    ])
    content = response.content
    if isinstance(content, list):
        content = " ".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    return content.strip()


def rag_worker(state: dict) -> dict:
    """RAG worker: retrieve with parent expansion → LLM summarizes."""
    task = state.get("task", "")
    logger.info(f"[RAG WORKER] {task[:100]}")

    from agent.tools import get_active_collection

    collection = get_active_collection()
    sources: list[str] = []
    try:
        from rag.retrieval import retrieve_with_parents
        result = retrieve_with_parents(task, collection_name=collection, k=8)
        context_blocks = result.context_blocks
        sources = result.sources
        context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no results found)"
    except Exception as e:
        logger.warning(f"Parent retrieval failed, falling back to basic: {e}")
        from rag.indexing import get_vectorstore
        vs = get_vectorstore(collection)
        results = vs.similarity_search(task, k=8)
        context_parts = []
        for i, doc in enumerate(results, 1):
            src = doc.metadata.get("source", "?")
            fname = Path(src).name
            context_parts.append(f"[{i}] {fname}:\n{doc.page_content[:800]}")
            if fname not in sources:
                sources.append(fname)
        context = "\n\n---\n\n".join(context_parts) if context_parts else "(no results found)"

    result = _llm_summarize(
        system="You are a knowledge base analyst. Answer based ONLY on the provided context. Cite sources as [1], [2]. If info is missing, say so.",
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
        list_directory, read_file, search_code,
        CODE_BASE_DIR, _get_project_dir,
    )
    from rag.tool import rag_query

    if project:
        project_dir = CODE_BASE_DIR / project
        if not project_dir.exists():
            project_dir = _get_project_dir()
    else:
        project_dir = _get_project_dir()

    from langchain.agents import create_agent

    agent = create_agent(
        model=_worker_model,
        tools=[list_directory, read_file, search_code, rag_query],
        system_prompt=(
            f"You are a code architect analyzing the project: {project_dir.name}\n"
            "Workflow:\n"
            "1. Start with `list_directory` to understand the project structure.\n"
            "2. Use `search_code` to find relevant functions, classes, or patterns.\n"
            "3. Use `read_file` to examine specific files (use start_line/end_line for large files).\n"
            "4. Use `rag_query` for semantic search when you don't know which file to look at.\n"
            "5. Follow imports and dependencies: if a file references another module, read that too.\n\n"
            "Rules:\n"
            "- Maximum 6 tool calls. Prioritize the most important files.\n"
            "- Always reference file:line in your answer.\n"
            "- For architecture questions, focus on entry points, key abstractions, and data flow.\n"
            "- For 'how does X work' questions, trace the code path step by step.\n"
            "- Be structured: use headings, bullet points, and code snippets.\n"
            "- Answer in the same language as the user's question.\n"
            "- If the codebase is large, focus on the most relevant 3-5 files."
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
            "You are a database analyst for a PostgreSQL database.\n"
            "Workflow:\n"
            "1. Start by calling `list_tables` to see all table names.\n"
            "2. Use `describe_table('<name>')` to see full columns of specific tables you need.\n"
            "3. Based on the schema, construct SELECT queries using `query_database` to gather data.\n"
            "4. ONLY use SELECT queries. Never use INSERT, UPDATE, DELETE, DROP, ALTER, or any data modification.\n"
            "5. Always add LIMIT 50 to queries that might return many rows.\n"
            "5. For relationship questions, use JOIN queries.\n"
            "6. For count/aggregation questions, use COUNT, SUM, AVG, etc.\n"
            "7. If a query fails, adjust and retry once. If it still fails, explain the error.\n\n"
            "Rules:\n"
            "- Be concise. Use markdown tables for data.\n"
            "- Answer in the same language as the user's question.\n"
            "- If the question is ambiguous, make a reasonable assumption and state it.\n"
            "- Maximum 3 tool calls to answer. If you can't solve it, summarize what you found."
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
        system="You are a web research analyst. Summarize findings with source URLs. Be concise.",
        context=str(search_results)[:15000],
        task=task,
    )
    import re
    sources = re.findall(r'https?://\S+', str(search_results))[:5]
    logger.info(f"[WEB WORKER] Done: {len(result)} chars, {len(sources)} sources")
    return {"web_result": result, "web_sources": sources}
