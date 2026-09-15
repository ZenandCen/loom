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
    """Code worker: RAG with parent expansion + read key files → LLM explains."""
    task = state.get("task", "")
    project = state.get("project", "")
    logger.info(f"[CODE WORKER] project={project} task={task[:100]}")

    from agent.tools import CODE_BASE_DIR, _get_project_dir, get_active_collection
    if project:
        project_dir = CODE_BASE_DIR / project
        if not project_dir.exists():
            project_dir = _get_project_dir()
    else:
        project_dir = _get_project_dir()
    context_parts = []

    # 1. Directory structure (top-level only)
    try:
        entries = sorted(project_dir.iterdir())
        tree = "Project structure:\n"
        for e in entries[:30]:
            if e.name.startswith(".") or e.name in (".venv", "venv"):
                continue
            if e.is_dir():
                sub_files = list(e.iterdir())[:5]
                tree += f"  {e.name}/\n"
                for sf in sub_files:
                    tree += f"    {sf.name}\n"
            else:
                tree += f"  {e.name}\n"
        context_parts.append(tree)
    except Exception:
        pass

    # 2. RAG search with parent expansion for code
    try:
        from rag.retrieval import retrieve_with_parents
        collection = get_active_collection()
        code_result = retrieve_with_parents(task, collection_name=collection, k=8)
        code_blocks = code_result.context_blocks
        if code_blocks:
            context_parts.append("Relevant code (with full file context):\n" + "\n\n---\n\n".join(code_blocks[:6]))
    except Exception as e:
        logger.warning(f"Code RAG parent search failed, falling back: {e}")
        try:
            from rag.indexing import get_vectorstore
            collection = get_active_collection()
            vs = get_vectorstore(collection)
            code_results = vs.similarity_search(task, k=8)
            code_chunks = []
            for doc in code_results:
                if doc.metadata.get("type") == "code":
                    src = doc.metadata.get("source", "")
                    rel = src.replace(str(project_dir) + "/", "") if src.startswith(str(project_dir)) else src
                    code_chunks.append(f"[{rel}]\n{doc.page_content[:600]}")
            if code_chunks:
                context_parts.append("Relevant code (from RAG):\n" + "\n\n---\n\n".join(code_chunks[:5]))
        except Exception as e2:
            logger.warning(f"Code RAG fallback also failed: {e2}")

    # 3. If task mentions specific files, read them
    import re
    mentioned_files = re.findall(r'[\w/._-]+\.py', task)
    for mf in mentioned_files[:2]:
        fpath = project_dir / mf
        if fpath.exists():
            try:
                lines = fpath.read_text(encoding="utf-8").splitlines(keepends=True)
                head = lines[:100]
                context_parts.append(f"--- {mf} (first 100 of {len(lines)} lines) ---\n{''.join(head)}")
            except Exception:
                continue

    context = "\n\n".join(context_parts) if context_parts else "(no files found)"

    # Collect sources
    sources: list[str] = []
    for part in context_parts:
        if part.startswith("["):
            src = part.split("]")[0].lstrip("[")
            if src and src not in sources:
                sources.append(src)

    result = _llm_summarize(
        system="You are a code architect. Explain how the code works based on the provided file structure, RAG code chunks (with full file context), and file samples. Reference file:line. Be structured and concise. Answer in the same language as the task.",
        context=context,
        task=task,
    )
    logger.info(f"[CODE WORKER] Done: {len(result)} chars, {len(sources)} sources")
    return {"code_result": result, "code_sources": sources}


def db_worker(state: dict) -> dict:
    """DB worker: query schema → LLM formats."""
    task = state.get("task", "")
    logger.info(f"[DB WORKER] {task[:100]}")

    from agent.tools import list_tables
    try:
        tables_info = list_tables.invoke({})
    except Exception as e:
        tables_info = f"Error: {e}"

    result = _llm_summarize(
        system="You are a database analyst. Describe the schema based on the provided table list. Use markdown tables.",
        context=f"Tables:\n{tables_info[:10000]}",
        task=task,
    )
    logger.info(f"[DB WORKER] Done: {len(result)} chars")
    return {"db_result": result}


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
