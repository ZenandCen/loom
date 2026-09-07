"""LangGraph node functions for the RAG pipeline.

Each node is a pure function: (state) -> partial_state_update.
Nodes are grouped by responsibility:
    - Routing: decide whether to retrieve or answer directly
    - Retrieval: fetch documents from the vector store
    - Grading: evaluate document relevance and answer quality
    - Generation: produce the final answer
    - Rewrite: improve the query for better retrieval
"""

import logging
from typing import Literal

from langchain_core.documents import Document

from rag.config import get_rag_settings
from rag.enums import (
    GradeDecision,
    HallucinationResult,
    PipelineLevel,
    QualityCheckDecision,
    QualityResult,
    RouteDecision,
)
from rag.retrieval import get_retriever, rerank_documents
from rag.state import RAGState
from utils.models import LLM, get_llm

logger = logging.getLogger(__name__)

# Shared LLM instance for all RAG nodes (temperature=0 for determinism)
_llm = get_llm(LLM.OPENAI)


# ═══════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════


def route_question(state: RAGState) -> RouteDecision:
    """Decide if the question needs external document retrieval.

    Uses LLM to classify whether the question is general knowledge
    (answer directly) or requires specific knowledge base lookup.

    Args:
        state: Current RAG pipeline state.

    Returns:
        RouteDecision.RETRIEVE or RouteDecision.DIRECT_ANSWER
    """
    prompt = (
        "Decide if this question needs external documents to answer. "
        "If it's a general knowledge question or greeting, answer 'direct_answer'. "
        "If it requires specific information from a knowledge base, answer 'retrieve'.\n\n"
        f"Question: {state['question']}\n\n"
        "Answer with ONLY 'retrieve' or 'direct_answer':"
    )
    decision = _llm.invoke(prompt).content.strip().lower()
    result = RouteDecision.RETRIEVE if "retrieve" in decision else RouteDecision.DIRECT_ANSWER
    logger.info(f"Route: {result.value}")
    return result


# ═══════════════════════════════════════════════════════════════
# Retrieval
# ═══════════════════════════════════════════════════════════════


def retrieve(state: RAGState) -> dict:
    """Retrieve relevant documents from the vector store and rerank them.

    Uses over-fetching strategy: retrieves 3x the target count, then
    reranks down to the final k. This gives the reranker more candidates
    to work with, significantly improving precision.

    Example: target k=4 → fetch 12 → rerank to top 4.

    Args:
        state: Current RAG pipeline state (uses `question` field).

    Returns:
        Partial state update with `documents` and `question`.
    """
    question = state["question"]
    settings = get_rag_settings()

    # Over-fetch: retrieve 3x more candidates than needed
    # This gives the reranker a larger pool to select from
    overfetch_k = settings.retrieval_k * 3
    retriever = get_retriever(k=overfetch_k)
    docs = retriever.invoke(question)

    # Rerank: select the truly most relevant from the larger pool
    docs = rerank_documents(question, docs, top_k=settings.retrieval_k)

    logger.info(f"Retrieved {len(docs)} documents (from {overfetch_k} candidates)")
    return {"documents": docs, "question": question}


def multi_source_retrieve(state: RAGState) -> dict:
    """Retrieve from multiple sources and merge with deduplication.

    Currently supports vector store. Web search can be added
    as an additional source.

    Args:
        state: Current RAG pipeline state.

    Returns:
        Partial state update with merged `documents`.
    """
    question = state["question"]
    settings = get_rag_settings()
    all_docs: list[Document] = []

    # Source 1: Vector store
    try:
        vector_retriever = get_retriever()
        all_docs.extend(vector_retriever.invoke(question))
    except Exception as e:
        logger.warning(f"Vector retrieval failed: {e}")

    # Source 2: Web search (using Tavily)
    try:
        web_docs = all_tools.tavily_search.invoke(question)
        all_docs.extend(web_docs)
    except Exception as e:
        logger.warning(f"Web search failed: {e}")

    # Deduplicate by content prefix
    seen: set[str] = set()
    unique_docs: list[Document] = []
    for doc in all_docs:
        key = doc.page_content[:200]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    logger.info(f"Multi-source: {len(unique_docs)} unique docs")
    return {"documents": unique_docs[: settings.retrieval_k]}


# ═══════════════════════════════════════════════════════════════
# Grading (conditional edge routers)
# ═══════════════════════════════════════════════════════════════


def grade_documents(state: RAGState) -> GradeDecision:
    """Grade retrieved documents for relevance to the question.

    If no documents were retrieved, automatically triggers rewrite.
    Otherwise, uses LLM to assess if the top documents contain information
    that can answer the question.

    Uses full content of top documents (up to 1000 chars each) for
    a more accurate relevance assessment than truncated snippets.

    Args:
        state: Current RAG pipeline state.

    Returns:
        GradeDecision.GENERATE if docs are relevant, GradeDecision.REWRITE if not.
    """
    if not state.get("documents"):
        logger.info("No documents retrieved → rewrite")
        return GradeDecision.REWRITE

    # Use top 3 docs, up to 1000 chars each (enough to judge relevance)
    context = "\n\n".join(
        f"[Doc {i+1}]: {d.page_content[:1000]}" for i, d in enumerate(state["documents"][:3])
    )
    prompt = (
        "Given the question and retrieved documents, determine if the documents "
        "contain relevant information to answer the question.\n\n"
        f"Question: {state['question']}\n\n"
        f"Retrieved Documents:\n{context}\n\n"
        "Consider: Do these documents directly address the topic of the question? "
        "Are they on the right subject, or are they about something else entirely?\n\n"
        "Answer 'yes' if the documents are relevant, 'no' if they are not:"
    )
    result = _llm.invoke(prompt).content.strip().lower()
    decision = GradeDecision.GENERATE if "yes" in result else GradeDecision.REWRITE
    logger.info(f"Grade: {decision.value}")
    return decision


def check_hallucination(state: RAGState) -> Literal["check_quality", GradeDecision.REWRITE]:
    """Verify that the generated answer is grounded in the retrieved context.

    Detects hallucinations: claims in the answer that are not supported
    by the retrieved documents.

    Uses full document content (up to 1500 chars per doc) to give the
    LLM enough context to verify each claim in the answer.

    Args:
        state: Current RAG pipeline state (uses `documents` and `generation`).

    Returns:
        "check_quality" if grounded, GradeDecision.REWRITE if hallucinated.
    """
    context = "\n\n".join(
        f"[{i+1}]: {d.page_content[:1500]}" for i, d in enumerate(state.get("documents", [])[:3])
    )
    prompt = (
        "Check if the answer is grounded in the provided context. "
        "Look for specific claims, numbers, names, or facts in the answer "
        "that are NOT present in the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Answer to check: {state.get('generation', '')}\n\n"
        "If every factual claim in the answer is supported by the context, answer 'grounded'. "
        "If the answer contains ANY unsupported claim, answer 'not_grounded':"
    )
    result = _llm.invoke(prompt).content.strip().lower()
    # Check "not_grounded" first to avoid substring match with "grounded"
    if "not_grounded" in result:
        logger.info("Hallucination detected → rewrite")
        return GradeDecision.REWRITE
    logger.info("Answer grounded → check quality")
    return "check_quality"


def check_answer_quality(state: RAGState) -> QualityCheckDecision:
    """Assess whether the answer is useful and complete for the question.

    Catches answers that are technically grounded but still inadequate
    (too vague, incomplete, or off-topic).

    Args:
        state: Current RAG pipeline state (uses `question` and `generation`).

    Returns:
        QualityCheckDecision.FINISH if useful, QualityCheckDecision.REWRITE if not.
    """
    prompt = (
        "Is this answer useful and complete for the question?\n\n"
        f"Question: {state['question']}\n\n"
        f"Answer: {state.get('generation', '')}\n\n"
        "Answer 'useful' if adequate, 'not_useful' if incomplete/vague/off-topic:"
    )
    result = _llm.invoke(prompt).content.strip().lower()
    # Check "not_useful" first to avoid substring match with "useful"
    if "not_useful" in result:
        logger.info("Answer not useful → rewrite")
        return QualityCheckDecision.REWRITE
    logger.info("Answer quality OK → finish")
    return QualityCheckDecision.FINISH


# ═══════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════


def generate(state: RAGState) -> dict:
    """Generate an answer using retrieved documents as context.

    Instructs the LLM to answer ONLY from the provided context
    and to acknowledge when information is insufficient.

    Each document in the context is prefixed with its source location
    (file + page) so the LLM can reference specific parts, and so
    the final sources list includes page-level citations.

    Args:
        state: Current RAG pipeline state (uses `question` and `documents`).

    Returns:
        Partial state update with `generation` and `sources`.
    """
    docs = state.get("documents", [])

    # Build context with source labels for grounding + citation
    context_parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"[Source {i+1}: {source}" + (f", page {page}" if page else "") + "]"
        context_parts.append(f"{label}\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "Answer the question using ONLY the information in the context below. "
        "If the answer cannot be found in the context, say "
        "'I don't have enough information to answer that.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['question']}\n\n"
        "Answer:"
    )
    response = _llm.invoke(prompt)

    # Build source citations with page numbers when available
    sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        citation = f"{source} (page {page})" if page else source
        if citation not in seen:
            seen.add(citation)
            sources.append(citation)

    logger.info(f"Generated answer ({len(response.content)} chars, {len(sources)} sources)")
    return {"generation": response.content, "sources": sources}


def direct_answer(state: RAGState) -> dict:
    """Answer a question directly without retrieval.

    Used for general knowledge questions that don't need the knowledge base.

    Args:
        state: Current RAG pipeline state (uses `question`).

    Returns:
        Partial state update with `generation` and empty `sources`.
    """
    prompt = f"Answer this question concisely: {state['question']}\n\nAnswer:"
    response = _llm.invoke(prompt)
    logger.info(f"Direct answer ({len(response.content)} chars)")
    return {"generation": response.content, "sources": []}


# ═══════════════════════════════════════════════════════════════
# Query Rewrite (loop)
# ═══════════════════════════════════════════════════════════════


def rewrite_query(state: RAGState) -> dict:
    """Rewrite the search query for better retrieval on the next iteration.

    Includes a loop guard: if max_rewrite_count is exceeded,
    stops rewriting and proceeds with best-effort generation.

    Args:
        state: Current RAG pipeline state.

    Returns:
        Partial state update with new `question` and incremented `query_rewrite_count`.
    """
    settings = get_rag_settings()
    count = state.get("query_rewrite_count", 0) + 1

    # Loop guard: stop after max rewrites to prevent infinite loops
    if count > settings.max_rewrite_count:
        logger.warning(f"Max rewrites reached ({count}), proceeding with best effort")
        return {"query_rewrite_count": count}

    # Provide context from previous (poor) results to guide the rewrite
    context = "\n\n".join(d.page_content[:300] for d in state.get("documents", [])[:2])
    prompt = (
        "The previous retrieval didn't find relevant documents. "
        "Rewrite the search query to be more specific or use different terms.\n\n"
        f"Original question: {state['question']}\n\n"
        f"Previous results (not relevant):\n{context}\n\n"
        "Rewritten query (one line, search-oriented):"
    )
    new_query = _llm.invoke(prompt).content.strip()
    logger.info(f"Query rewritten: '{state['question']}' → '{new_query}'")

    return {"question": new_query, "query_rewrite_count": count}
