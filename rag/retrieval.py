"""Retrieval strategies: vector search, hybrid search, reranking.

Provides multiple retrieval approaches:
    - Vector similarity search (semantic)
    - Hybrid search (vector + BM25 via Reciprocal Rank Fusion)
    - Reranking (cross-encoder or keyword overlap fallback)
    - Fallback retrieval (CRAG pattern)
"""

import logging
from typing import Optional

from langchain_core.documents import Document

from rag.config import get_rag_settings

logger = logging.getLogger(__name__)


def get_retriever(collection_name: Optional[str] = None, k: Optional[int] = None):
    """Create a vector similarity retriever from the configured store.

    Args:
        collection_name: Vector store collection to search.
        k: Number of documents to retrieve per query.

    Returns:
        A LangChain Retriever that returns top-k similar documents.
    """
    from rag.indexing import get_vectorstore

    settings = get_rag_settings()
    k = k or settings.retrieval_k
    vectorstore = get_vectorstore(collection_name)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def hybrid_retrieve(
    query: str,
    vector_retriever,
    bm25_retriever=None,
    k: Optional[int] = None,
    alpha: float = 0.5,
) -> list[Document]:
    """Hybrid retrieval combining vector + BM25 via Reciprocal Rank Fusion.

    Vector search captures semantic meaning; BM25 captures exact keyword
    matches. RRF merges both result lists without needing score normalization.

    If no BM25 retriever is provided, falls back to vector-only search.

    Args:
        query: Search query string.
        vector_retriever: Vector similarity retriever (required).
        bm25_retriever: Optional BM25/keyword retriever.
        k: Number of results to return.
        alpha: Weight for vector results (1-alpha for BM25).

    Returns:
        Merged and ranked documents.
    """
    settings = get_rag_settings()
    k = k or settings.retrieval_k

    vector_docs = vector_retriever.invoke(query)

    if bm25_retriever is None:
        logger.debug("No BM25 retriever, using vector-only")
        return vector_docs

    bm25_docs = bm25_retriever.invoke(query)
    return _reciprocal_rank_fusion(vector_docs, bm25_docs, k=k, alpha=alpha)


def _reciprocal_rank_fusion(
    list1: list[Document],
    list2: list[Document],
    k: int = 4,
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> list[Document]:
    """Merge two ranked lists using Reciprocal Rank Fusion (RRF).

    Formula: score(doc) = alpha/(rrf_k + rank_in_list1) + (1-alpha)/(rrf_k + rank_in_list2)

    Documents appearing in BOTH lists get the highest scores.
    The rrf_k=60 constant smooths rank differences (from the original RRF paper).

    Args:
        list1: First ranked list (e.g., vector results).
        list2: Second ranked list (e.g., BM25 results).
        k: Number of top results to return.
        alpha: Weight for list1 (1-alpha for list2).
        rrf_k: Smoothing constant (standard value: 60).

    Returns:
        Merged documents sorted by RRF score (descending).
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    # Score documents from list1 (vector)
    for rank, doc in enumerate(list1, start=1):
        key = doc.page_content[:100]  # Dedup key: first 100 chars
        scores[key] = scores.get(key, 0) + alpha / (rrf_k + rank)
        doc_map[key] = doc

    # Score documents from list2 (BM25)
    for rank, doc in enumerate(list2, start=1):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + (1 - alpha) / (rrf_k + rank)
        if key not in doc_map:
            doc_map[key] = doc

    # Return top-k by score
    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in ranked_keys[:k]]


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: Optional[int] = None,
) -> list[Document]:
    """Rerank retrieved documents for better precision.

    Tries Cohere cross-encoder first (most accurate), falls back to
    keyword overlap scoring if Cohere is unavailable.

    Args:
        query: Original search query.
        documents: Retrieved documents to rerank.
        top_k: Number of top documents to keep after reranking.

    Returns:
        Reranked documents (most relevant first).
    """
    settings = get_rag_settings()
    top_k = top_k or settings.retrieval_k

    if not documents:
        return documents

    # Try Cohere cross-encoder reranker
    try:
        import cohere

        client = cohere.ClientV2()
        results = client.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=[d.page_content for d in documents],
            top_n=top_k,
        )
        reranked = [documents[r.index] for r in results.results]
        logger.debug(f"Cohere rerank: {len(reranked)} docs")
        return reranked
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Cohere rerank failed, using fallback: {e}")

    # Fallback: keyword overlap scoring
    return _simple_rerank(query, documents, top_k)


def _simple_rerank(query: str, documents: list[Document], top_k: int) -> list[Document]:
    """Keyword overlap reranking as a dependency-free fallback.

    Scores each document by the fraction of query words that appear in it.
    Less accurate than a cross-encoder but requires no API calls.

    Args:
        query: Search query.
        documents: Documents to score.
        top_k: Number of top documents to keep.

    Returns:
        Documents sorted by keyword overlap score (descending).
    """
    query_words = set(query.lower().split())

    def score(doc: Document) -> float:
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_words & doc_words)
        return overlap / max(len(query_words), 1)

    scored = sorted(documents, key=score, reverse=True)
    return scored[:top_k]


def retrieve_with_fallback(
    query: str,
    primary_retriever,
    fallback_retriever=None,
    k: Optional[int] = None,
) -> list[Document]:
    """Retrieve with fallback: try primary source, fall back if empty.

    Implements the CRAG (Corrective RAG) pattern:
    if vector search returns nothing, try an alternative source
    (e.g., web search).

    Args:
        query: Search query.
        primary_retriever: First-choice retriever (e.g., vector store).
        fallback_retriever: Backup retriever (e.g., web search).
        k: Maximum number of results.

    Returns:
        Retrieved documents (from primary or fallback).
    """
    settings = get_rag_settings()
    k = k or settings.retrieval_k

    docs = primary_retriever.invoke(query)

    if not docs and fallback_retriever is not None:
        logger.info("Primary retrieval empty, using fallback")
        docs = fallback_retriever.invoke(query)

    return docs[:k]
