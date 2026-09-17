"""Retrieval strategies: vector search, hybrid search, reranking.

Provides multiple retrieval approaches:
    - Vector similarity search (semantic)
    - Hybrid search (vector + BM25 via Reciprocal Rank Fusion)
    - Reranking (cross-encoder or keyword overlap fallback)
    - Fallback retrieval (CRAG pattern)
    - Parent expansion (retrieve children → expand to full parent context)
"""

import logging
from typing import Optional

from langchain_core.documents import Document

from rag.config import get_rag_settings

logger = logging.getLogger(__name__)


def retrieve_with_parents(
    query: str,
    collection_name: Optional[str] = None,
    k: Optional[int] = None,
    parent_max_chars: int = 4000,
) -> "ParentRetrievalResult":
    """Retrieve with parent expansion.

    Searches for relevant children (small, precise chunks), then expands
    each hit to its full parent (large context) from the rag_parents table.

    Args:
        query: Search query.
        collection_name: Vector store collection.
        k: Number of child chunks to retrieve.
        parent_max_chars: Max characters per parent in the output (safety limit).

    Returns:
        ParentRetrievalResult with hits, context_blocks, and sources.
    """
    from pathlib import Path

    from rag.indexing import get_vectorstore
    from rag.parents import fetch_parents
    from rag.schemas import ParentRetrievalResult, RetrievalHit

    settings = get_rag_settings()
    k = k or settings.retrieval_k
    collection_name = collection_name or settings.default_collection

    vs = get_vectorstore(collection_name)
    results = vs.similarity_search(query, k=k)
    if not results:
        return ParentRetrievalResult()

    # Collect unique parent_ids
    parent_ids = list({r.metadata.get("parent_id") for r in results if r.metadata.get("parent_id")})

    # Fetch parents
    parent_map: dict[str, str] = {}
    if parent_ids:
        parents = fetch_parents(parent_ids, collection=collection_name)
        for p in parents:
            parent_map[p.id] = p.content[:parent_max_chars]

    # Build hits and context blocks
    hits: list[RetrievalHit] = []
    context_blocks: list[str] = []
    sources: list[str] = []

    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("source", "?")
        fname = Path(src).name
        ftype = doc.metadata.get("type", "?")
        extra = ""
        if doc.metadata.get("class"):
            fn = doc.metadata.get("function", "")
            extra = f" [{doc.metadata['class']}.{fn}]" if fn else f" [{doc.metadata['class']}]"

        pid = doc.metadata.get("parent_id", "")
        parent_content = parent_map.get(pid, "")

        block = f"[{i}] ({ftype}) {fname}{extra}"
        if parent_content:
            block += f"\n\n--- Full context ---\n{parent_content}"
        block += f"\n\n--- Relevant section ---\n{doc.page_content}"
        context_blocks.append(block)

        hits.append(RetrievalHit(
            content=doc.page_content,
            source=src,
            file_type=ftype,
            parent_content=parent_content if parent_content else None,
        ))

        if fname not in sources:
            sources.append(fname)

    logger.info(
        f"retrieve_with_parents: {len(results)} children → "
        f"{len(parent_ids)} parents expanded"
    )
    return ParentRetrievalResult(
        hits=hits,
        context_blocks=context_blocks,
        sources=sources,
    )


def discover_projects(
    query: str,
    collections: list[str],
    k: int = 1,
) -> list[tuple[str, float]]:
    """Probe each collection with the query and rank by relevance.

    Free-mode (rag_kb "homepage") discovery: instead of "list all collections",
    we let VECTOR similarity decide which projects are actually relevant to the
    user's input. E.g. query "các project DWH" surfaces the fpt_dwh_* collections
    near the top rather than everything.

    Args:
        query: The user's question (embedded and compared against each collection).
        collections: Collection (project) names to probe.
        k: Number of hits to sample per collection (only the top hit scores it).

    Returns:
        List of (collection_name, relevance) sorted by relevance (desc). Relevance
        is a 0..1-ish value derived from the top hit's vector distance; it is used
        only for ranking/display in the human-in-the-loop prompt, never as a hard gate.
    """
    from rag.indexing import get_vectorstore

    scored: list[tuple[str, float]] = []
    for col in collections:
        try:
            vs = get_vectorstore(col)
            hits = vs.similarity_search_with_score(query, k=k)
            if hits:
                _doc, dist = hits[0]
                # pgvector returns a distance (lower = more similar). Normalise to a
                # 0..1-ish relevance for ranking. Clamp negatives to 0.
                rel = max(0.0, 1.0 - float(dist))
                scored.append((col, rel))
            else:
                scored.append((col, 0.0))
        except Exception as e:
            logger.warning(f"discover_projects: probe failed for '{col}': {e}")
            scored.append((col, 0.0))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def retrieve_across_collections(
    query: str,
    collections: list[str],
    k_per_collection: int = 4,
    max_total: int = 20,
    parent_max_chars: int = 4000,
) -> "ParentRetrievalResult":
    """Search multiple collections, expand parents, and group context by project.

    Free-mode (rag_kb "homepage") retrieval: pulls the top-k_per_collection hits from
    EACH requested collection (so every confirmed project is represented), expands each
    hit to its full parent, and groups the context under a per-project header so the LLM
    can attribute evidence to its project and analyse cross-project relationships.

    Collections should be passed in relevance order (from discover_projects) so the most
    relevant projects get full representation when max_total is reached.

    Args:
        query: The user's question.
        collections: The (confirmed) collection names to search.
        k_per_collection: Child chunks to pull per collection.
        max_total: Cap on total chunk blocks across all collections.
        parent_max_chars: Safety limit per expanded parent.

    Returns:
        ParentRetrievalResult with hits, per-project context_blocks, and sources.
    """
    from pathlib import Path

    from rag.indexing import get_vectorstore
    from rag.parents import fetch_parents
    from rag.schemas import ParentRetrievalResult, RetrievalHit

    collections = list(dict.fromkeys(collections))  # dedupe, keep order
    hits: list[RetrievalHit] = []
    context_blocks: list[str] = []
    sources: list[str] = []
    idx = 0

    for col in collections:
        if idx >= max_total:
            break
        try:
            vs = get_vectorstore(col)
            results = vs.similarity_search(query, k=k_per_collection)
        except Exception as e:
            logger.warning(f"retrieve_across_collections: search failed for '{col}': {e}")
            continue
        if not results:
            continue

        # Expand this collection's hits to their parents (per-collection fetch is safe
        # and does not rely on parent-id uniqueness across collections).
        parent_ids = [d.metadata.get("parent_id") for d in results if d.metadata.get("parent_id")]
        parent_map: dict[str, str] = {}
        if parent_ids:
            for p in fetch_parents(parent_ids, collection=col):
                parent_map[p.id] = p.content[:parent_max_chars]

        col_blocks: list[str] = []
        for d in results:
            if idx >= max_total:
                break
            idx += 1
            src = d.metadata.get("source", "?")
            fname = Path(src).name
            ftype = d.metadata.get("type", "?")
            extra = ""
            if d.metadata.get("class"):
                fn = d.metadata.get("function", "")
                extra = f" [{d.metadata['class']}.{fn}]" if fn else f" [{d.metadata['class']}]"
            parent_content = parent_map.get(d.metadata.get("parent_id", ""), "")
            block = f"[{idx}] ({ftype}) {fname}{extra}"
            if parent_content:
                block += f"\n\n--- Full context ---\n{parent_content}"
            block += f"\n\n--- Relevant section ---\n{d.page_content}"
            col_blocks.append(block)
            hits.append(RetrievalHit(
                content=d.page_content,
                source=src,
                file_type=ftype,
                parent_content=parent_content if parent_content else None,
            ))
            if fname not in sources:
                sources.append(fname)

        if col_blocks:
            context_blocks.append(f"## Project: {col}")
            context_blocks.extend(col_blocks)

    logger.info(
        f"retrieve_across_collections: {len(collections)} collections → "
        f"{len(hits)} hits from {len(sources)} files"
    )
    return ParentRetrievalResult(hits=hits, context_blocks=context_blocks, sources=sources)


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
