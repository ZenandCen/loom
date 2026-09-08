"""RAG pipeline assembly — LangGraph StateGraph definitions.

Provides 4 pipeline levels of increasing complexity:
    1. BASIC:         retrieve → generate
    2. ADAPTIVE:      route → {direct | retrieve → grade → {generate | rewrite}}
    3. SELF_RAG:      adaptive + hallucination check + quality check
    4. MULTI_SOURCE:  multi_retrieve → generate

Each pipeline is a compiled LangGraph that can be invoked with
`{"question": "...", "query_rewrite_count": 0}`.
"""

import logging
from typing import Optional

from langgraph.graph import END, START, StateGraph

from rag.config import get_rag_settings
from rag.enums import (
    GradeDecision,
    PipelineLevel,
    QualityCheckDecision,
    RouteDecision,
)
from rag.nodes import (
    check_answer_quality,
    check_hallucination,
    direct_answer,
    generate,
    grade_documents,
    grade_router,
    hallucination_router,
    multi_source_retrieve,
    quality_router,
    retrieve,
    rewrite_query,
    rewrite_router,
    route_question,
)
from rag.state import RAGState

logger = logging.getLogger(__name__)


def basic_pipeline():
    """Level 1: Simplest RAG — always retrieve, always generate.

    Graph:
        START → retrieve → generate → END

    No intelligence about whether retrieval is needed.
    Best for: prototyping, testing retrieval quality in isolation.
    """
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def adaptive_pipeline():
    """Level 2: Adaptive RAG with smart routing and rewrite loop.

    Graph:
        START → route_question
                  ├── direct_answer → END
                  └── retrieve → grade_documents
                                    ├── generate → END
                                    └── rewrite_query → retrieve (loop, max N)

    Intelligence:
        - Routes simple questions to direct answer (saves retrieval cost)
        - Grades document relevance before generating
        - Rewrites query if documents are not relevant (with loop guard)

    Best for: production RAG handling mixed question types.
    """
    graph = StateGraph(RAGState)

    # Define nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("direct_answer", direct_answer)

    # Entry: route the question
    graph.add_conditional_edges(
        START,
        route_question,
        {
            RouteDecision.RETRIEVE: "retrieve",
            RouteDecision.DIRECT_ANSWER: "direct_answer",
        },
    )

    # Retrieval → Grade
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        grade_router,
        {
            GradeDecision.GENERATE: "generate",
            GradeDecision.REWRITE: "rewrite_query",
        },
    )
    # Rewrite: loop back or give up
    graph.add_conditional_edges(
        "rewrite_query",
        rewrite_router,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )

    # Terminal edges
    graph.add_edge("generate", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


def self_rag_pipeline():
    """Level 3: Self-RAG with full self-correction.

    Graph:
        START → route_question
                  ├── direct_answer → END
                  └── retrieve → grade_documents
                                    ├── generate → check_hallucination
                                    │                  ├── check_answer_quality
                                    │                  │     ├── finish → END
                                    │                  │     └── rewrite → retrieve
                                    │                  └── rewrite → retrieve
                                    └── rewrite → retrieve

    Intelligence (beyond Level 2):
        - Hallucination check: is the answer grounded in context?
        - Quality check: is the answer useful and complete?
        - Either failure triggers a rewrite + re-retrieval loop

    Best for: high-stakes applications (medical, legal, financial)
    where hallucination is unacceptable.
    """
    graph = StateGraph(RAGState)

    # Define nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("check_answer_quality", check_answer_quality)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("direct_answer", direct_answer)

    # Entry routing
    graph.add_conditional_edges(
        START,
        route_question,
        {
            RouteDecision.RETRIEVE: "retrieve",
            RouteDecision.DIRECT_ANSWER: "direct_answer",
        },
    )

    # Retrieval → Grade
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        grade_router,
        {
            GradeDecision.GENERATE: "generate",
            GradeDecision.REWRITE: "rewrite_query",
        },
    )
    # Rewrite: loop back or give up
    graph.add_conditional_edges(
        "rewrite_query",
        rewrite_router,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )

    # Generation → Verification chain
    graph.add_edge("generate", "check_hallucination")
    graph.add_conditional_edges(
        "check_hallucination",
        hallucination_router,
        {
            "check_quality": "check_answer_quality",
            GradeDecision.REWRITE: "rewrite_query",
        },
    )
    graph.add_conditional_edges(
        "check_answer_quality",
        quality_router,
        {
            QualityCheckDecision.FINISH: END,
            QualityCheckDecision.REWRITE: "rewrite_query",
        },
    )

    # Direct answer → done
    graph.add_edge("direct_answer", END)

    return graph.compile()


def multi_source_pipeline():
    """Level 4: Multi-source RAG — retrieve from multiple sources.

    Graph:
        START → multi_retrieve → generate → END

    Retrieves from vector store + (optionally) web search,
    deduplicates, merges, then generates.

    Best for: when a single knowledge base is insufficient.
    """
    graph = StateGraph(RAGState)
    graph.add_node("multi_retrieve", multi_source_retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "multi_retrieve")
    graph.add_edge("multi_retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════
# Factory & Runner
# ═══════════════════════════════════════════════════════════════

# Registry mapping pipeline levels to builder functions
_PIPELINES: dict[PipelineLevel, callable] = {
    PipelineLevel.BASIC: basic_pipeline,
    PipelineLevel.ADAPTIVE: adaptive_pipeline,
    PipelineLevel.SELF_RAG: self_rag_pipeline,
    PipelineLevel.MULTI_SOURCE: multi_source_pipeline,
}


def get_pipeline(level: PipelineLevel = PipelineLevel.ADAPTIVE):
    """Factory: get a compiled pipeline by level.

    Args:
        level: Which pipeline complexity to use.

    Returns:
        Compiled LangGraph app ready for .invoke().

    Raises:
        ValueError: If level is not a valid PipelineLevel.
    """
    if level not in _PIPELINES:
        raise ValueError(f"Unknown pipeline level: {level}. Choose from {list(_PIPELINES.keys())}")

    logger.info(f"Building '{level.value}' pipeline")
    return _PIPELINES[level]()


def run_rag(
    question: str,
    level: Optional[PipelineLevel] = None,
    collection_name: Optional[str] = None,
) -> dict:
    """Convenience runner: execute a RAG query end-to-end.

    This is the main entry point for using RAG from the loom agent.
    Wraps pipeline creation + invocation in a single call.

    Args:
        question: The user's question.
        level: Pipeline level (default: from settings).
        collection_name: Vector store collection (default: from settings).

    Returns:
        Dictionary with:
            - "generation": The answer text
            - "sources": List of source document paths
            - "documents_count": Number of documents retrieved
    """
    settings = get_rag_settings()
    level = level or settings.default_pipeline

    pipeline = get_pipeline(level)
    result = pipeline.invoke({"question": question, "query_rewrite_count": 0})

    return {
        "generation": result.get("generation", ""),
        "sources": result.get("sources", []),
        "documents_count": len(result.get("documents", [])),
    }
