"""State definitions for the RAG LangGraph pipelines.

TypedDicts define the shared state that flows between graph nodes.
Using `total=False` allows nodes to return partial updates.
"""

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages

from rag.enums import GradeDecision, HallucinationResult, QualityResult, RouteDecision


class RAGState(TypedDict, total=False):
    """Core state shared across all RAG pipeline nodes.

    Fields:
        question: The user's question (may be rewritten during loop)
        documents: Retrieved documents from vector store
        generation: The LLM-generated answer
        route_decision: Whether to retrieve or answer directly
        grade_decision: Whether docs are relevant enough to generate
        hallucination_result: Whether the answer is grounded in context
        quality_result: Whether the answer is useful and complete
        query_rewrite_count: Loop guard counter for query rewrites
        sources: File paths of documents used in the answer
    """

    question: str
    documents: list[Document]
    generation: str
    # Control signals (written by grading/routing nodes)
    route_decision: RouteDecision
    grade_decision: GradeDecision
    hallucination_result: HallucinationResult
    quality_result: QualityResult
    # Loop guard
    query_rewrite_count: int
    # Metadata
    sources: list[str]


class ChatRAGState(RAGState):
    """Extended state for conversational (multi-turn) RAG.

    Adds message history with automatic message merging
    via the `add_messages` reducer.
    """

    messages: Annotated[list, add_messages]


class RetrievalResult(TypedDict):
    """Result from a single retrieval source (for multi-source pipelines)."""

    source: str
    documents: list[Document]
    scores: list[float]


class EvaluationResult(TypedDict):
    """Scores from RAGAS evaluation metrics."""

    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    overall_score: float
