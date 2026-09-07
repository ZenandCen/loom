"""Enums for all configurable string values in the RAG module.

Using enums instead of raw strings ensures type safety,
IDE autocomplete, and prevents typos in config.
"""

from enum import Enum


class ChunkingStrategy(str, Enum):
    """Strategy for splitting documents into chunks.

    - RECURSIVE: Split by paragraphs → sentences → words → chars (default)
    - HEADER: Split on Markdown/HTML headers, preserve structure
    - PARENT_CHILD: Small children for retrieval, large parents for context
    - CONTEXTUAL: LLM generates context prefix per chunk (best precision)
    """

    RECURSIVE = "recursive"
    HEADER = "header"
    PARENT_CHILD = "parent_child"
    CONTEXTUAL = "contextual"


class VectorDBType(str, Enum):
    """Vector database backend for storing embeddings.

    Selection priority: QDRANT > PGVECTOR > CHROMA (first configured wins).
    """

    CHROMA = "chroma"
    QDRANT = "qdrant"
    PGVECTOR = "pgvector"


class PipelineLevel(str, Enum):
    """Complexity level of the RAG pipeline.

    - BASIC: retrieve → generate (no intelligence)
    - ADAPTIVE: smart routing + document grading + query rewrite loop
    - SELF_RAG: full self-correction with hallucination + quality checks
    - MULTI_SOURCE: retrieve from multiple sources and merge
    """

    BASIC = "basic"
    ADAPTIVE = "adaptive"
    SELF_RAG = "self_rag"
    MULTI_SOURCE = "multi_source"


class RouteDecision(str, Enum):
    """Routing decision: does the question need external documents?"""

    RETRIEVE = "retrieve"
    DIRECT_ANSWER = "direct_answer"


class GradeDecision(str, Enum):
    """Document relevance grade after retrieval."""

    GENERATE = "generate"
    REWRITE = "rewrite"


class HallucinationResult(str, Enum):
    """Result of hallucination check on generated answer."""

    GROUNDED = "grounded"
    NOT_GROUNDED = "not_grounded"


class QualityResult(str, Enum):
    """Result of answer quality check."""

    USEFUL = "useful"
    NOT_USEFUL = "not_useful"


class QualityCheckDecision(str, Enum):
    """Decision after answer quality check."""

    FINISH = "finish"
    REWRITE = "rewrite"


class EmbeddingProvider(str, Enum):
    """Embedding model provider.

    - OPENAI: Cloud embeddings (text-embedding-3-small, etc.)
    - OLLAMA: Local embeddings via Ollama (nomic-embed-text, etc.)
    """

    OPENAI = "openai"
    OLLAMA = "ollama"
