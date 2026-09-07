"""RAG configuration — settings loaded from environment variables.

All tunable parameters for the RAG pipeline are centralized here.
Uses enums for type-safe string config values.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from rag.enums import ChunkingStrategy, PipelineLevel, VectorDBType


@dataclass
class RAGSettings:
    """Centralized RAG configuration.

    All values can be overridden via environment variables.
    Priority: env var > default value.
    """

    # --- Embedding ---
    # Model used to generate vector embeddings for documents and queries
    embedding_model: str = field(
        default_factory=lambda: os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # --- Vector DB ---
    # Persistence directory for Chroma (default backend)
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("RAG_CHROMA_DIR", "./chroma_data")
    )
    # Qdrant connection (set to enable Qdrant backend)
    qdrant_url: str = field(default_factory=lambda: os.getenv("RAG_QDRANT_URL", ""))
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("RAG_QDRANT_API_KEY", ""))
    # pgvector connection (set to enable pgvector backend)
    postgres_dsn: str = field(default_factory=lambda: os.getenv("RAG_PG_DSN", ""))

    # --- Chunking ---
    # Target token count per chunk
    chunk_size: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_SIZE", "512")))
    # Overlapping tokens between consecutive chunks (preserves context)
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "50")))
    # Default chunking strategy
    chunking_strategy: ChunkingStrategy = field(
        default_factory=lambda: ChunkingStrategy(
            os.getenv("RAG_CHUNKING_STRATEGY", "recursive")
        )
    )
    # Parent-child specific: parent chunk size (for context)
    parent_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("RAG_PARENT_CHUNK_SIZE", "2000"))
    )
    # Parent-child specific: child chunk size (for retrieval)
    child_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("RAG_CHILD_CHUNK_SIZE", "400"))
    )

    # --- Retrieval ---
    # Number of documents to retrieve per query
    retrieval_k: int = field(default_factory=lambda: int(os.getenv("RAG_RETRIEVAL_K", "4")))
    # Maximum query rewrites before giving up (loop guard)
    max_rewrite_count: int = field(
        default_factory=lambda: int(os.getenv("RAG_MAX_REWRITE_COUNT", "2"))
    )
    # Default pipeline level
    default_pipeline: PipelineLevel = field(
        default_factory=lambda: PipelineLevel(os.getenv("RAG_PIPELINE", "adaptive"))
    )

    # --- Paths ---
    # Directory containing source documents to index
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_DATA_DIR", "./data")))
    # Path to evaluation dataset JSON
    eval_dataset_path: str = field(
        default_factory=lambda: os.getenv("RAG_EVAL_DATASET", "./data/eval_questions.json")
    )

    # --- Collection ---
    # Default vector store collection name
    default_collection: str = field(
        default_factory=lambda: os.getenv("RAG_COLLECTION", "rag_kb")
    )

    @property
    def vector_db_type(self) -> VectorDBType:
        """Determine active vector DB backend based on configured connection strings."""
        if self.qdrant_url:
            return VectorDBType.QDRANT
        if self.postgres_dsn:
            return VectorDBType.PGVECTOR
        return VectorDBType.CHROMA


# Singleton instance — import this everywhere instead of creating new Settings
_settings: RAGSettings | None = None


def get_rag_settings() -> RAGSettings:
    """Return the shared RAGSettings instance (lazy singleton)."""
    global _settings
    if _settings is None:
        _settings = RAGSettings()
    return _settings
