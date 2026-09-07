"""RAG module — Retrieval-Augmented Generation pipeline for loom.

Provides document indexing, retrieval, and answer generation
integrated with the main loom agent.
"""

from rag.enums import ChunkingStrategy, PipelineLevel, VectorDBType
from rag.config import RAGSettings, get_rag_settings
from rag.pipeline import get_pipeline, run_rag

__version__ = "0.1.0"

__all__ = [
    "ChunkingStrategy",
    "PipelineLevel",
    "VectorDBType",
    "RAGSettings",
    "get_rag_settings",
    "get_pipeline",
    "run_rag",
]
