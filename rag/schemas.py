"""Pydantic schemas for RAG domain — request/response models for RAG operations."""

from pydantic import BaseModel, Field


class RAGQueryResult(BaseModel):
    """Result from run_rag() pipeline execution."""

    generation: str = Field(description="LLM-generated answer")
    sources: list[str] = Field(default_factory=list, description="Source document paths")
    documents_count: int = Field(default=0, description="Number of documents retrieved")


class RetrievalHit(BaseModel):
    """A single retrieval hit with optional parent context."""

    content: str = Field(description="Child chunk content")
    source: str = Field(default="", description="Source file path")
    file_type: str = Field(default="", description="File type (md, code, pdf, etc.)")
    parent_content: str | None = Field(default=None, description="Full parent context (if available)")
    score: float | None = Field(default=None, description="Similarity score (if available)")


class ParentRetrievalResult(BaseModel):
    """Result from retrieve_with_parents() — children expanded to parents."""

    hits: list[RetrievalHit] = Field(default_factory=list, description="Individual retrieval hits")
    context_blocks: list[str] = Field(default_factory=list, description="Formatted context blocks for LLM")
    sources: list[str] = Field(default_factory=list, description="Unique source file names")
