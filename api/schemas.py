"""Pydantic schemas for API request/response bodies."""

from datetime import datetime

from pydantic import BaseModel, Field


# ─── Request ──────────────────────────────────────────────────────────────────


class ReindexRequest(BaseModel):
    collection: str | None = None


# ─── Documents Response ───────────────────────────────────────────────────────


class DocumentUploadResponse(BaseModel):
    status: str
    file: str
    storage_key: str
    chunks_indexed: int
    collection: str
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    bucket: str
    count: int
    documents: list[dict]


class DocumentDeleteResponse(BaseModel):
    status: str
    deleted: str
    note: str


# ─── Index Response ───────────────────────────────────────────────────────────


class IndexStatusResponse(BaseModel):
    collection: str
    chunk_count: int
    files_indexed: list[str]


class ReindexResponse(BaseModel):
    status: str
    collection: str
    chunks_indexed: int
    source: str


# ─── Memory API ───────────────────────────────────────────────────────────────


class MemoryItem(BaseModel):
    namespace: list[str]
    key: str
    value: dict
    updated_at: str | None = None


class MemoryQueryResponse(BaseModel):
    project: str
    user: str
    category: str | None = None
    query: str | None = None
    count: int
    items: list[MemoryItem] = Field(default_factory=list)


class NamespaceListResponse(BaseModel):
    prefix: list[str] | str
    count: int
    namespaces: list[list[str]] = Field(default_factory=list)


class MemoryStatsResponse(BaseModel):
    total_items: int
    total_namespaces: int
    by_project: dict[str, int] = Field(default_factory=dict)


class MemoryDeleteResponse(BaseModel):
    status: str
    deleted: dict


# ─── Slack File Handling ─────────────────────────────────────────────────────


class SlackFileInfo(BaseModel):
    """Metadata from a Slack file attachment event."""

    id: str
    name: str
    filetype: str
    size: int
    mimetype: str
    url_private_download: str


class FileProcessingResult(BaseModel):
    """Result of processing a Slack-uploaded file into RAG."""

    filename: str
    file_type: str
    chars_extracted: int
    chunks_indexed: int
    collection: str
