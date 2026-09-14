"""Pydantic schemas for API request/response bodies."""

from datetime import datetime

from pydantic import BaseModel


# ─── Request ──────────────────────────────────────────────────────────────────


class ReindexRequest(BaseModel):
    collection: str | None = None


# ─── Response ─────────────────────────────────────────────────────────────────


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


class IndexStatusResponse(BaseModel):
    collection: str
    chunk_count: int
    files_indexed: list[str]


class ReindexResponse(BaseModel):
    status: str
    collection: str
    chunks_indexed: int
    source: str
