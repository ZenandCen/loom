"""Document upload/list/delete routes.

Handles: POST /api/documents, GET /api/documents, DELETE /api/documents/{key}
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentUploadResponse,
)
from rag.chunking import ChunkingConfig, chunk_documents
from rag.config import get_rag_settings
from rag.indexing import SUPPORTED_EXTENSIONS, get_vectorstore, load_document
from storage.base import BaseStorage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

# Injected at startup (see api/app.py)
_storage: BaseStorage | None = None
_data_dir: Path = Path("./data")


def init(storage: BaseStorage, data_dir: Path | None = None):
    global _storage, _data_dir
    _storage = storage
    if data_dir:
        _data_dir = data_dir


def _get_storage() -> BaseStorage:
    if _storage is None:
        raise HTTPException(500, "Storage not initialized")
    return _storage


@router.post("", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection: str | None = None,
):
    """Upload file → save to storage → index → return chunk count."""
    settings = get_rag_settings()
    collection = collection or settings.default_collection

    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported type '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    # Save to storage
    storage = _get_storage()
    storage_key = filename
    storage.put(storage_key, content, file.content_type or "application/octet-stream")

    # Index: load → chunk → embed → vector store
    try:
        tmp_path = _data_dir / f".tmp_{filename}"
        tmp_path.write_bytes(content)
        docs = load_document(tmp_path)
        tmp_path.unlink(missing_ok=True)

        if not docs:
            raise HTTPException(400, f"Cannot extract content from '{filename}'")

        chunks = chunk_documents(docs, ChunkingConfig())
        vs = get_vectorstore(collection)
        vs.add_documents(chunks)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(500, f"Indexing failed: {e}")

    return DocumentUploadResponse(
        status="ok",
        file=filename,
        storage_key=storage_key,
        chunks_indexed=len(chunks),
        collection=collection,
        uploaded_at=datetime.now(timezone.utc),
    )


@router.get("", response_model=DocumentListResponse)
def list_documents():
    """List all documents in storage."""
    storage = _get_storage()
    objects = storage.list_objects()
    return DocumentListResponse(
        bucket=type(storage).__name__,
        count=len(objects),
        documents=[{"key": o.key, "size": o.size, "last_modified": o.last_modified} for o in objects],
    )


@router.delete("/{key}", response_model=DocumentDeleteResponse)
def delete_document(key: str):
    """Remove document from storage."""
    storage = _get_storage()
    if not storage.exists(key):
        raise HTTPException(404, f"Document '{key}' not found")

    storage.delete(key)
    return DocumentDeleteResponse(
        status="ok",
        deleted=key,
        note="Chunks still in vector store. Run POST /api/index/reindex to rebuild.",
    )
