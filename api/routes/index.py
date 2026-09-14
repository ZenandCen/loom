"""Index management routes.

Handles: GET /api/index/status, POST /api/index/reindex
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import IndexStatusResponse, ReindexRequest, ReindexResponse
from rag.config import get_rag_settings
from rag.indexing import get_vectorstore, index_documents

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/index", tags=["index"])

_data_dir: Path = Path("./data")
_storage = None


def init(storage, data_dir: Path | None = None):
    global _storage, _data_dir
    _storage = storage
    if data_dir:
        _data_dir = data_dir


@router.get("/status", response_model=IndexStatusResponse)
def get_status(collection: str | None = None):
    """Get indexing status for a collection."""
    settings = get_rag_settings()
    collection = collection or settings.default_collection

    vs = get_vectorstore(collection)
    try:
        docs = vs.similarity_search("test", k=1000)
        chunk_count = len(docs)
        sources = sorted({d.metadata.get("source", "unknown") for d in docs})
    except Exception:
        chunk_count = 0
        sources = []

    return IndexStatusResponse(collection=collection, chunk_count=chunk_count, files_indexed=sources)


@router.post("/reindex", response_model=ReindexResponse)
def reindex(body: ReindexRequest | None = None):
    """Re-index all documents from storage."""
    settings = get_rag_settings()
    collection = (body.collection if body else None) or settings.default_collection

    # Download all files from storage to local dir
    if _storage:
        from storage.base import BaseStorage

        objects = _storage.list_objects()
        for obj in objects:
            local_path = _data_dir / obj.key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(_storage.get(obj.key))
        logger.info(f"Downloaded {len(objects)} files from storage")

    count = index_documents(_data_dir, collection)
    return ReindexResponse(status="ok", collection=collection, chunks_indexed=count, source="storage")
