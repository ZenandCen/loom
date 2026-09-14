"""Loom RAG MCP Server.

Exposes RAG search + document management as MCP tools for AI agents.

Run (stdio):
    python -m mcp_server.server

Test with MCP inspector:
    npx @modelcontextprotocol/inspector python -m mcp_server.server
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from rag.config import get_rag_settings
from rag.retrieval import get_retriever
from rag.indexing import get_vectorstore, index_documents
from storage.minio import MinIOStorage

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App instance ─────────────────────────────────────────────────────────────
mcp = MCPServer(
    name="loom-rag",
    description="Loom RAG system: semantic search, document management, and indexing.",
)

# ─── Lazy singletons ──────────────────────────────────────────────────────────
_storage: MinIOStorage | None = None
_data_dir: Path = Path(os.getenv("RAG_DATA_DIR", "./data"))


def _get_storage() -> MinIOStorage:
    global _storage
    if _storage is None:
        _storage = MinIOStorage()
    return _storage


# ─── Tools ────────────────────────────────────────────────────────────────────


@mcp.tool(description="Search the RAG knowledge base using semantic similarity. Returns top-k most similar document chunks with source metadata.")
def search_rag(query: str, k: int = 4, collection: str | None = None) -> str:
    """Search indexed documents by semantic similarity."""
    k = min(k, 20)
    retriever = get_retriever(collection_name=collection, k=k)
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("type", "unknown")
        results.append(f"[{i}] ({doc_type}) {source}\n{doc.page_content[:500]}")

    return f"Found {len(results)} relevant chunks:\n\n" + "\n\n---\n\n".join(results)


@mcp.tool(description="List all documents stored in the RAG system with file names, sizes, and upload timestamps.")
def list_documents() -> str:
    """List all stored documents."""
    storage = _get_storage()
    objects = storage.list_objects()

    if not objects:
        return "No documents stored."

    lines = [f"Total: {len(objects)} documents\n"]
    for obj in objects:
        lines.append(f"  - {obj.key} ({obj.size:,} bytes, modified: {obj.last_modified[:10]})")

    return "\n".join(lines)


@mcp.tool(description="Upload and index a local file into the RAG knowledge base. File is saved to MinIO and indexed for retrieval.")
def upload_document(file_path: str, collection: str | None = None) -> str:
    """Upload a file from disk and index it."""
    from rag.chunking import ChunkingConfig, chunk_documents
    from rag.indexing import SUPPORTED_EXTENSIONS, load_document

    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found: {file_path}"

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return f"Error: Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"

    content = path.read_bytes()
    storage = _get_storage()
    storage.put(path.name, content)

    docs = load_document(path)
    if not docs:
        return f"Error: Could not extract content from '{path.name}'"

    chunks = chunk_documents(docs, ChunkingConfig())
    vs = get_vectorstore(collection)
    vs.add_documents(chunks)

    return f"Indexed '{path.name}' → {len(chunks)} chunks into collection."


@mcp.tool(description="Remove a document from RAG storage by its key (filename). Run reindex after to update vector DB.")
def delete_document(key: str) -> str:
    """Delete a document from storage."""
    storage = _get_storage()
    if not storage.exists(key):
        return f"Error: Document '{key}' not found in storage."

    storage.delete(key)
    return f"Deleted '{key}' from storage. Run reindex to update vector DB."


@mcp.tool(description="Get the current indexing status: chunk count and indexed files for a collection.")
def get_index_status(collection: str | None = None) -> str:
    """Get indexing status."""
    settings = get_rag_settings()
    collection = collection or settings.default_collection

    vs = get_vectorstore(collection)
    try:
        docs = vs.similarity_search("test", k=1000)
        chunk_count = len(docs)
        sources = sorted({d.metadata.get("source", "unknown") for d in docs})
    except Exception:
        return f"Collection '{collection}' is empty or inaccessible."

    lines = [f"Collection: {collection}", f"Chunks: {chunk_count}", f"Files: {len(sources)}"]
    for s in sources:
        lines.append(f"  - {s}")

    return "\n".join(lines)


@mcp.tool(description="Reindex all documents from MinIO storage into the vector database. Use after bulk deletions.")
def reindex(collection: str | None = None) -> str:
    """Rebuild the vector index from stored documents."""
    settings = get_rag_settings()
    collection = collection or settings.default_collection

    storage = _get_storage()
    objects = storage.list_objects()
    for obj in objects:
        local_path = _data_dir / obj.key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(storage.get(obj.key))

    count = index_documents(_data_dir, collection)
    return f"Reindexed {count} chunks into '{collection}' from {len(objects)} files."


# ─── Resources ────────────────────────────────────────────────────────────────


@mcp.resource("loom://config", description="RAG system configuration")
def get_config() -> str:
    """RAG system configuration (embedding model, vector DB, chunking params)."""
    settings = get_rag_settings()
    return json.dumps({
        "embedding_provider": settings.embedding_provider.value,
        "embedding_model": settings.embedding_model,
        "vector_db": settings.vector_db_type.value,
        "default_collection": settings.default_collection,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "retrieval_k": settings.retrieval_k,
        "pipeline": settings.default_pipeline.value,
    }, indent=2)


# ─── Entry point ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    mcp.run()
