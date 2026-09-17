"""Document loading and indexing pipeline.

Handles the ingestion side of RAG:
    Load files → Chunk → Embed → Store in vector DB.

Design principle: Each logical unit (page, row, section) becomes a
separate Document to preserve fine-grained metadata and improve
retrieval precision.
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from rag.chunking import ChunkingConfig, chunk_documents, chunk_documents_with_parents
from rag.config import get_embeddings, get_rag_settings
from rag.enums import ChunkingStrategy, VectorDBType

logger = logging.getLogger(__name__)

# File extensions that the loader supports
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".md", ".txt", ".html", ".docx", ".csv", ".xlsx", ".xls"}


def load_documents(input_dir: Path) -> list[Document]:
    """Recursively load all supported documents from a directory.

    Each file may produce multiple Documents (e.g., a 50-page PDF → 50 docs).
    Skips dotfiles and unsupported formats.

    Args:
        input_dir: Directory to scan for documents.

    Returns:
        Flat list of Documents (before chunking).
    """
    documents: list[Document] = []

    if not input_dir.exists():
        logger.warning(f"Input directory does not exist: {input_dir}")
        return documents

    for file_path in sorted(input_dir.rglob("*")):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if file_path.name.startswith("."):
            continue

        try:
            docs = load_document(file_path)
            if docs:
                # Filter out empty documents (e.g., scanned PDFs, empty rows)
                docs = [d for d in docs if d.page_content.strip()]
                if docs:
                    documents.extend(docs)
                    logger.info(f"Loaded: {file_path.name} ({len(docs)} units, {sum(len(d.page_content) for d in docs)} chars)")
                else:
                    logger.warning(f"Skipped (empty content): {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def load_document(file_path: Path) -> Optional[list[Document]]:
    """Load a single file into one or more Documents.

    Each logical unit becomes a separate Document:
        - PDF: one Document per page (preserves page number metadata)
        - CSV: one Document per row (structured "Column: Value" format)
        - HTML/DOCX: one Document per section (as split by the loader)
        - MD/TXT: one Document for the whole file (no natural sub-units)

    This granularity improves retrieval precision because:
        - Smaller units → more focused embeddings → better cosine matching
        - Page/row metadata preserved → accurate citations
        - Chunking operates on already-meaningful units

    Args:
        file_path: Path to the file to load.

    Returns:
        List of Documents, or None if format is unsupported.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        docs = PyPDFLoader(str(file_path)).load()
        # PyPDFLoader preserves page number in metadata — keep it for citations
        docs = [d for d in docs if d.page_content.strip()]
        if not docs:
            logger.warning(f"PDF has no extractable text (scanned image?): {file_path.name}")
            return None
        for doc in docs:
            doc.metadata["type"] = "pdf"
            doc.metadata["source"] = str(file_path)
        return docs

    elif suffix in (".md", ".txt"):
        content = file_path.read_text(encoding="utf-8")
        return [Document(page_content=content, metadata={"source": str(file_path), "type": suffix[1:]})]

    elif suffix == ".csv":
        from langchain_community.document_loaders import CSVLoader

        # Each row → 1 Document: "Col A: val \n Col B: val"
        # Better than raw text: retrieval can match specific rows
        docs = CSVLoader(str(file_path)).load()
        for doc in docs:
            doc.metadata["type"] = "csv"
            doc.metadata["source"] = str(file_path)
        return docs
    

    elif suffix in (".xlsx", ".xls"):
        import pandas as pd

        df = pd.read_excel(file_path)
        docs = []
        for _, row in df.iterrows():
            content = "\n".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            docs.append(Document(
                page_content=content,
                metadata={"source": str(file_path), "type": suffix[1:], "sheet": "default"},
            ))
        return docs

    elif suffix == ".html":
        from langchain_community.document_loaders import BSHTMLLoader

        docs = BSHTMLLoader(str(file_path)).load()
        for doc in docs:
            doc.metadata["type"] = "html"
            doc.metadata["source"] = str(file_path)
        return docs

    elif suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        docs = Docx2txtLoader(str(file_path)).load()
        for doc in docs:
            doc.metadata["type"] = "docx"
            doc.metadata["source"] = str(file_path)
        return docs

    return None


def get_vectorstore(collection_name: Optional[str] = None):
    """Factory: return a vector store instance based on configured backend.

    Selection logic (first match wins):
        1. Qdrant — if RAG_QDRANT_URL is set
        2. pgvector — if RAG_PG_DSN is set
        3. Chroma — default (local, file-based)

    Args:
        collection_name: Name of the collection/table to use.

    Returns:
        A LangChain-compatible vector store instance.
    """
    settings = get_rag_settings()
    collection_name = collection_name or settings.default_collection
    embeddings = get_embeddings()

    if settings.vector_db_type == VectorDBType.QDRANT:
        from langchain_qdrant import QdrantVectorStore
        import qdrant_client

        client = qdrant_client.QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )

    elif settings.vector_db_type == VectorDBType.PGVECTOR:
        from langchain_postgres import PGVector

        return PGVector(
            embeddings=embeddings,
            connection=settings.postgres_dsn,
            collection_name=collection_name,
            create_extension=True,
        )

    else:  # Chroma (default, local)
        from langchain_chroma import Chroma

        return Chroma(
            persist_directory=settings.chroma_persist_dir,
            collection_name=collection_name,
            embedding_function=embeddings,
        )


def collection_exists(collection_name: str) -> bool:
    """Check if a collection already exists in the vector store."""
    settings = get_rag_settings()
    if settings.vector_db_type == VectorDBType.PGVECTOR:
        try:
            import psycopg
            conn = psycopg.connect(settings.postgres_dsn, autocommit=True)
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM langchain_pg_collection WHERE name = %s", (collection_name,))
            exists = cur.fetchone() is not None
            conn.close()
            return exists
        except Exception:
            return False
    elif settings.vector_db_type == VectorDBType.CHROMA:
        try:
            from langchain_chroma import Chroma
            embeddings = get_embeddings()
            vs = Chroma(
                persist_directory=settings.chroma_persist_dir,
                collection_name=collection_name,
                embedding_function=embeddings,
            )
            return vs._collection.count() > 0
        except Exception:
            return False
    return False


def get_collection_count(collection_name: str) -> int:
    """Get number of embeddings in a collection."""
    settings = get_rag_settings()
    if settings.vector_db_type == VectorDBType.PGVECTOR:
        try:
            import psycopg
            conn = psycopg.connect(settings.postgres_dsn, autocommit=True)
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(*) FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
            """, (collection_name,))
            count = cur.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0
    return 0


def list_collections(min_count: int = 1) -> list[tuple[str, int]]:
    """List collections that contain data, as (name, chunk_count) sorted by count desc.

    Used in free mode (rag_kb "homepage") to discover which projects have been
    learned, and to drive query-driven cross-project discovery.

    Args:
        min_count: Only return collections with at least this many chunks.

    Returns:
        List of (collection_name, chunk_count) tuples, most-chunked first.
        Empty list if no data or the backend is unsupported.
    """
    settings = get_rag_settings()
    if settings.vector_db_type == VectorDBType.PGVECTOR:
        try:
            import psycopg
            conn = psycopg.connect(settings.postgres_dsn, autocommit=True)
            cur = conn.cursor()
            cur.execute("""
                SELECT c.name, COUNT(e.id) AS cnt
                FROM langchain_pg_collection c
                LEFT JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
                GROUP BY c.name
                HAVING COUNT(e.id) >= %s
                ORDER BY cnt DESC
            """, (min_count,))
            rows = cur.fetchall()
            conn.close()
            return [(r[0], int(r[1])) for r in rows]
        except Exception as e:
            logger.warning(f"list_collections (pgvector) failed: {e}")
            return []
    elif settings.vector_db_type == VectorDBType.CHROMA:
        try:
            import chromadb
            client = chromadb.PersistentClient(settings.chroma_persist_dir)
            out: list[tuple[str, int]] = []
            for c in client.list_collections():
                name = getattr(c, "name", None) or str(c)
                try:
                    cnt = int(c.count())
                except Exception:
                    cnt = 0
                if cnt >= min_count:
                    out.append((name, cnt))
            return sorted(out, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.warning(f"list_collections (chroma) failed: {e}")
            return []
    return []


def clear_vectorstore_collection(collection_name: str) -> int:
    """Clear all embeddings from a collection. Returns number of rows deleted."""
    settings = get_rag_settings()
    if settings.vector_db_type == VectorDBType.PGVECTOR:
        try:
            import psycopg
            conn = psycopg.connect(settings.postgres_dsn, autocommit=True)
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM langchain_pg_embedding WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = %s)",
                (collection_name,),
            )
            deleted = cur.rowcount
            conn.close()
            logger.info(f"Cleared {deleted} embeddings from collection '{collection_name}'")
            return deleted
        except Exception as e:
            logger.warning(f"Failed to clear collection '{collection_name}': {e}")
            return 0
    elif settings.vector_db_type == VectorDBType.CHROMA:
        try:
            from langchain_chroma import Chroma
            embeddings = get_embeddings()
            vs = Chroma(
                persist_directory=settings.chroma_persist_dir,
                collection_name=collection_name,
                embedding_function=embeddings,
            )
            vs._collection.delete(where={"$ne": None})
            return 0
        except Exception:
            return 0
    return 0


def index_documents(
    input_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
    chunking_config: Optional[ChunkingConfig] = None,
    llm=None,
) -> int:
    """Full indexing pipeline: load → chunk → embed → store.

    This is the main entry point for ingesting documents into the RAG system.

    Args:
        input_dir: Directory with source documents (default: settings.data_dir).
        collection_name: Vector store collection (default: settings.default_collection).
        chunking_config: Chunking strategy and parameters.
        llm: Required only for CONTEXTUAL chunking strategy.

    Returns:
        Number of chunks successfully indexed.
    """
    settings = get_rag_settings()
    input_dir = input_dir or settings.data_dir
    collection_name = collection_name or settings.default_collection
    chunking_config = chunking_config or ChunkingConfig()

    # Step 1: Load raw documents from disk
    logger.info("Step 1/4: Loading documents...")
    documents = load_documents(input_dir)
    if not documents:
        logger.warning("No documents found to index.")
        return 0

    # Step 2: Chunk documents using the configured strategy
    logger.info(f"Step 2/4: Chunking ({chunking_config.strategy.value})...")
    if chunking_config.strategy == ChunkingStrategy.PARENT_CHILD:
        chunks, parent_records = chunk_documents_with_parents(documents, chunking_config, llm=llm)
    else:
        chunks = chunk_documents(documents, chunking_config, llm=llm)
        parent_records = []

    # Step 3: Store parent records (if any)
    if parent_records:
        logger.info(f"Step 3/4: Storing {len(parent_records)} parent records...")
        from rag.parents import store_parents, clear_collection
        clear_collection(collection_name)
        store_parents(parent_records, collection_name)

    # Step 4: Embed and store in vector DB
    logger.info("Step 4/4: Embedding and storing...")
    vectorstore = get_vectorstore(collection_name)
    vectorstore.add_documents(chunks)

    logger.info(f"Indexed {len(chunks)} chunks into '{collection_name}'")
    return len(chunks)
