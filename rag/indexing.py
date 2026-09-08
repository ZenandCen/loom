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

from rag.chunking import ChunkingConfig, chunk_documents
from rag.config import get_embeddings, get_rag_settings
from rag.enums import VectorDBType

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
                documents.extend(docs)
                logger.info(f"Loaded: {file_path.name} ({len(docs)} units, {sum(len(d.page_content) for d in docs)} chars)")
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
    logger.info("Step 1/3: Loading documents...")
    documents = load_documents(input_dir)
    if not documents:
        logger.warning("No documents found to index.")
        return 0

    # Step 2: Chunk documents using the configured strategy
    logger.info(f"Step 2/3: Chunking ({chunking_config.strategy.value})...")
    chunks = chunk_documents(documents, chunking_config, llm=llm)

    # Step 3: Embed and store in vector DB
    logger.info("Step 3/3: Embedding and storing...")
    vectorstore = get_vectorstore(collection_name)
    vectorstore.add_documents(chunks)

    logger.info(f"Indexed {len(chunks)} chunks into '{collection_name}'")
    return len(chunks)
