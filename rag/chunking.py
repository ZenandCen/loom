"""Document chunking strategies.

Chunking splits documents into smaller pieces for embedding and retrieval.
The right strategy depends on document structure and use case.

Strategies:
    - RECURSIVE: General-purpose, splits by text boundaries
    - HEADER: Structure-aware, splits on Markdown headers
    - PARENT_CHILD: Hierarchical, small children for retrieval + large parents for context
    - CONTEXTUAL: LLM-enhanced, prepends context prefix to each chunk
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.config import get_rag_settings
from rag.enums import ChunkingStrategy

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Tunable parameters for chunking.

    Defaults are loaded from RAGSettings but can be overridden
    per-call without changing global config.
    """

    strategy: ChunkingStrategy = field(default_factory=lambda: get_rag_settings().chunking_strategy)
    chunk_size: int = field(default_factory=lambda: get_rag_settings().chunk_size)
    chunk_overlap: int = field(default_factory=lambda: get_rag_settings().chunk_overlap)
    # Parent-child specific
    parent_chunk_size: int = field(default_factory=lambda: get_rag_settings().parent_chunk_size)
    child_chunk_size: int = field(default_factory=lambda: get_rag_settings().child_chunk_size)
    # Header specific: (marker, metadata_key) pairs
    headers_to_split_on: list[tuple[str, str]] = field(
        default_factory=lambda: [("#", "h1"), ("##", "h2"), ("###", "h3")]
    )


def recursive_chunk(text: str, config: Optional[ChunkingConfig] = None) -> list[str]:
    """Split text by hierarchy: paragraphs → sentences → words → chars.

    The default workhorse strategy. Respects natural text boundaries
    and works well for most document types.

    Args:
        text: Raw document text to split.
        config: Chunking parameters (size, overlap, separators).

    Returns:
        List of text chunks, each ≤ chunk_size tokens with overlap.
    """
    config = config or ChunkingConfig()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def header_chunk(text: str, config: Optional[ChunkingConfig] = None) -> list[Document]:
    """Split Markdown document on headers, preserving section hierarchy.

    Each chunk carries its header path as metadata, enabling
    filtered retrieval (e.g., "only search in the API section").

    Args:
        text: Markdown document with #, ##, ### headers.
        config: Chunking parameters.

    Returns:
        List of Documents with header metadata.
    """
    config = config or ChunkingConfig()
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=config.headers_to_split_on)
    return splitter.split_text(text)


def parent_child_chunk(
    text: str, config: Optional[ChunkingConfig] = None, source: str = ""
) -> tuple[list[Document], list[dict]]:
    """Two-level hierarchical chunking.

    Creates small children (good for precise retrieval/embedding)
    and large parents (good for providing full context to LLM).
    Each child references its parent via `parent_id` metadata (UUID string).

    Args:
        text: Raw document text.
        config: Chunking parameters.
        source: Source file path (stored in parent record).

    Returns:
        Tuple of (children, parent_records):
            - children: Index these in the vector store (parent_id = UUID string)
            - parent_records: List of dicts {"id": uuid_str, "content": str, "source": str}
              to be stored in the rag_parents table
    """
    config = config or ChunkingConfig()

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=config.parent_chunk_size)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.child_chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    parent_texts = parent_splitter.split_text(text)
    children: list[Document] = []
    parent_records: list[dict] = []

    for i, parent_text in enumerate(parent_texts):
        parent_id = str(uuid.uuid4())
        parent_records.append({
            "id": parent_id,
            "content": parent_text,
            "source": source,
        })
        child_chunks = child_splitter.split_text(parent_text)
        for j, child in enumerate(child_chunks):
            children.append(
                Document(
                    page_content=child,
                    metadata={"id": f"{parent_id}_c{j}", "parent_id": parent_id},
                )
            )

    logger.debug(f"Parent-child: {len(parent_records)} parents, {len(children)} children")
    return children, parent_records


def contextual_chunk(
    text: str,
    llm,
    config: Optional[ChunkingConfig] = None,
) -> list[Document]:
    """LLM-enhanced chunking with contextual prefixes.

    For each chunk, uses the LLM to generate a 1-2 sentence prefix
    that situates the chunk within the full document. This dramatically
    improves retrieval precision by giving embeddings more context.

    Reference: Anthropic's contextual retrieval (-49% retrieval failures).

    Args:
        text: Full document text.
        llm: LangChain LLM instance for generating context.
        config: Chunking parameters.

    Returns:
        List of Documents with contextual prefixes prepended.
    """
    config = config or ChunkingConfig()
    base_chunks = recursive_chunk(text, config)
    # Truncate document for cost control (LLM context window)
    doc_summary = text[:8000]

    contextual_docs: list[Document] = []
    for i, chunk in enumerate(base_chunks):
        prompt = (
            f"Given the following document, generate a 1-2 sentence context prefix "
            f"that situates this chunk within the document. Be specific about the "
            f"section/topic this chunk covers.\n\n"
            f"Document:\n{doc_summary}\n\n"
            f"Chunk:\n{chunk}\n\n"
            f"Context prefix (1-2 sentences):"
        )
        context = llm.invoke(prompt).content.strip()
        contextual_docs.append(
            Document(
                page_content=f"{context}\n\n{chunk}",
                metadata={"chunk_index": i, "original_chunk": chunk},
            )
        )

    logger.debug(f"Contextual: {len(contextual_docs)} chunks generated")
    return contextual_docs


def chunk_documents(
    documents: list[Document],
    config: Optional[ChunkingConfig] = None,
    llm=None,
) -> list[Document]:
    """Apply the configured chunking strategy to a list of documents.

    This is the main entry point for chunking. It dispatches to the
    appropriate strategy based on config and merges source metadata.

    Fast-path optimization: if a document unit is already ≤ chunk_size,
    it's kept as-is without splitting. This is critical for per-page PDFs
    where most pages are naturally small — splitting them would destroy
    the page-level coherence that makes retrieval precise.

    Args:
        documents: Input documents to chunk (may already be fine-grained units).
        config: Chunking strategy and parameters.
        llm: Required only for CONTEXTUAL strategy.

    Returns:
        Flattened list of chunked Documents with source metadata preserved.
    """
    chunks, _ = chunk_documents_with_parents(documents, config, llm=llm)
    return chunks


def chunk_documents_with_parents(
    documents: list[Document],
    config: Optional[ChunkingConfig] = None,
    llm=None,
) -> tuple[list[Document], list[dict]]:
    """Apply chunking and return both chunks and parent records.

    Same as chunk_documents() but also returns parent records for
    parent_child strategy (to be stored in rag_parents table).

    Args:
        documents: Input documents to chunk.
        config: Chunking strategy and parameters.
        llm: Required only for CONTEXTUAL strategy.

    Returns:
        Tuple of (chunks, parent_records):
            - chunks: Flattened list of chunked Documents
            - parent_records: List of dicts for rag_parents (empty if not parent_child)
    """
    config = config or ChunkingConfig()
    all_chunks: list[Document] = []
    all_parents: list[dict] = []
    skipped = 0

    for doc in documents:
        source = doc.metadata.get("source", "")

        # Fast-path: unit already fits within chunk_size → no need to split
        if len(doc.page_content) <= config.chunk_size:
            all_chunks.append(Document(page_content=doc.page_content, metadata=doc.metadata))
            skipped += 1
            continue

        if config.strategy == ChunkingStrategy.HEADER:
            chunks = header_chunk(doc.page_content, config)
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
            all_chunks.extend(chunks)

        elif config.strategy == ChunkingStrategy.PARENT_CHILD:
            children, parent_records = parent_child_chunk(doc.page_content, config, source=source)
            for child in children:
                child.metadata.update(doc.metadata)
            all_chunks.extend(children)
            all_parents.extend(parent_records)

        elif config.strategy == ChunkingStrategy.CONTEXTUAL and llm is not None:
            chunks = contextual_chunk(doc.page_content, llm, config)
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
            all_chunks.extend(chunks)

        else:  # RECURSIVE (default)
            chunks = recursive_chunk(doc.page_content, config)
            all_chunks.extend(
                Document(page_content=c, metadata={**doc.metadata, "chunk_index": i})
                for i, c in enumerate(chunks)
            )

    logger.info(
        f"Chunked {len(documents)} docs → {len(all_chunks)} chunks "
        f"({config.strategy}, {skipped} kept whole, {len(all_parents)} parents)"
    )
    return all_chunks, all_parents
