"""Parent chunk storage and retrieval.

Stores full parent chunks in a separate `rag_parents` Postgres table.
At retrieval time, children (small, precise) are matched via vector search,
then their parents (large, contextual) are fetched for LLM context.
"""

import json
import logging
from typing import Optional

import psycopg2
import psycopg2.extras
from pydantic import BaseModel, Field

from rag.config import get_rag_settings

logger = logging.getLogger(__name__)


class ParentRecord(BaseModel):
    """A parent chunk stored in the rag_parents table."""

    id: str
    collection: str
    source: str = ""
    content: str
    metadata: dict = Field(default_factory=dict)


def _get_connection():
    settings = get_rag_settings()
    return psycopg2.connect(settings.postgres_dsn)


def store_parents(
    parents: list[dict],
    collection: str,
) -> int:
    """Bulk upsert parent records into rag_parents table.

    Args:
        parents: List of dicts with keys: id, source, content, metadata (optional).
        collection: Vector store collection name these parents belong to.

    Returns:
        Number of records upserted.
    """
    if not parents:
        return 0

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO rag_parents (id, collection, source, content, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    source = EXCLUDED.source,
                    metadata = EXCLUDED.metadata
                """,
                [
                    (
                        p["id"],
                        collection,
                        p.get("source", ""),
                        p["content"],
                        json.dumps(p.get("metadata", {})),
                    )
                    for p in parents
                ],
            )
        conn.commit()
        logger.info(f"Stored {len(parents)} parent records for '{collection}'")
        return len(parents)
    finally:
        conn.close()


def fetch_parents(
    parent_ids: list[str],
    collection: Optional[str] = None,
) -> list[ParentRecord]:
    """Fetch parent records by their IDs.

    Args:
        parent_ids: List of parent UUID strings.
        collection: Optional filter by collection.

    Returns:
        List of ParentRecord objects (may be fewer than requested if some IDs missing).
    """
    if not parent_ids:
        return []

    conn = _get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if collection:
                cur.execute(
                    """
                    SELECT id, collection, source, content, metadata
                    FROM rag_parents
                    WHERE id = ANY(%s) AND collection = %s
                    """,
                    (parent_ids, collection),
                )
            else:
                cur.execute(
                    """
                    SELECT id, collection, source, content, metadata
                    FROM rag_parents
                    WHERE id = ANY(%s)
                    """,
                    (parent_ids,),
                )
            rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for row in rows:
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        results.append(
            ParentRecord(
                id=row["id"],
                collection=row["collection"],
                source=row["source"] or "",
                content=row["content"],
                metadata=meta or {},
            )
        )
    return results


def clear_collection(collection: str) -> int:
    """Remove all parent records for a collection (used before reindex).

    Returns:
        Number of records deleted.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM rag_parents WHERE collection = %s", (collection,))
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()
