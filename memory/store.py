"""PostgreSQL-backed memory store with semantic search.

Implements the LangGraph BaseStore interface backed by PostgreSQL + pgvector.
Supports hybrid search (vector cosine similarity + keyword ILIKE).

Schema:
    memory_store (
        id BIGSERIAL PRIMARY KEY,
        namespace TEXT[] NOT NULL,
        key TEXT NOT NULL,
        value JSONB NOT NULL,
        embedding vector(768),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE(namespace, key)
    )
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# Lazy-loaded embedding function (avoids import-time Ollama connection)
_embedding_fn = None


def _get_embeddings():
    """Lazily load the embedding function from RAG config."""
    global _embedding_fn
    if _embedding_fn is None:
        try:
            from rag.config import get_embeddings
            _embedding_fn = get_embeddings()
        except Exception as e:
            logger.warning(f"Embeddings unavailable: {e}")
            _embedding_fn = False
    return _embedding_fn if _embedding_fn else None


@dataclass
class Item:
    """A single stored item (matches LangGraph BaseStore interface)."""
    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any]
    created_at: str = ""
    updated_at: str = ""


@dataclass
class SearchItem(Item):
    """Result item from search with score."""
    score: float | None = None


class PostgresStore:
    """PostgreSQL-backed store implementing LangGraph BaseStore interface.

    Compatible with: store.put(), store.get(), store.search(),
    store.list_namespaces(), store.delete()
    """

    def __init__(self, dsn: str | None = None, table: str = "memory_store"):
        self.dsn = dsn or os.getenv("RAG_PG_DSN", "")
        if not self.dsn:
            raise ValueError("PostgresStore requires RAG_PG_DSN env var")
        self.table = table
        self._ensure_schema()
        logger.info(f"PostgresStore ready (table={table})")

    def _conn(self):
        return psycopg2.connect(self.dsn)

    def _ensure_schema(self):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE EXTENSION IF NOT EXISTS vector
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS memory_store (
                        id BIGSERIAL PRIMARY KEY,
                        namespace TEXT[] NOT NULL,
                        key TEXT NOT NULL,
                        value JSONB NOT NULL,
                        embedding vector(768),
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(namespace, key)
                    )
                """)
                # Migrate: add embedding column if table existed before
                cur.execute("""
                    ALTER TABLE memory_store ADD COLUMN IF NOT EXISTS embedding vector(768)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_namespace
                    ON memory_store USING GIN(namespace)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_namespace_prefix
                    ON memory_store (namespace)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_embedding
                    ON memory_store USING ivfflat (embedding vector_cosine_ops)
                """)
        logger.info("Schema verified (with pgvector)")

    def _generate_embedding(self, key: str, value: dict[str, Any]) -> list[float] | None:
        """Generate embedding for key+value text."""
        emb = _get_embeddings()
        if emb is None:
            return None
        try:
            text = f"{key}: {json.dumps(value, ensure_ascii=False)}"[:2000]
            return emb.embed_query(text)
        except Exception as e:
            logger.debug(f"Embedding generation failed: {e}")
            return None

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        """Insert or update an item (with embedding)."""
        ns = list(namespace)
        embedding = self._generate_embedding(key, value)
        with self._conn() as conn:
            with conn.cursor() as cur:
                if embedding:
                    cur.execute(
                        f"""
                        INSERT INTO {self.table} (namespace, key, value, embedding, updated_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        ON CONFLICT (namespace, key)
                        DO UPDATE SET value = EXCLUDED.value, embedding = EXCLUDED.embedding, updated_at = NOW()
                        """,
                        (ns, key, json.dumps(value), str(embedding)),
                    )
                else:
                    cur.execute(
                        f"""
                        INSERT INTO {self.table} (namespace, key, value, updated_at)
                        VALUES (%s, %s, %s, NOW())
                        ON CONFLICT (namespace, key)
                        DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                        """,
                        (ns, key, json.dumps(value)),
                    )
        logger.debug(f"PUT {ns}/{key}")

    def get(self, namespace: tuple[str, ...], key: str) -> Item | None:
        """Retrieve a single item by namespace + key."""
        ns = list(namespace)
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"SELECT namespace, key, value, created_at, updated_at FROM {self.table} WHERE namespace = %s AND key = %s",
                    (ns, key),
                )
                row = cur.fetchone()
        if not row:
            return None
        return Item(
            namespace=tuple(row["namespace"]),
            key=row["key"],
            value=row["value"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def search(
        self,
        namespace: tuple[str, ...],
        query: str = "",
        limit: int = 10,
        **kwargs,
    ) -> list[SearchItem]:
        """Hybrid search: vector cosine similarity + keyword ILIKE.

        If query is provided and embeddings are available, uses vector search
        merged with keyword results. Falls back to keyword-only if no embeddings.
        """
        ns = list(namespace)
        ns_len = len(ns)

        if not query:
            with self._conn() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        f"""
                        SELECT namespace, key, value, created_at, updated_at
                        FROM {self.table}
                        WHERE namespace[1:%s] = %s
                        ORDER BY updated_at DESC
                        LIMIT %s
                        """,
                        (ns_len, ns, limit),
                    )
                    rows = cur.fetchall()
            return [
                SearchItem(
                    namespace=tuple(r["namespace"]),
                    key=r["key"],
                    value=r["value"],
                    created_at=str(r["created_at"]),
                    updated_at=str(r["updated_at"]),
                )
                for r in rows
            ]

        # Hybrid: try vector + keyword, merge results
        vector_results: list[SearchItem] = []
        keyword_results: list[SearchItem] = []

        # Vector search
        emb = _get_embeddings()
        if emb:
            try:
                query_vec = emb.embed_query(query[:2000])
                with self._conn() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(
                            f"""
                            SELECT namespace, key, value, created_at, updated_at,
                                   1 - (embedding <=> %s::vector) AS score
                            FROM {self.table}
                            WHERE namespace[1:%s] = %s
                            AND embedding IS NOT NULL
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s
                            """,
                            (str(query_vec), ns_len, ns, str(query_vec), limit * 2),
                        )
                        rows = cur.fetchall()
                vector_results = [
                    SearchItem(
                        namespace=tuple(r["namespace"]),
                        key=r["key"],
                        value=r["value"],
                        created_at=str(r["created_at"]),
                        updated_at=str(r["updated_at"]),
                        score=float(r["score"]),
                    )
                    for r in rows
                ]
            except Exception as e:
                logger.debug(f"Vector search failed, falling back to keyword: {e}")

        # Keyword search (always run as fallback/complement)
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT namespace, key, value, created_at, updated_at
                    FROM {self.table}
                    WHERE namespace[1:%s] = %s
                    AND (value::text ILIKE %s OR key ILIKE %s)
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (ns_len, ns, f"%{query}%", f"%{query}%", limit * 2),
                )
                rows = cur.fetchall()
        keyword_results = [
            SearchItem(
                namespace=tuple(r["namespace"]),
                key=r["key"],
                value=r["value"],
                created_at=str(r["created_at"]),
                updated_at=str(r["updated_at"]),
            )
            for r in rows
        ]

        # Merge: vector results first (scored), then keyword-only (deduped)
        seen_keys: set[tuple[str, str]] = set()
        merged: list[SearchItem] = []
        for item in vector_results:
            ident = (str(item.namespace), item.key)
            if ident not in seen_keys:
                seen_keys.add(ident)
                merged.append(item)
        for item in keyword_results:
            ident = (str(item.namespace), item.key)
            if ident not in seen_keys:
                seen_keys.add(ident)
                merged.append(item)
        return merged[:limit]

    def list_namespaces(self, prefix: tuple[str, ...] = (), limit: int = 100) -> list[tuple[str, ...]]:
        """List all unique namespaces, optionally filtered by prefix."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                if prefix:
                    prefix_list = list(prefix)
                    cur.execute(
                        f"SELECT DISTINCT namespace FROM {self.table} WHERE namespace[1:%s] = %s ORDER BY 1 LIMIT %s",
                        (len(prefix_list), prefix_list, limit),
                    )
                else:
                    cur.execute(
                        f"SELECT DISTINCT namespace FROM {self.table} ORDER BY 1 LIMIT %s",
                        (limit,),
                    )
                rows = cur.fetchall()
        return [tuple(r[0]) for r in rows]

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete an item by namespace + key."""
        ns = list(namespace)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.table} WHERE namespace = %s AND key = %s",
                    (ns, key),
                )
        logger.debug(f"DELETE {ns}/{key}")

    def count(self, namespace: tuple[str, ...] | None = None) -> int:
        """Count items, optionally filtered by namespace prefix."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                if namespace:
                    cur.execute(
                        f"SELECT COUNT(*) FROM {self.table} WHERE namespace[1:%s] = %s",
                        (len(namespace), list(namespace)),
                    )
                else:
                    cur.execute(f"SELECT COUNT(*) FROM {self.table}")
                return cur.fetchone()[0]

    def raw_query(self, namespace: tuple[str, ...] | None = None, key: str | None = None, limit: int = 50) -> list[dict]:
        """Raw query for API inspection. Returns all columns as dicts."""
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                conditions = []
                params = []
                if namespace:
                    conditions.append("namespace[1:%s] = %s")
                    params.extend([len(namespace), list(namespace)])
                if key:
                    conditions.append("key = %s")
                    params.append(key)

                where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
                params.append(limit)

                cur.execute(
                    f"SELECT namespace, key, value, created_at, updated_at FROM {self.table} {where} ORDER BY updated_at DESC LIMIT %s",
                    params,
                )
                rows = cur.fetchall()
        return [dict(r) for r in rows]
