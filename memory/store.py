"""PostgreSQL-backed memory store.

Implements the LangGraph BaseStore interface backed by PostgreSQL.
Replaces InMemoryStore for production use — data persists across restarts.

Schema:
    memory_store (
        id BIGSERIAL PRIMARY KEY,
        namespace TEXT[] NOT NULL,
        key TEXT NOT NULL,
        value JSONB NOT NULL,
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
                    CREATE TABLE IF NOT EXISTS memory_store (
                        id BIGSERIAL PRIMARY KEY,
                        namespace TEXT[] NOT NULL,
                        key TEXT NOT NULL,
                        value JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(namespace, key)
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_namespace
                    ON memory_store USING GIN(namespace)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_namespace_prefix
                    ON memory_store (namespace)
                """)
        logger.info("Schema verified")

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        """Insert or update an item."""
        ns = list(namespace)
        with self._conn() as conn:
            with conn.cursor() as cur:
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
        """Search items by namespace prefix + keyword in value."""
        ns = list(namespace)
        ns_len = len(ns)
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if query:
                    cur.execute(
                        f"""
                        SELECT namespace, key, value, created_at, updated_at
                        FROM {self.table}
                        WHERE namespace[1:%s] = %s
                        AND (value::text ILIKE %s OR key ILIKE %s)
                        ORDER BY updated_at DESC
                        LIMIT %s
                        """,
                        (ns_len, ns, f"%{query}%", f"%{query}%", limit),
                    )
                else:
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
