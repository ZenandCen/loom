"""Centralized infrastructure setup. Import from here in all modules."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(override=True)

# --- Patch: Fix unicode encoding in OpenAI client ---
# Model/tool có thể trả về surrogate chars (VD: Vietnamese diacritics).
# Khi OpenAI client serialize message history → .encode("utf-8") → crash.
# Fix: patch openapi_dumps TẠI VỊ TRÍ NÓ ĐƯỢC DÙNG (trong _base_client).
import json as _json
import openai._base_client as _openai_client


def _safe_dumps(data, **kwargs):
    """Drop surrogates before JSON serialization to prevent UnicodeEncodeError."""
    def _fix_surrogates(obj):
        if isinstance(obj, str):
            return obj.encode("utf-8", errors="replace").decode("utf-8")
        elif isinstance(obj, dict):
            return {k: _fix_surrogates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_fix_surrogates(i) for i in obj]
        return obj

    data = _fix_surrogates(data)
    kwargs.setdefault("ensure_ascii", False)
    return _json.dumps(data, **kwargs).encode("utf-8")


_openai_client.openapi_dumps = _safe_dumps

# --- LangSmith Tracing ---
if os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes"):
    import langsmith

    langsmith.api_key = os.environ["LANGSMITH_API_KEY"]
    langsmith.project_name = os.getenv("LANGSMITH_PROJECT", "default")
    langsmith.tracing_v2 = True
    print(f"  [LANGSMITH] Tracing → {langsmith.project_name}")

# --- Model (delegated to utils) ---
from utils.models import LLM, get_llm  # noqa: F401

# --- Memory Infrastructure ---
# Checkpointer: per-session state (persistent via Postgres connection pool)
_pg_dsn = os.getenv("RAG_PG_DSN", "")

# PostgresSaver in the installed langgraph-checkpoint-postgres only implements
# sync methods; its async methods (aget_tuple, aput, ...) raise NotImplementedError.
# Since the team graph nodes are async-only (must use ainvoke), we subclass and
# bridge the async methods to the working sync ones via a thread pool.
import asyncio
from langgraph.checkpoint.postgres import PostgresSaver


class AsyncPostgresSaver(PostgresSaver):
    """PostgresSaver with async methods bridged to sync implementations."""

    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        results = await asyncio.to_thread(
            list, self.list(config, filter=filter, before=before, limit=limit)
        )
        for item in results:
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id):
        return await asyncio.to_thread(self.delete_thread, thread_id)


try:
    if _pg_dsn:
        import psycopg
        from psycopg_pool import ConnectionPool

        # Run DDL setup with autocommit (CREATE INDEX CONCURRENTLY requires it)
        _setup_conn = psycopg.connect(_pg_dsn, autocommit=True)
        _setup_cur = _setup_conn.cursor()
        _setup_cur.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                checkpoint JSONB NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}',
                PRIMARY KEY (thread_id, checkpoint_id)
            )
        """)
        _setup_cur.execute("""
            CREATE TABLE IF NOT EXISTS checkpoint_writes (
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                blob BYTEA NOT NULL,
                PRIMARY KEY (thread_id, checkpoint_id, task_id, idx)
            )
        """)
        _setup_cur.execute("""
            CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                thread_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                version TEXT NOT NULL,
                type TEXT NOT NULL,
                blob BYTEA NOT NULL,
                PRIMARY KEY (thread_id, channel, version)
            )
        """)
        _setup_cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_parents (
                id TEXT PRIMARY KEY,
                collection TEXT NOT NULL,
                source TEXT,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        _setup_cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_parents_collection ON rag_parents(collection)
        """)
        _setup_conn.close()

        # Use pool for runtime operations
        _pg_pool = ConnectionPool(_pg_dsn, open=True, min_size=1, max_size=5)
        checkpointer = AsyncPostgresSaver(_pg_pool)
        print(f"  [CHECKPOINTER] AsyncPostgresSaver (pool) ready")
    else:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        print(f"  [CHECKPOINTER] MemorySaver (no PG_DSN)")
except Exception as e:
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    print(f"  [CHECKPOINTER] MemorySaver (fallback: {e})")

# Store: long-term memory per (project, user, category)
# Use PostgresStore if RAG_PG_DSN is set, fallback to InMemoryStore
try:
    from memory.store import PostgresStore
    store = PostgresStore()
except Exception:
    from langgraph.store.memory import InMemoryStore
    store = InMemoryStore()

# --- Backend (Composite — hybrid storage) ---
from deepagents.backends import (
    StateBackend,
    FilesystemBackend,
    StoreBackend,
    CompositeBackend,
)

WORKSPACE_DIR = project_root / "workspace"
WORKSPACE_DIR.mkdir(exist_ok=True)


def _memories_namespace(rt):
    """Resolve per-user namespace for StoreBackend."""
    try:
        from langgraph.config import get_config
        config = get_config()
        user_id = config.get("configurable", {}).get("user_id", "default")
    except Exception:
        user_id = "default"
    return ("loom", user_id, "filesystem")


def build_backend():
    """Composite backend: scratch → state, workspace → disk, memories → store.

    Routes:
        /            → State (RAM, session-scoped)
        /workspace/  → Disk (./workspace/ directory, persistent)
        /memories/   → PostgresStore (per-user namespace, persistent)
    """
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": FilesystemBackend(root_dir=WORKSPACE_DIR, virtual_mode=True),
            "/memories/": StoreBackend(
                namespace=_memories_namespace
            ),
        },
    )
