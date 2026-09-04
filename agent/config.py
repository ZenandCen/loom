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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# DEV: in-memory
# PROD: PostgresSaver + PlatformStore
checkpointer = MemorySaver()
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


def build_backend():
    """Composite backend: scratch → state, workspace → disk, memories → store."""
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": FilesystemBackend(root_dir=WORKSPACE_DIR, virtual_mode=True),
            "/memories/": StoreBackend(
                namespace=lambda rt: ("loom", "filesystem")
            ),
        },
    )
