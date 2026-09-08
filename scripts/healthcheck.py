"""Healthcheck — verifies the entire system is ready to run.

Usage:
    python -m scripts.healthcheck

Checks:
    1. Environment variables (presence + format)
    2. Enum values parse correctly
    3. Ollama server reachable + model available
    4. Vector DB connectivity
    5. Data directory exists
    6. RAG module imports + pipelines compile
    7. Quick embedding round-trip
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class HealthCheck:
    """Collects check results and reports at the end."""

    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        self.results.append((name, condition, detail))
        return condition

    def report(self) -> bool:
        all_pass = True
        width = 40
        print("\n" + "=" * (width + 12))
        print("  LOOM HEALTHCHECK")
        print("=" * (width + 12))

        for name, passed, detail in self.results:
            icon = "✓" if passed else "✗"
            line = f"  {icon} {name:<{width}}"
            if not passed:
                line += f"  FAIL"
            print(line)
            if detail and not passed:
                print(f"    → {detail}")
            all_pass = all_pass and passed

        passed_count = sum(1 for _, p, _ in self.results if p)
        total = len(self.results)
        print("-" * (width + 12))
        if all_pass:
            print(f"  ALL {total} CHECKS PASSED")
        else:
            print(f"  {passed_count}/{total} PASSED, {total - passed_count} FAILED")
        print("=" * (width + 12) + "\n")

        return all_pass


def check_env_vars(hc: HealthCheck):
    """Verify required environment variables are present and well-formed."""
    from dotenv import load_dotenv
    load_dotenv(override=True)

    required = {
        "OPENAI_API_KEY": "LLM API key",
        "TAVILY_API_KEY": "Web search API key",
    }

    for var, desc in required.items():
        value = os.getenv(var, "")
        hc.check(
            f"Env: {var}",
            bool(value) and len(value) > 5,
            f"Missing or too short ({desc})",
        )

    # Check for trailing whitespace in enum values
    enum_vars = {
        "RAG_CHUNKING_STRATEGY": ["recursive", "header", "parent_child", "contextual"],
        "RAG_PIPELINE": ["basic", "adaptive", "self_rag", "multi_source"],
        "RAG_EMBEDDING_PROVIDER": ["openai", "ollama"],
    }

    for var, valid_values in enum_vars.items():
        value = os.getenv(var, "")
        if value:
            is_valid = value in valid_values
            hc.check(
                f"Env: {var} = '{value}'",
                is_valid,
                f"Value '{value}' not in {valid_values} (check for trailing spaces)",
            )


def check_rag_config(hc: HealthCheck):
    """Verify RAG settings parse correctly (no enum errors)."""
    try:
        from rag.config import get_rag_settings
        from rag.enums import ChunkingStrategy, PipelineLevel, EmbeddingProvider, VectorDBType

        settings = get_rag_settings()

        hc.check("RAG: chunking_strategy parses", True, "")
        hc.check("RAG: pipeline_level parses", True, "")
        hc.check("RAG: embedding_provider parses", True, "")

        # Check vector DB type resolution
        vdb = settings.vector_db_type
        hc.check(
            f"RAG: vector_db = {vdb.value}",
            True,
            ""
        )

        # Warn if both Chroma and PG are configured
        if settings.chroma_persist_dir and settings.postgres_dsn:
            hc.check(
                "RAG: single vector DB configured",
                False,
                f"Both RAG_CHROMA_DIR and RAG_PG_DSN set. "
                f"Active: {vdb.value} (priority: qdrant > pgvector > chroma). "
                f"Comment out the one you don't use.",
            )

    except Exception as e:
        hc.check("RAG: config loads", False, str(e))


def check_ollama(hc: HealthCheck):
    """Verify Ollama server is running and model is available."""
    from rag.config import get_rag_settings
    from rag.enums import EmbeddingProvider

    settings = get_rag_settings()

    if settings.embedding_provider != EmbeddingProvider.OLLAMA:
        print("  (skip Ollama check — using different provider)")
        return

    try:
        import urllib.request
        import json

        url = f"{settings.ollama_url}/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())

        models = [m["name"] for m in data.get("models", [])]
        hc.check("Ollama: server reachable", True, "")

        # Check if the specific model is available
        model_base = settings.embedding_model.split(":")[0]
        model_found = any(model_base in m for m in models)
        hc.check(
            f"Ollama: model '{settings.embedding_model}' available",
            model_found,
            f"Available models: {models}. Run: ollama pull {settings.embedding_model}",
        )

    except Exception as e:
        hc.check("Ollama: server reachable", False, str(e))


def check_vector_db(hc: HealthCheck):
    """Verify vector DB is accessible."""
    from rag.config import get_rag_settings
    from rag.enums import VectorDBType

    settings = get_rag_settings()

    try:
        if settings.vector_db_type == VectorDBType.CHROMA:
            # Chroma is embedded — just check the directory can be created
            chroma_dir = Path(settings.chroma_persist_dir)
            chroma_dir.mkdir(parents=True, exist_ok=True)
            hc.check("VectorDB: Chroma directory writable", True, "")

        elif settings.vector_db_type == VectorDBType.PGVECTOR:
            import psycopg2
            # Parse DSN
            import re
            dsn = settings.postgres_dsn
            match = re.match(
                r"postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(\w+)", dsn
            )
            if not match:
                hc.check("VectorDB: PG DSN format", False, f"Invalid DSN: {dsn[:30]}...")
                return
            user, password, host, port, dbname = match.groups()
            conn = psycopg2.connect(
                host=host, port=port, dbname=dbname,
                user=user, password=password, connect_timeout=5
            )
            cur = conn.cursor()
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            has_vector = cur.fetchone() is not None
            cur.close()
            conn.close()
            hc.check("VectorDB: PostgreSQL reachable", True, "")
            hc.check(
                "VectorDB: pgvector extension installed",
                has_vector,
                "Run: CREATE EXTENSION vector; in your database",
            )

        elif settings.vector_db_type == VectorDBType.QDRANT:
            import urllib.request
            url = f"{settings.qdrant_url}/collections"
            with urllib.request.urlopen(url, timeout=5) as resp:
                resp.read()
            hc.check("VectorDB: Qdrant reachable", True, "")

    except ImportError as e:
        hc.check("VectorDB: client library installed", False, f"pip install missing: {e}")
    except Exception as e:
        hc.check("VectorDB: connectivity", False, str(e))


def check_data_dir(hc: HealthCheck):
    """Verify data directory exists and has files."""
    from rag.config import get_rag_settings

    settings = get_rag_settings()
    data_dir = Path(settings.data_dir)

    hc.check(
        f"Data: directory '{data_dir}' exists",
        data_dir.exists(),
        f"Create it: mkdir -p {data_dir} (add documents to index)",
    )

    if data_dir.exists():
        from rag.indexing import SUPPORTED_EXTENSIONS
        files = [
            f for f in data_dir.rglob("*")
            if f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.startswith(".")
        ]
        hc.check(
            f"Data: {len(files)} indexable files found",
            len(files) > 0,
            f"No supported files in {data_dir} (PDF, MD, TXT, HTML, DOCX, CSV)",
        )


def check_rag_module(hc: HealthCheck):
    """Verify RAG module imports and pipelines compile."""
    try:
        from rag.pipeline import basic_pipeline, adaptive_pipeline, self_rag_pipeline
        from rag.tool import rag_query

        p1 = basic_pipeline()
        p2 = adaptive_pipeline()
        p3 = self_rag_pipeline()

        hc.check("RAG: basic pipeline compiles", True, "")
        hc.check("RAG: adaptive pipeline compiles", True, "")
        hc.check("RAG: self_rag pipeline compiles", True, "")
        hc.check("RAG: rag_query tool exists", rag_query is not None, "")

    except Exception as e:
        hc.check("RAG: module imports", False, str(e))


def check_embedding_roundtrip(hc: HealthCheck):
    """Generate an embedding to verify the full embedding pipeline works."""
    try:
        from rag.config import get_embeddings

        emb = get_embeddings()
        start = time.time()
        vectors = emb.embed_documents(["hello world"])
        elapsed = (time.time() - start) * 1000

        dim = len(vectors[0])
        hc.check(
            f"Embedding: generates {dim}-dim vector ({elapsed:.0f}ms)",
            dim > 0,
            ""
        )

    except Exception as e:
        hc.check("Embedding: round-trip test", False, str(e))


def check_llm(hc: HealthCheck):
    """Verify LLM is reachable (quick test)."""
    try:
        from utils.models import LLM, get_llm

        llm = get_llm(LLM.OPENAI)
        start = time.time()
        response = llm.invoke("Say 'ok'")
        elapsed = (time.time() - start) * 1000

        hc.check(
            f"LLM: responds ({elapsed:.0f}ms)",
            bool(response.content.strip()),
            ""
        )

    except Exception as e:
        hc.check("LLM: connectivity", False, str(e))


def main():
    print("\nRunning loom healthcheck...")
    hc = HealthCheck()

    check_env_vars(hc)
    check_rag_config(hc)
    check_ollama(hc)
    check_vector_db(hc)
    check_data_dir(hc)
    check_rag_module(hc)
    check_embedding_roundtrip(hc)
    check_llm(hc)

    all_pass = hc.report()
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
