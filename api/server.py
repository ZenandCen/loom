"""FastAPI application factory.

Run: python main.py serve
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from api.routes import documents, index, memory, slack
from memory.store import PostgresStore
from storage.minio import MinIOStorage

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "./data"))
DATA_DIR.mkdir(exist_ok=True)


def create_app() -> FastAPI:
    app = FastAPI(title="Loom Multi-Agent", version="0.1.0")

    # Health check
    @app.get("/health")
    def health():
        return {"status": "ok", "service": "loom"}

    # Initialize storage
    storage = MinIOStorage()

    # Initialize memory store
    try:
        mem_store = PostgresStore()
        memory.init(mem_store)
    except Exception as e:
        logger.warning(f"Memory store unavailable: {e}")

    # Inject dependencies into route modules
    documents.init(storage=storage, data_dir=DATA_DIR)
    index.init(storage=storage, data_dir=DATA_DIR)

    # Register routers
    app.include_router(documents.router)
    app.include_router(index.router)
    app.include_router(memory.router)
    app.include_router(slack.router)

    return app


app = create_app()
