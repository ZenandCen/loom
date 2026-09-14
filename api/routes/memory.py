"""Memory inspection routes.

Handles:
    GET /api/memory?project=loom&user=dev_zenchung&category=preference&query=...
    GET /api/memory/namespaces
    DELETE /api/memory/{project}/{user}/{category}/{key}
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from memory.store import PostgresStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/memory", tags=["memory"])

_store: PostgresStore | None = None


def init(store: PostgresStore):
    global _store
    _store = store


def _get_store() -> PostgresStore:
    if _store is None:
        raise HTTPException(500, "Memory store not initialized")
    return _store


@router.get("")
def query_memory(
    project: str = Query(..., description="Project name"),
    user: str = Query(..., description="User ID"),
    category: str | None = Query(None, description="Category (preference, sessions, episodic, etc.)"),
    query: str | None = Query(None, description="Search keyword"),
    limit: int = Query(50, le=200, description="Max results"),
):
    """Query memory items for a user in a project.

    Examples:
        GET /api/memory?project=loom&user=dev_zenchung
        GET /api/memory?project=loom&user=dev_zenchung&category=preference
        GET /api/memory?project=loom&user=dev_zenchung&query=vietnamese
    """
    store = _get_store()

    if category:
        namespace = (project, user, category)
    else:
        namespace = (project, user)

    if query:
        items = store.search(namespace, query=query, limit=limit)
    else:
        items = store.search(namespace, limit=limit)

    return {
        "project": project,
        "user": user,
        "category": category,
        "query": query,
        "count": len(items),
        "items": [
            {
                "namespace": list(item.namespace),
                "key": item.key,
                "value": item.value,
                "updated_at": item.updated_at,
            }
            for item in items
        ],
    }


@router.get("/namespaces")
def list_namespaces(
    project: str | None = Query(None),
    user: str | None = Query(None),
    limit: int = Query(100, le=500),
):
    """List all memory namespaces.

    Examples:
        GET /api/memory/namespaces
        GET /api/memory/namespaces?project=loom
        GET /api/memory/namespaces?project=loom&user=dev_zenchung
    """
    store = _get_store()

    prefix = ()
    if project:
        prefix = (project,)
        if user:
            prefix = (project, user)

    namespaces = store.list_namespaces(prefix=prefix, limit=limit)

    return {
        "prefix": list(prefix) if prefix else "all",
        "count": len(namespaces),
        "namespaces": [list(ns) for ns in namespaces],
    }


@router.get("/stats")
def memory_stats():
    """Get overall memory statistics."""
    store = _get_store()
    total = store.count()
    namespaces = store.list_namespaces(limit=500)

    # Group by project (first element of namespace)
    by_project: dict[str, int] = {}
    for ns in namespaces:
        project = ns[0] if ns else "unknown"
        by_project[project] = by_project.get(project, 0) + 1

    return {
        "total_items": total,
        "total_namespaces": len(namespaces),
        "by_project": by_project,
    }


@router.delete("/{project}/{user}/{category}/{key}")
def delete_memory(project: str, user: str, category: str, key: str):
    """Delete a specific memory item."""
    store = _get_store()
    namespace = (project, user, category)

    item = store.get(namespace, key)
    if not item:
        raise HTTPException(404, f"Memory item not found: {namespace}/{key}")

    store.delete(namespace, key)
    return {
        "status": "ok",
        "deleted": {"namespace": list(namespace), "key": key},
    }
