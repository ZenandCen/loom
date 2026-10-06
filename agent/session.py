"""Redis-backed session state management.

Replaces in-memory dicts (_active_session, _pending_messages, _rag_scope_cache)
with persistent Redis keys that survive server restarts.

Redis key layout:
    loom:session:{user_id}      → thread_id (string, no TTL)
    loom:scope:{user_id}        → JSON list of confirmed RAG scope (24h TTL)
    loom:pending:{user_id}      → original message for HITL (5min TTL)
"""

import json
import logging
import os
import time

import redis

logger = logging.getLogger(__name__)


class SessionManager:
    """Redis-backed session state. Survives restarts, supports TTL."""

    def __init__(self, url: str | None = None):
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._r: redis.Redis = None
        self._available = False
        try:
            self._r = redis.Redis.from_url(
                self.url, decode_responses=True, socket_connect_timeout=2
            )
            self._r.ping()
            self._available = True
            logger.info(f"[SESSION] Redis connected: {self.url}")
        except Exception as e:
            logger.warning(f"[SESSION] Redis unavailable ({e}), using in-memory fallback")

    # ── Fallback in-memory store (when Redis is down) ──
    _mem_session: dict[str, str] = {}
    _mem_scope: dict[str, tuple[list, float]] = {}
    _mem_pending: dict[str, tuple[str, float]] = {}
    _mem_attach: dict[str, dict[str, tuple[str, float]]] = {}

    # ── Session (thread_id) ──

    def get_session(self, user_id: str) -> str | None:
        """Get the user's current session (thread_id). Returns None if no active session."""
        if self._available:
            val = self._r.get(f"loom:session:{user_id}")
            return val
        return self._mem_session.get(user_id)

    def set_session(self, user_id: str, thread_id: str) -> None:
        """Set the user's current session."""
        if self._available:
            self._r.set(f"loom:session:{user_id}", thread_id)
        else:
            self._mem_session[user_id] = thread_id
        logger.debug(f"session:{user_id} = {thread_id[:20]}")

    def clear_session(self, user_id: str) -> None:
        """Clear the user's current session."""
        if self._available:
            self._r.delete(f"loom:session:{user_id}")
        else:
            self._mem_session.pop(user_id, None)

    def list_sessions(self, user_id: str) -> list[str]:
        """List all thread_ids associated with a user (for !sessions command).
        Falls back to checkpoint table query if needed.
        """
        # Sessions are stored in checkpoints table; this is just the "active" pointer.
        # For listing, we query checkpoints directly (handled by slack.py).
        active = self.get_session(user_id)
        return [active] if active else []

    # ── RAG Scope Cache ──

    def get_scope(self, user_id: str) -> list[str] | None:
        """Get confirmed RAG scope for user. Returns None if not set or expired."""
        if self._available:
            val = self._r.get(f"loom:scope:{user_id}")
            if val:
                return json.loads(val)
            return None
        # In-memory with TTL
        entry = self._mem_scope.get(user_id)
        if entry:
            scope, expires = entry
            if time.time() < expires:
                return scope
            del self._mem_scope[user_id]
        return None

    def set_scope(self, user_id: str, scope: list[str], ttl: int = 86400) -> None:
        """Set confirmed RAG scope for user (default 24h TTL)."""
        if self._available:
            self._r.set(f"loom:scope:{user_id}", json.dumps(scope), ex=ttl)
        else:
            self._mem_scope[user_id] = (scope, time.time() + ttl)
        logger.info(f"scope:{user_id} = {scope} (ttl={ttl}s)")

    def clear_scope(self, user_id: str) -> None:
        """Clear RAG scope cache for user."""
        if self._available:
            self._r.delete(f"loom:scope:{user_id}")
        else:
            self._mem_scope.pop(user_id, None)

    # ── Pending (HITL) ──

    def get_pending(self, user_id: str) -> str | None:
        """Get pending original message for HITL resume. Returns None if none or expired."""
        if self._available:
            val = self._r.get(f"loom:pending:{user_id}")
            return val
        entry = self._mem_pending.get(user_id)
        if entry:
            msg, expires = entry
            if time.time() < expires:
                return msg
            del self._mem_pending[user_id]
        return None

    def set_pending(self, user_id: str, msg: str, ttl: int = 300) -> None:
        """Store original message for HITL resume (default 5min TTL)."""
        if self._available:
            self._r.set(f"loom:pending:{user_id}", msg, ex=ttl)
        else:
            self._mem_pending[user_id] = (msg, time.time() + ttl)

    def clear_pending(self, user_id: str) -> None:
        """Clear pending message."""
        if self._available:
            self._r.delete(f"loom:pending:{user_id}")
        else:
            self._mem_pending.pop(user_id, None)

    # ── Attachments (recent image/file OCR content) ──

    def set_attachment(self, user_id: str, filename: str, content: str, ttl: int = 3600) -> None:
        """Store OCR/vision content for a user's recent attachment (default 1h TTL)."""
        if self._available:
            self._r.set(
                f"loom:attach:{user_id}:{filename}", content, ex=ttl
            )
        else:
            self._mem_attach[user_id] = (
                self._mem_attach.get(user_id, {})
            )
            self._mem_attach[user_id][filename] = (content, time.time() + ttl)
        logger.info(f"attachment stored: user={user_id} file={filename} len={len(content)}")

    def get_attachments(self, user_id: str) -> list[tuple[str, str]]:
        """Get all non-expired attachments for a user. Returns list of (filename, content)."""
        if self._available:
            keys = self._r.keys(f"loom:attach:{user_id}:*")
            results = []
            for key in keys:
                val = self._r.get(key)
                if val:
                    filename = key.replace(f"loom:attach:{user_id}:", "")
                    results.append((filename, val))
            return results
        # In-memory with TTL check
        user_attaches = self._mem_attach.get(user_id, {})
        now = time.time()
        results = []
        for filename, (content, expires) in list(user_attaches.items()):
            if now < expires:
                results.append((filename, content))
            else:
                del user_attaches[filename]
        return results

    def clear_attachments(self, user_id: str) -> None:
        """Clear all attachments for a user (called on !reset)."""
        if self._available:
            keys = self._r.keys(f"loom:attach:{user_id}:*")
            if keys:
                self._r.delete(*keys)
        else:
            self._mem_attach.pop(user_id, None)

    # ── Bulk operations ──

    def reset_user(self, user_id: str) -> None:
        """Clear all state for a user (called on !reset)."""
        self.clear_session(user_id)
        self.clear_scope(user_id)
        self.clear_pending(user_id)
        self.clear_attachments(user_id)
        logger.info(f"reset_user: {user_id}")

    @property
    def available(self) -> bool:
        return self._available


# Global instance (initialized in agent/config.py)
session_mgr: SessionManager | None = None


def init_session_manager(url: str | None = None) -> SessionManager:
    """Initialize the global SessionManager."""
    global session_mgr
    session_mgr = SessionManager(url)
    return session_mgr


def get_session_mgr() -> SessionManager:
    """Get the global SessionManager (lazy init)."""
    global session_mgr
    if session_mgr is None:
        session_mgr = SessionManager()
    return session_mgr
