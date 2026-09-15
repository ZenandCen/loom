"""
Slack Webhook Handler
---------------------
Single entry: POST /slack/webhook
Handles: URL Verification, Signature, and Events (app_mention, message).

Fix for Duplicate Replies:
Slack uses "at least once" delivery. If the AI takes > 3s, Slack assumes a timeout
and retries, sending the same event 3 to 5 times.
Solution:
1. Return 200 OK immediately (using background_tasks).
2. Use event_ts as a lock to process a specific message only once (idempotency).
"""
import os
import re
import time
import hmac
import hashlib
import logging
import httpx
import asyncio

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse

logger = logging.getLogger("slack")
router = APIRouter()

# ── Config ──────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
VERIFY_SIG = os.getenv("SLACK_VERIFY", "true").lower() == "true"

# ── Idempotency Lock ────────────────────────────────────────────
_processed_events: set[str] = set()
_event_lock = asyncio.Lock()

# Resume map: user_id → target thread_id (persistent until !reset or new !resume)
_resume_map: dict[str, str] = {}


def _load_thread_history(thread_id: str) -> list:
    """Load full message history from checkpoint_writes for a given thread."""
    try:
        import psycopg
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        from agent.config import _pg_dsn

        serde = JsonPlusSerializer()
        conn = psycopg.connect(_pg_dsn, autocommit=True)
        cur = conn.cursor()
        cur.execute(
            """SELECT type, blob FROM checkpoint_writes
               WHERE thread_id = %s AND channel = 'messages'
               ORDER BY checkpoint_id ASC, idx ASC""",
            (thread_id,),
        )
        rows = cur.fetchall()
        conn.close()

        messages = []
        for typ, blob in rows:
            data = serde.loads_typed((typ, blob))
            if isinstance(data, list):
                messages.extend(data)
        return messages
    except Exception as e:
        logger.error(f"Failed to load thread history: {e}")
        return []


class SlackBot:
    def __init__(self, token: str):
        self.token = token
        self._bot_user_id: str | None = None

    async def get_bot_user_id(self) -> str:
        """Get the bot's own user ID (cached)."""
        if self._bot_user_id is None:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://slack.com/api/auth.test",
                    headers={"Authorization": f"Bearer {self.token}"},
                )
                data = resp.json()
                self._bot_user_id = data.get("user_id", "")
        return self._bot_user_id

    async def post_message(self, channel: str, text: str, thread_ts: str | None = None):
        payload = {"channel": channel, "text": text[:3000]}
        if thread_ts:
            payload["thread_ts"] = thread_ts

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://slack.com/api/chat.postMessage",
                json=payload,
                headers={"Authorization": f"Bearer {self.token}"},
            )
            return resp.json()

    async def delete_message(self, channel: str, ts: str) -> bool:
        """Delete a specific message by timestamp."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://slack.com/api/chat.delete",
                json={"channel": channel, "ts": ts},
                headers={"Authorization": f"Bearer {self.token}"},
            )
            return resp.json().get("ok", False)

    async def get_thread_messages(self, channel: str, thread_ts: str, limit: int = 100) -> list[dict]:
        """Get all messages in a thread."""
        messages = []
        cursor = ""
        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                params = {"channel": channel, "ts": thread_ts, "limit": str(limit)}
                if cursor:
                    params["cursor"] = cursor
                resp = await client.get(
                    "https://slack.com/api/conversations.history",
                    params=params,
                    headers={"Authorization": f"Bearer {self.token}"},
                )
                data = resp.json()
                messages.extend(data.get("messages", []))
                cursor = data.get("response_metadata", {}).get("next_cursor", "")
                if not cursor or not data.get("has_more"):
                    break
        return messages


slack_bot = SlackBot(BOT_TOKEN)

# ── Lazy init Agent ─────────────────────────────────────────────
_agent = None


async def get_agent():
    global _agent
    if _agent is None:
        from agent.build import build_agent
        _agent = build_agent()
    return _agent


# ── Helpers ─────────────────────────────────────────────────────
def strip_bot_mention(text: str) -> str:
    # Strip Slack internal mention format: <@U12345>
    text = re.sub(r"<@\w+>", "", text)
    # Strip display name mention: @loom
    text = re.sub(r"^@\w+\s*", "", text)
    return text.strip()


def verify_signature(raw_body: str, request: Request) -> bool:
    if not SIGNING_SECRET:
        return True
    sig = request.headers.get("x-slack-signature", "")
    ts = request.headers.get("x-slack-request-timestamp", "")
    if not sig or not ts:
        return False
    try:
        req_ts = int(ts)
    except ValueError:
        return False
    if abs(time.time() - req_ts) > 600:
        return False
    base = f"v0:{req_ts}:{raw_body}"
    expected = "v0=" + hmac.new(
        SIGNING_SECRET.encode(), base.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, sig)


# ── Background Processing ───────────────────────────────────────
async def process_message(
    event_ts: str, user_id: str, channel_id: str, msg: str, thread_ts: str | None
):
    """Runs in background. Checks idempotency before asking Agent."""
    async with _event_lock:
        if event_ts in _processed_events:
            logger.info(f"Skipping Slack retry (duplicate event_ts={event_ts})")
            return
        _processed_events.add(event_ts)
        if len(_processed_events) > 5000:
            for _ in range(1000):
                try:
                    _processed_events.pop()
                except Exception:
                    break

    # Slack thread continuity:
    # - Root message: thread_ts=None, use event_ts (becomes thread_ts for replies)
    # - Reply in thread: thread_ts = root message's event_ts (consistent)
    thread_id = thread_ts or event_ts
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}

    # Command: !reset — clear conversation state for this thread
    if msg.strip().lower() in ("!reset", "!new"):
        try:
            from agent.config import checkpointer
            # Clear any resume binding for this user
            _resume_map.pop(user_id, None)
            checkpointer.delete_thread(thread_id)
            await slack_bot.post_message(
                channel_id, "Conversation reset. Starting fresh.", thread_ts=thread_ts
            )
        except Exception as e:
            await slack_bot.post_message(
                channel_id, f"Reset: {e}", thread_ts=thread_ts
            )
        return

    # Command: !clear — delete all messages (bot + user) from this thread
    if msg.strip().lower() == "!clear":
        try:
            # Get the root thread_ts (use thread_ts if replying, else event_ts)
            root_ts = thread_ts or event_ts
            messages = await slack_bot.get_thread_messages(channel_id, root_ts)
            deleted, failed = 0, 0
            for m in messages:
                if await slack_bot.delete_message(channel_id, m["ts"]):
                    deleted += 1
                else:
                    failed += 1
            msg_text = f"🗑️ Cleared {deleted} messages from this thread."
            if failed:
                msg_text += f" ({failed} could not be deleted)"
            await slack_bot.post_message(
                channel_id, msg_text, thread_ts=thread_ts,
            )
        except Exception as e:
            await slack_bot.post_message(
                channel_id, f"Clear error: {e}", thread_ts=thread_ts
            )
        return

    # Command: !sessions — list all sessions with summaries
    if msg.strip().lower() in ("!sessions", "!list", "!ls"):
        try:
            from agent.config import _pg_dsn
            import psycopg

            conn = psycopg.connect(_pg_dsn, autocommit=True)
            cur = conn.cursor()
            # Get distinct thread_ids (one row per thread, latest checkpoint)
            cur.execute("""
                SELECT DISTINCT thread_id
                FROM checkpoints
            """)
            threads = [r[0] for r in cur.fetchall()]
            # Sort by thread_id (Slack event_ts = chronological)
            threads.sort(reverse=True)
            conn.close()

            if not threads:
                await slack_bot.post_message(channel_id, "No sessions found.", thread_ts=thread_ts)
                return

            import msgpack
            lines = [f"📋 *Sessions* ({len(threads)} total):\n"]
            for i, tid in enumerate(threads[:5], 1):
                # Get first user message as summary from __start__ channel
                summary = ""
                try:
                    conn2 = psycopg.connect(_pg_dsn, autocommit=True)
                    cur2 = conn2.cursor()
                    cur2.execute("""
                        SELECT blob FROM checkpoint_blobs
                        WHERE thread_id = %s AND channel = '__start__'
                        ORDER BY version ASC LIMIT 1
                    """, (tid,))
                    row = cur2.fetchone()
                    conn2.close()

                    if row:
                        data = msgpack.unpackb(row[0], raw=False)
                        if isinstance(data, dict):
                            msgs = data.get("messages", [])
                            for m in msgs:
                                if isinstance(m, dict) and m.get("role") == "user":
                                    content = m.get("content", "")
                                    if isinstance(content, list):
                                        content = " ".join(
                                            b.get("text", "") for b in content if isinstance(b, dict)
                                        )
                                    if content.strip():
                                        summary = content.strip()[:40]
                                        break
                except Exception:
                    pass

                lines.append(f"{i}. `{tid[:14]}` — {summary or '(empty)'}")

            lines.append("\nResume: `!resume <id>`")
            await slack_bot.post_message(channel_id, "\n".join(lines)[:3000], thread_ts=thread_ts)
        except Exception as e:
            await slack_bot.post_message(channel_id, f"Sessions error: {e}", thread_ts=thread_ts)
        return

    # Command: !resume <id> — resume a previous session
    if msg.strip().lower().startswith("!resume"):
        parts = msg.strip().split(maxsplit=1)
        if len(parts) < 2:
            await slack_bot.post_message(
                channel_id, "Usage: `!resume <session_id>` (use !sessions to list)", thread_ts=thread_ts
            )
            return

        session_id = parts[1].strip().strip("`")
        # Find full thread_id matching the prefix
        try:
            from agent.config import _pg_dsn
            import psycopg

            conn = psycopg.connect(_pg_dsn, autocommit=True)
            cur = conn.cursor()
            cur.execute(
                "SELECT thread_id FROM checkpoints WHERE thread_id LIKE %s LIMIT 1",
                (f"{session_id}%",),
            )
            row = cur.fetchone()
            conn.close()

            if not row:
                await slack_bot.post_message(
                    channel_id, f"Session '{session_id}' not found. Use `!sessions` to list.", thread_ts=thread_ts
                )
                return

            full_thread_id = row[0]
            await slack_bot.post_message(
                channel_id,
                f"✅ Resumed session `{full_thread_id[:14]}`. Next message will continue that conversation.",
                thread_ts=thread_ts,
            )
            # Store the resume target by user_id (works across threads)
            _resume_map[user_id] = full_thread_id

        except Exception as e:
            await slack_bot.post_message(channel_id, f"Resume error: {e}", thread_ts=thread_ts)
        return

    # Check if this user has a pending/persistent resume
    if user_id in _resume_map:
        old_thread_id = _resume_map[user_id]
        config["configurable"]["thread_id"] = old_thread_id
        logger.info(f"RESUME: user={user_id} using old session thread_id={old_thread_id}")
        # Load full message history from checkpoint_writes
        history = _load_thread_history(old_thread_id)
        if history:
            logger.info(f"RESUME: loaded {len(history)} messages from history")
            input_messages = history + [{"role": "user", "content": msg}]
        else:
            input_messages = [{"role": "user", "content": msg}]
    else:
        input_messages = [{"role": "user", "content": msg}]

    # === COMMAND ROUTER ===
    # Intercept special commands before team graph
    from agent.tools import CODE_BASE_DIR
    msg_lower = msg.lower().strip()

    # Command: list projects ("!projects" or "list projects" or "danh sách project")
    if msg_lower in ("!projects", "list projects", "danh sách project", "danh sach project"):
        try:
            from pathlib import Path
            projects = []
            for entry in sorted(CODE_BASE_DIR.rglob("*")):
                if entry.is_dir() and not entry.name.startswith("."):
                    if entry.name in ("workspace", "data", "chroma_data", "node_modules", "__pycache__"):
                        continue
                    if any(p in {".venv", "venv", "node_modules", "__pycache__"} for p in entry.parts):
                        continue
                    # Only include dirs that look like projects (have source files)
                    has_code = any(entry.glob("**/*.py")) or any(entry.glob("**/*.ts"))
                    if has_code:
                        rel = str(entry.relative_to(CODE_BASE_DIR))
                        projects.append(rel)
            lines = [f"📁 *Projects* ({len(projects)}):\n"]
            for i, p in enumerate(projects[:20], 1):
                lines.append(f"{i}. `{p}`")
            if len(projects) > 20:
                lines.append(f"... and {len(projects) - 20} more")
            lines.append("\nUse: `project <name>` to select")
            await slack_bot.post_message(channel_id, "\n".join(lines)[:3000], thread_ts=thread_ts)
        except Exception as e:
            await slack_bot.post_message(channel_id, f"List projects error: {e}", thread_ts=thread_ts)
        return

    # Command: set project ("project X" or "set project X")
    if msg_lower.startswith("project ") or msg_lower.startswith("set project "):
        project_path = msg.replace("set ", "", 1) if msg_lower.startswith("set ") else msg
        project_path = project_path.replace("project ", "", 1).strip()
        try:
            from agent.tools import set_project, _get_project_dir
            from rag.code_indexer import SKIP_DIRS, CODE_EXTENSIONS
            from pathlib import Path

            result = set_project.invoke({"path": project_path})

            # If not found directly, try fuzzy search by name
            if result.startswith("Error"):
                matches = []
                for p in CODE_BASE_DIR.rglob("*"):
                    if p.is_dir() and p.name == project_path:
                        if any(skip in p.parts for skip in SKIP_DIRS):
                            continue
                        matches.append(p)
                if len(matches) == 1:
                    rel = str(matches[0].relative_to(CODE_BASE_DIR))
                    set_project.invoke({"path": rel})
                elif len(matches) > 1:
                    rels = [str(m.relative_to(CODE_BASE_DIR)) for m in matches]
                    await slack_bot.post_message(
                        channel_id,
                        f"⚠️ Multiple matches for '{project_path}':\n" +
                        "\n".join(f"{i}. `{r}`" for i, r in enumerate(rels, 1)) +
                        "\n\nUse full path: `project <relative/path>`",
                        thread_ts=thread_ts,
                    )
                    return
                else:
                    # Show available top-level dirs
                    available = sorted([
                        str(e.relative_to(CODE_BASE_DIR))
                        for e in CODE_BASE_DIR.iterdir()
                        if e.is_dir() and not e.name.startswith(".")
                    ])[:15]
                    await slack_bot.post_message(
                        channel_id,
                        f"❌ Project '{project_path}' not found.\n\n"
                        f"Available directories:\n" +
                        "\n".join(f"  - `{a}`" for a in available) +
                        "\n\nUse: `project <relative/path>` or `!projects` to list all",
                        thread_ts=thread_ts,
                    )
                    return

            project_dir = _get_project_dir()
            if str(project_dir) == str(CODE_BASE_DIR):
                await slack_bot.post_message(
                    channel_id, f"❌ Could not resolve project '{project_path}'.", thread_ts=thread_ts
                )
                return
            collection = project_dir.name.lower().replace("-", "_").replace(" ", "_")

            # Scan project for docs + code
            doc_exts = {".pdf", ".md", ".txt", ".html", ".docx", ".csv", ".xlsx", ".xls"}
            n_docs, n_code = 0, 0
            for f in project_dir.rglob("*"):
                if not f.is_file() or f.name.startswith("."):
                    continue
                if any(p in SKIP_DIRS for p in f.parts):
                    continue
                if f.suffix.lower() in doc_exts:
                    n_docs += 1
                elif f.suffix.lower() in CODE_EXTENSIONS:
                    n_code += 1

            # Show relative path if not at top level
            rel_display = ""
            try:
                rel = project_dir.relative_to(CODE_BASE_DIR)
                if str(rel) != rel.name:
                    rel_display = f" (`{rel}`)"
            except ValueError:
                pass

            await slack_bot.post_message(
                channel_id,
                f"✅ Project: **{project_dir.name}**{rel_display}\n"
                f"Collection: `{collection}`\n\n"
                f"📄 Docs: {n_docs} files\n"
                f"💻 Code: {n_code} files\n\n"
                f"Gửi `học tất cả` để index toàn bộ, hoặc:\n"
                f"- `reindex docs` — chỉ tài liệu\n"
                f"- `index code` — chỉ source code",
                thread_ts=thread_ts,
            )
        except Exception as e:
            await slack_bot.post_message(channel_id, f"Set project error: {e}", thread_ts=thread_ts)
        return

    # Command: reindex docs ("reindex docs", "học tài liệu", "index docs")
    if any(kw in msg_lower for kw in ["reindex docs", "học tài liệu", "index docs", "reindex folder docs", "học docs", "index folder docs"]):
        try:
            from agent.tools import _get_project_dir, get_active_collection
            from rag.indexing import load_documents, get_vectorstore
            from rag.chunking import ChunkingConfig, chunk_documents
            from rag.code_indexer import SKIP_DIRS

            project_dir = _get_project_dir()
            if str(project_dir) == str(CODE_BASE_DIR):
                await slack_bot.post_message(channel_id, "❌ Set project trước: `project <path>`", thread_ts=thread_ts)
                return

            await slack_bot.post_message(channel_id, "📚 Đang index tài liệu (toàn bộ project)...", thread_ts=thread_ts)

            import rag.indexing as idx
            orig_ext = idx.SUPPORTED_EXTENSIONS.copy()
            idx.SUPPORTED_EXTENSIONS = orig_ext - {".html"}  # skip .html duplicates

            collection = get_active_collection()
            # Scan entire project (not just docs/)
            docs = load_documents(project_dir)
            chunks = chunk_documents(docs, ChunkingConfig())
            vs = get_vectorstore(collection)
            vs.add_documents(chunks)
            files = set(d.metadata.get("source", "?").split("/")[-1] for d in docs)

            idx.SUPPORTED_EXTENSIONS = orig_ext

            await slack_bot.post_message(
                channel_id,
                f"✅ Đã index **{len(files)} files** → **{len(chunks)} chunks**\n"
                f"Collection: `{collection}`\n\n"
                f"Files: {', '.join(sorted(files)[:10])}{'...' if len(files) > 10 else ''}",
                thread_ts=thread_ts,
            )
        except Exception as e:
            await slack_bot.post_message(channel_id, f"Reindex error: {e}", thread_ts=thread_ts)
        return

    # Command: index code ("index code", "reindex code", "học code", "index python")
    if any(kw in msg_lower for kw in ["index code", "reindex code", "học code", "index python", "reindex python", "index source"]):
        try:
            from agent.tools import _get_project_dir, get_active_collection
            from rag.code_indexer import index_code_directory

            project_dir = _get_project_dir()
            if str(project_dir) == str(CODE_BASE_DIR):
                await slack_bot.post_message(channel_id, "❌ Set project trước: `project <path>`", thread_ts=thread_ts)
                return

            await slack_bot.post_message(channel_id, "💻 Đang index Python files...", thread_ts=thread_ts)

            collection = get_active_collection()
            n = index_code_directory(project_dir, collection)

            await slack_bot.post_message(
                channel_id,
                f"✅ Đã index **{n} code chunks** (AST-based, per function/class)\n"
                f"Collection: `{collection}`\n\n"
                f"Bây giờ có thể hỏi về code, architecture, flow...",
                thread_ts=thread_ts,
            )
        except Exception as e:
            await slack_bot.post_message(channel_id, f"Index code error: {e}", thread_ts=thread_ts)
        return

    # Command: reindex all (docs + code) ("reindex all", "học tất cả", "index all")
    if any(kw in msg_lower for kw in ["reindex all", "học tất cả", "index all", "reindex tất cả", "học project"]):
        try:
            from agent.tools import _get_project_dir, get_active_collection
            from rag.indexing import load_documents, get_vectorstore
            from rag.chunking import ChunkingConfig, chunk_documents
            from rag.code_indexer import index_code_directory

            project_dir = _get_project_dir()
            if str(project_dir) == str(CODE_BASE_DIR):
                await slack_bot.post_message(channel_id, "❌ Set project trước: `project <path>`", thread_ts=thread_ts)
                return

            await slack_bot.post_message(channel_id, "📚 Đang index docs + code (toàn bộ project)...", thread_ts=thread_ts)

            collection = get_active_collection()
            total_chunks = 0

            # Index docs (entire project tree, skip .html)
            import rag.indexing as idx
            orig_ext = idx.SUPPORTED_EXTENSIONS.copy()
            idx.SUPPORTED_EXTENSIONS = orig_ext - {".html"}
            docs = load_documents(project_dir)
            chunks = chunk_documents(docs, ChunkingConfig())
            vs = get_vectorstore(collection)
            vs.add_documents(chunks)
            idx.SUPPORTED_EXTENSIONS = orig_ext
            total_chunks += len(chunks)
            n_docs = len(set(d.metadata.get("source", "?").split("/")[-1] for d in docs))

            # Index code (Python + TypeScript, entire project tree)
            n_code = index_code_directory(project_dir, collection)
            total_chunks += n_code

            await slack_bot.post_message(
                channel_id,
                f"✅ Done!\n"
                f"📄 Docs: {n_docs} files\n"
                f"💻 Code: {n_code} chunks\n"
                f"📊 Total: {total_chunks} chunks → `{collection}`\n\n"
                f"Bây giờ có thể hỏi về project...",
                thread_ts=thread_ts,
            )
        except Exception as e:
            await slack_bot.post_message(channel_id, f"Reindex all error: {e}", thread_ts=thread_ts)
        return

    # === END COMMAND ROUTER ===

    logger.info(f"AI processing... (User={user_id}, Ch={channel_id}, thread_id={config['configurable']['thread_id']})")
    try:
        # Use team graph (Supervisor + parallel workers)
        from agent.build import build_team
        from agent.team.schemas import TeamInput
        from agent.team.synthesizer import build_team_output
        from agent.tools import _get_project_dir, CODE_BASE_DIR

        team = build_team()

        # Build validated input
        project_dir = _get_project_dir()
        project_name = "" if project_dir == CODE_BASE_DIR else project_dir.name
        team_input = TeamInput(
            user_query=msg,
            user_id=user_id,
            project=project_name,
        )

        team_config = {"configurable": {"thread_id": config["configurable"]["thread_id"]}}

        raw_result = await team.ainvoke(
            team_input.model_dump(),
            team_config,
        )

        # Build structured output
        output = build_team_output(raw_result)
        response = output.to_slack_text()

        if not response or response == "I processed your request but have no response to share.":
            response = "I processed your request but have no response to share."

        logger.info(f"AI reply: {len(response)} chars, {len(output.workers)} workers, diagram={output.diagram_type.value}")
        result_slack = await slack_bot.post_message(channel_id, response, thread_ts=thread_ts)
        if result_slack.get("ok"):
            logger.info(f"Message posted to Slack successfully")
        else:
            logger.error(f"Slack API error for channel={channel_id}: {result_slack}")

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        err_msg = str(e)
        if "recursion limit" in err_msg.lower():
            reply = (
                "I got stuck in a loop trying to process that. "
                "Try rephrasing your request, or send `!reset` to start fresh."
            )
        else:
            reply = f"Sorry, I encountered an error: {err_msg[:500]}"
        try:
            await slack_bot.post_message(channel_id, reply, thread_ts=thread_ts)
        except Exception:
            pass


# ── Webhook Handlers ────────────────────────────────────────────
@router.get("/slack/webhook")
@router.get("/slack/webhook/")
async def webhook_get():
    return {"status": "ok", "note": "Slack webhook accepts POST only"}


@router.post("/slack/webhook")
@router.post("/slack/webhook/")
async def webhook_post(request: Request, background_tasks: BackgroundTasks):
    raw_body = (await request.body()).decode("utf-8")
    body = await request.json()
    event_type = body.get("type", "")
    event = body.get("event", {})

    # Challenge verification
    if event_type == "url_verification":
        return PlainTextResponse(body.get("challenge", ""))

    # Signature verification
    if VERIFY_SIG and not verify_signature(raw_body, request):
        logger.error("Signature verification FAILED")
        raise HTTPException(status_code=401, detail="Invalid Slack signature")

    # Handle Events
    if event_type == "event_callback":
        subtype = event.get("type", "")
        bot_id = event.get("bot_id")

        if bot_id:
            logger.info(f"Skipped bot message (bot_id={bot_id})")

        elif subtype in ("app_mention", "message"):
            user_id = event.get("user", "unknown")
            channel_id = event.get("channel", "unknown")
            raw_text = event.get("text", "")
            thread_ts = event.get("thread_ts")
            event_ts = event.get("event_ts")

            msg = strip_bot_mention(raw_text)
            logger.info(
                f"Received event={subtype} | User={user_id} | Ch={channel_id} | Msg='{msg}'"
            )

            if msg and event_ts:
                background_tasks.add_task(
                    process_message, event_ts, user_id, channel_id, msg, thread_ts
                )
        else:
            logger.info(f"Ignoring event subtype: {subtype}")

    return PlainTextResponse("")
