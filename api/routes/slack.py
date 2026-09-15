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
    return re.sub(r"<@\w+>", "", text).strip()


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
    if msg.strip().lower() in ("!reset", "!new", "!clear"):
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

    logger.info(f"AI processing... (User={user_id}, Ch={channel_id}, thread_id={config['configurable']['thread_id']})")
    try:
        agent = await get_agent()

        # Middleware only supports sync — run in thread executor
        result = await asyncio.to_thread(
            agent.invoke,
            {"messages": input_messages},
            config,
        )

        # Extract final AI message
        response = ""
        for message in reversed(result.get("messages", [])):
            if hasattr(message, "type") and message.type == "ai" and message.content:
                response = message.content if isinstance(message.content, str) else str(message.content)
                break
            elif isinstance(message, dict) and message.get("role") == "assistant" and message.get("content"):
                response = message["content"]
                break

        if not response:
            response = "I processed your request but have no response to share."

        logger.info(f"AI reply generated ({len(response)} chars). Posting to Slack...")
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
