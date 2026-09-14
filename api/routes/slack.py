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

    logger.info(f"AI processing... (User={user_id}, Ch={channel_id})")
    try:
        agent = await get_agent()
        # Use Slack thread as thread_id for conversation continuity
        thread_id = thread_ts or f"slack-{channel_id}-{user_id}"
        config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}

        # Middleware only supports sync — run in thread executor
        result = await asyncio.to_thread(
            agent.invoke,
            {"messages": [{"role": "user", "content": msg}]},
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
        try:
            await slack_bot.post_message(
                channel_id, f"Sorry, I encountered an error: {e}", thread_ts=thread_ts
            )
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
