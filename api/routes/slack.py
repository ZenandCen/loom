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

# Discovered DB connections from project config scan (for `db 1`, `db 2` shorthand)
_discovered_connections: list[dict] = []  # [{"label": "uri.source", "type": "mysql", "dsn": "mysql://..."}]
# Discovered databases per connection number (for `db <conn> <db_num>`)
_discovered_databases: dict[int, list[str]] = {}  # {3: ["db1", "db2", ...]}


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


# ── Config Scanner ────────────────────────────────────────────────────────────

# Patterns to detect connection strings
_CONN_PATTERNS = {
    "postgresql": re.compile(r"postgresql(\+[\w-]+)?://"),
    "mysql": re.compile(r"mysql(\+[\w-]+)?://"),
    "mongodb": re.compile(r"mongodb(\+srv)?://"),
    "redis": re.compile(r"rediss?://"),
    "amqp": re.compile(r"amqps?://"),
    "kafka": re.compile(r"kafka://|KAFKA_"),
    "elasticsearch": re.compile(r"elasticsearch://|ELASTICSEARCH_"),
    "s3/minio": re.compile(r"https?://.*\.(s3|storage|minio)|S3_|MINIO_"),
}

# Variable name → human-readable purpose (fallback when no comment)
_VAR_PURPOSES = {
    "DEBUG": "Bật/tắt debug mode",
    "SECRET_KEY": "Khóa bí mật cho session/token",
    "SECRET": "Khóa bí mật",
    "API_KEY": "API key cho service bên thứ 3",
    "API_URL": "URL API endpoint",
    "API_BASE": "Base URL cho API",
    "DOMAIN": "Domain chính của app",
    "ALLOWED_HOSTS": "Các host được phép truy cập",
    "ENV": "Môi trường (dev/staging/prod)",
    "ENVIRONMENT": "Môi trường (dev/staging/prod)",
    "STAGE": "Stage deployment",
    "LOG_LEVEL": "Cấp độ log",
    "LOGGING": "Cấu hình logging",
    "TIMEOUT": "Timeout (giây)",
    "PORT": "Cổng mạng",
    "HOST": "Hostname/IP",
    "URL": "URL endpoint",
    "BASE_URL": "Base URL",
    "TOKEN": "Token xác thực",
    "PASSWORD": "Mật khẩu",
    "USERNAME": "Tên user",
    "USER": "Tên user",
    "DB": "Database config",
    "DATABASE": "Database config",
    "REDIS": "Redis cache",
    "CACHE": "Cache config",
    "QUEUE": "Message queue",
    "CELERY": "Celery task queue",
    "RABBITMQ": "RabbitMQ broker",
    "SMTP": "Email SMTP server",
    "MAIL": "Email config",
    "AWS": "AWS credentials",
    "S3": "S3/MinIO storage",
    "GITHUB": "GitHub integration",
    "GITLAB": "GitLab integration",
    "JIRA": "Jira integration",
    "SLACK": "Slack integration",
    "GOCARDLESS": "Payment processing",
    "PAYMENT": "Payment config",
    "FRONTEND": "Frontend URL/config",
    "BACKEND": "Backend URL/config",
    "STATIC": "Static files config",
    "MEDIA": "Media files config",
    "CORS": "CORS allowed origins",
    "RATE_LIMIT": "Rate limiting",
    "JWT": "JWT auth config",
    "OAUTH": "OAuth2 config",
    "SSO": "Single Sign-On config",
    "LDAP": "LDAP directory",
    "SENTRY": "Error tracking",
    "STATSD": "Metrics",
    "PROMETHEUS": "Prometheus metrics",
    "ELASTIC": "Elasticsearch",
    "ES_": "Elasticsearch",
    "INDEX": "Index name",
    "MODEL": "ML model config",
    "NLP": "NLP service",
    "OCR": "OCR service",
    "VISION": "AI vision service",
}


def _scan_project_config(project_dir: "Path") -> dict:
    """Scan project for config files (.env, .properties, settings.py, etc.) and extract variables.

    Returns:
        {
            "connections": "formatted string of DB connections",
            "vars": "formatted string of other config vars",
        }
    """
    from pathlib import Path

    project_dir = Path(project_dir)
    connections: list[str] = []
    connections_structured: list[dict] = []  # [{"label": "uri.source", "type": "mysql", "dsn": "mysql://..."}]
    other_vars: list[tuple[str, str, str]] = []  # (name, value, purpose)
    scanned_files: list[str] = []

    # Find all config files (max depth 3)
    config_files = []
    # .env files
    for f in project_dir.rglob(".env*"):
        if not f.is_file():
            continue
        rel = f.relative_to(project_dir)
        if len(rel.parts) > 3:
            continue
        config_files.append((f, "env"))
    # .properties files
    for f in project_dir.rglob("*.properties"):
        if not f.is_file():
            continue
        rel = f.relative_to(project_dir)
        if len(rel.parts) > 3:
            continue
        config_files.append((f, "ini"))
    # Python settings/config
    for name in ["settings.py", "config.py", "settings/local.py", "settings/base.py"]:
        p = project_dir / name
        if p.exists() and p.is_file():
            config_files.append((p, "python"))
    # docker-compose
    for name in ["docker-compose.yml", "docker-compose.yaml"]:
        p = project_dir / name
        if p.exists() and p.is_file():
            config_files.append((p, "yaml"))

    for config_file, file_type in config_files[:8]:  # Limit to 8 files
        rel_name = str(config_file.relative_to(project_dir))
        scanned_files.append(rel_name)
        try:
            content = config_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if file_type == "env":
            pairs = _parse_env(content)
        elif file_type == "ini":
            pairs = _parse_ini(content)
        elif file_type == "python":
            pairs = _parse_python_config(content)
        else:
            pairs = []

        for key, value in pairs:
            if not key or not value:
                continue

            # Check if it's a connection string
            is_conn = False
            conn_type = ""
            for ctype, pattern in _CONN_PATTERNS.items():
                if pattern.search(value) or pattern.search(key):
                    conn_type = ctype
                    is_conn = True
                    break

            if is_conn:
                # Resolve {password} placeholders from nearby vars
                resolved = _resolve_placeholders(value, pairs, context_key=key)
                # Skip if still has unresolved placeholders
                if "{" in resolved and "}" in resolved:
                    continue
                # Mask password in display (handle @ in password by finding last @ before host)
                masked = _mask_dsn_password(resolved)
                num = len(connections_structured) + 1
                connections.append(f"🔗 **{num}. {key}** ({conn_type}):\n   `{masked}`\n   → `db {num}`")
                connections_structured.append({"label": key, "type": conn_type, "dsn": resolved})
            else:
                # Skip very long values (SQL, multi-line)
                if len(value) > 200:
                    continue
                # Skip values that are clearly SQL or code
                if value.strip().startswith(("SELECT", "INSERT", "UPDATE", "CREATE", "ALTER", "DROP", "IFNULL")):
                    continue
                # Skip keys that are clearly SQL fragments
                if any(ch in key for ch in ["(", ")", "<", ">", " "]):
                    continue
                purpose = _infer_purpose(key)
                # Mask secrets
                if any(s in key.upper() for s in ["PASSWORD", "SECRET", "TOKEN", "KEY", "PASS"]):
                    display_val = value[:4] + "••••" if len(value) > 4 else "••••"
                else:
                    display_val = value if len(value) <= 80 else value[:77] + "..."
                other_vars.append((key, display_val, purpose))

    # Format connections section
    conn_section = ""
    if connections:
        conn_section = "*🔗 Database/Service Connections:*\n" + "\n".join(connections[:6])
        if len(connections) > 6:
            conn_section += f"\n... +{len(connections) - 6} more"

    # Format vars section (limit to 20 for Slack)
    vars_section = ""
    if other_vars:
        vars_section = "*⚙️ Config Variables:*\n"
        for key, val, purpose in other_vars[:18]:
            vars_section += f"`{key}` = `{val}`\n   _{purpose}_\n"
        if len(other_vars) > 18:
            vars_section += f"... +{len(other_vars) - 18} more\n"

    # Add scanned files info
    files_note = ""
    if scanned_files:
        files_note = f"_Scanned: {', '.join(scanned_files[:4])}_" + (f" +{len(scanned_files)-4} more" if len(scanned_files) > 4 else "")

    return {"connections": conn_section, "vars": vars_section, "files": files_note, "connections_structured": connections_structured}


def _parse_env(content: str) -> list[tuple[str, str]]:
    """Parse .env file format (KEY=VALUE)."""
    pairs = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if " #" in line:
            line = line.split(" #")[0].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key.startswith("export "):
            key = key[7:].strip()
        if key and value:
            pairs.append((key, value))
    return pairs


def _parse_ini(content: str) -> list[tuple[str, str]]:
    """Parse .properties / INI file format with sections."""
    pairs = []
    current_section = ""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith(";"):
            continue
        # Section header
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1]
            continue
        # Key=value (may be indented)
        if "=" in stripped:
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key or not value:
                continue
            # Prefix with section if exists
            full_key = f"{current_section}.{key}" if current_section else key
            pairs.append((full_key, value))
    return pairs


def _parse_python_config(content: str) -> list[tuple[str, str]]:
    """Parse Python settings file for simple assignments (os.getenv or direct)."""
    pairs = []
    for line in content.splitlines():
        stripped = line.strip()
        # Match: VAR = "value" or VAR = os.getenv("VAR", "default")
        m = re.match(r'^([A-Z_][A-Z0-9_]*)\s*=\s*(.+)$', stripped)
        if m:
            key = m.group(1)
            value = m.group(2).strip()
            # Try to extract string value
            str_match = re.match(r'^["\'](.+?)["\']$', value)
            if str_match:
                pairs.append((key, str_match.group(1)))
            elif "getenv" in value or "environ" in value:
                # os.getenv("KEY", "default") → extract default
                default_match = re.search(r',\s*["\'](.+?)["\']\s*\)', value)
                if default_match:
                    pairs.append((key, default_match.group(1)))
    return pairs


def _parse_dsn(dsn: str) -> dict:
    """Parse a DSN URI into components. Returns dict with scheme, user, password, host, port, dbname."""
    from urllib.parse import unquote
    m = re.match(r"(\w[\w+]*?)://([^:]+):(.+)@([^@/:]+)(?::(\d+))?(/(\w+))?(?:\?.*)?$", dsn)
    if not m:
        return {}
    return {
        "scheme": m.group(1),
        "user": m.group(2),
        "password": unquote(m.group(3)),
        "host": m.group(4),
        "port": m.group(5) or "",
        "dbname": m.group(7) or "",
    }


def _build_dsn(parts: dict, encode: bool = True, strip_driver: bool = True) -> str:
    """Build a DSN URI from parsed components.

    encode: URL-encode the password
    strip_driver: remove +driver suffix (postgresql+psycopg2 → postgresql)
    """
    from urllib.parse import quote
    pw = quote(parts["password"], safe="") if encode else parts["password"]
    port = f":{parts['port']}" if parts.get("port") else ""
    db = f"/{parts['dbname']}" if parts.get("dbname") else ""
    scheme = parts["scheme"]
    if strip_driver:
        scheme = scheme.split("+")[0]
    return f"{scheme}://{parts['user']}:{pw}@{parts['host']}{port}{db}"


def _mask_dsn_password(dsn: str) -> str:
    """Mask the password in a DSN string. Handles passwords containing @ or :."""
    m = re.match(r"(\w[\w+]*://[^:]+):(.+)@([^@]+)$", dsn)
    if m:
        return f"{m.group(1)}:••••@{m.group(3)}"
    return dsn


def _encode_dsn_password(dsn: str) -> str:
    """URL-encode the password in a DSN to handle special chars (@, :, /, etc.)."""
    from urllib.parse import quote
    m = re.match(r"(\w[\w+]*://[^:]+):(.+)@([^@]+)$", dsn)
    if m:
        encoded_pw = quote(m.group(2), safe="")
        return f"{m.group(1)}:{encoded_pw}@{m.group(3)}"
    return dsn


def _resolve_placeholders(value: str, all_pairs: list[tuple[str, str]], context_key: str = "") -> str:
    """Resolve {placeholder} and *** references in connection strings using nearby config values.

    context_key: the key of the connection string itself (e.g. "uri.source")
    Used to resolve ambiguous placeholders like {password} → source_password vs destination_password
    """
    lookup_exact = {}
    for k, v in all_pairs:
        lookup_exact[k] = v

    # Determine prefix from context key (e.g. "uri.raw_destination" → extract "destination")
    prefix = context_key.lower() if context_key else ""
    # Also try last segment of context key for matching
    last_segment = prefix.split(".")[-1] if "." in prefix else prefix
    # Clean segment: strip raw_/main_/primary_ prefix
    clean_segment = re.sub(r'^(raw_|main_|primary_)', '', last_segment) if last_segment else ""

    def _find_value(placeholder: str) -> str:
        """Find a config value for a given placeholder name."""
        # Strategy 1: exact match with full prefix
        if prefix:
            candidate = f"{prefix}_{placeholder}"
            if candidate in lookup_exact:
                return lookup_exact[candidate]
            candidate2 = f"{prefix}.{placeholder}"
            if candidate2 in lookup_exact:
                return lookup_exact[candidate2]
        # Strategy 2: match by clean segment
        if clean_segment:
            candidate = f"{clean_segment}_{placeholder}"
            if candidate in lookup_exact:
                return lookup_exact[candidate]
            candidate2 = f"uri.{clean_segment}_{placeholder}"
            if candidate2 in lookup_exact:
                return lookup_exact[candidate2]
        # Strategy 3: any key containing both segment and placeholder
        if clean_segment:
            for k, v in all_pairs:
                kl = k.lower()
                if clean_segment in kl and placeholder in kl:
                    return v
        # Strategy 4: fallback to first match
        for k, v in all_pairs:
            if k.lower().endswith(f"_{placeholder}") or k.lower().endswith(f".{placeholder}"):
                return v
        return None

    # Resolve {variable} placeholders
    def replace_braced(match):
        placeholder = match.group(1).lower()
        result = _find_value(placeholder)
        return result if result else match.group(0)

    value = re.sub(r'\{(\w+)\}', replace_braced, value)

    # Resolve *** placeholders (username and host positions in DSN)
    # Pattern: scheme://***:pass@***:port/db
    if "***" in value:
        m = re.match(r"(\w[\w+]*?)://(\*\*\*):(.+)@(\*\*\*)(?::(\d+))?(/(\w+))?(?:\?.*)?$", value)
        if m:
            # Resolve username
            username = _find_value("username") or _find_value("user")
            host = _find_value("host")
            if username:
                value = value.replace("://***", f"://{username}", 1)
            if host:
                # Replace the *** after @ (the host part)
                value = re.sub(r"@(\*\*\*)", f"@{host}", value, count=1)

    return value


def _infer_purpose(var_name: str) -> str:
    """Infer the purpose of a config variable from its name."""
    upper = var_name.upper()
    # Exact match first
    if upper in _VAR_PURPOSES:
        return _VAR_PURPOSES[upper]
    # Prefix match
    for prefix, purpose in _VAR_PURPOSES.items():
        if upper.startswith(prefix):
            # Strip prefix to get the specific service
            suffix = var_name[len(prefix):].strip("_-")
            if suffix:
                return f"{purpose}: {suffix.lower()}"
            return purpose
    # Generic fallback based on value pattern
    return "Config"


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
        """Post a message, smart-splitting at section boundaries if > 4000 chars."""
        chunks = _smart_split_slack(text, limit=4000)

        results = []
        async with httpx.AsyncClient(timeout=30) as client:
            for i, chunk in enumerate(chunks):
                payload = {"channel": channel, "text": chunk}
                if thread_ts:
                    payload["thread_ts"] = thread_ts
                resp = await client.post(
                    "https://slack.com/api/chat.postMessage",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.token}"},
                )
                results.append(resp.json())
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.3)
        return results[0] if results else {}

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


def _smart_split_slack(text: str, limit: int = 4000) -> list[str]:
    """Split text into chunks ≤ limit, respecting code blocks and section boundaries.

    Splitting priority (highest first):
        1. Never split inside a code block (```...```)
        2. Markdown header (##, ###, etc.)
        3. Blank line (paragraph boundary)
        4. Any newline
        5. Hard cut (last resort)

    If a single code block exceeds the limit, it becomes its own chunk.
    """
    if len(text) <= limit:
        return [text]

    # Step 1: Parse into atomic segments (code blocks are atomic)
    segments: list[tuple[str, bool]] = []  # (text, is_code_block)
    in_code = False
    buf = ""
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            if in_code:
                buf += "\n" + line
                segments.append((buf, True))
                buf = ""
                in_code = False
            else:
                if buf.strip():
                    segments.append((buf, False))
                    buf = ""
                buf = line
                in_code = True
        else:
            buf += line + "\n"
    if buf:
        segments.append((buf, in_code))

    # Step 2: Greedily pack segments into chunks
    chunks: list[str] = []
    current = ""

    for seg_text, is_code in segments:
        # If this segment alone exceeds limit, flush current and make it standalone
        if len(seg_text) > limit:
            if current.strip():
                chunks.append(current.rstrip())
                current = ""
            # Code block too big — must hard-split (rare)
            while len(seg_text) > limit:
                chunks.append(seg_text[:limit])
                seg_text = seg_text[limit:]
            if seg_text:
                current = seg_text
            continue

        # Would adding this segment exceed the limit?
        if len(current) + len(seg_text) > limit and current.strip():
            # Need to split — find best point in `current`
            split_at = _find_best_split(current, limit - len(seg_text))
            if split_at > 0:
                chunks.append(current[:split_at].rstrip())
                current = current[split_at:].lstrip("\n") + seg_text
            else:
                chunks.append(current.rstrip())
                current = seg_text
        else:
            current += seg_text

    if current.strip():
        chunks.append(current.rstrip())

    return chunks


def _find_best_split(text: str, max_len: int) -> int:
    """Find the best split point in text, preferring headers > blank lines > newlines."""
    search_area = text[:max_len]

    # 1. Try to split at a markdown header (## or ###)
    for i in range(len(search_area) - 1, 50, -1):
        if search_area[i] == "\n" and search_area[i + 1:i + 4] in ("## ", "###"):
            return i + 1

    # 2. Try blank line (paragraph boundary)
    for i in range(len(search_area) - 1, 50, -1):
        if search_area[i:i + 2] == "\n\n":
            return i + 1

    # 3. Try any newline
    for i in range(len(search_area) - 1, 50, -1):
        if search_area[i] == "\n":
            return i + 1

    # 4. Hard cut
    return max_len


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

    # Command: !clear — delete bot messages from this thread
    if msg.strip().lower() == "!clear":
        try:
            bot_uid = await slack_bot.get_bot_user_id()
            root_ts = thread_ts or event_ts
            messages = await slack_bot.get_thread_messages(channel_id, root_ts)
            # Only delete bot's own messages (works without channel admin)
            bot_messages = [m for m in messages if m.get("user") == bot_uid or m.get("bot_id")]
            deleted, failed = 0, 0
            for i, m in enumerate(bot_messages):
                ok = await slack_bot.delete_message(channel_id, m["ts"])
                if ok:
                    deleted += 1
                else:
                    failed += 1
                if i < len(bot_messages) - 1:
                    await asyncio.sleep(0.12)
            msg_text = f"🗑️ Cleared {deleted} bot messages."
            if failed:
                msg_text += f" ({failed} failed — need scope `chat:delete`)"
            n_user = len(messages) - len(bot_messages)
            if n_user > 0:
                msg_text += f"\n💡 {n_user} user messages remain (select manually to delete)."
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
    if msg_lower in ("!projects", "!project", "list projects", "danh sách project", "danh sach project"):
        try:
            from pathlib import Path
            projects = []
            skip_dirs = {".venv", "venv", "node_modules", "__pycache__", ".git", ".git_modules", "workspace", "data", "chroma_data"}
            # Only scan top 2 levels (fast)
            for entry in sorted(CODE_BASE_DIR.iterdir()):
                if not entry.is_dir() or entry.name.startswith("."):
                    continue
                if entry.name in skip_dirs:
                    continue
                # Top-level dir with code
                try:
                    has_code = any(entry.glob("*.py")) or any(entry.glob("*.ts")) or any(entry.glob("*.go")) or any(entry.glob("*.java"))
                    if not has_code:
                        has_code = any(f.is_file() and f.suffix in (".py", ".ts", ".go", ".java") for f in entry.glob("*/*"))
                    if has_code:
                        projects.append(entry.name)
                except PermissionError:
                    continue
                # Sub-level dirs (e.g., DWH/fpt-dwh-reconcile-svc)
                for sub in sorted(entry.iterdir()):
                    if not sub.is_dir() or sub.name.startswith("."):
                        continue
                    if sub.name in skip_dirs:
                        continue
                    try:
                        has_code = any(sub.glob("*.py")) or any(sub.glob("*.ts")) or any(sub.glob("*.go")) or any(sub.glob("*.java"))
                        if not has_code:
                            has_code = any(f.is_file() and f.suffix in (".py", ".ts", ".go", ".java") for f in sub.glob("*/*"))
                        if has_code:
                            rel = str(sub.relative_to(CODE_BASE_DIR))
                            projects.append(rel)
                    except PermissionError:
                        continue
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

    # Command: list databases on a connection ("list db <number>")
    if msg_lower.startswith("list db "):
        conn_num_str = msg_lower.replace("list db ", "").strip()
        if not conn_num_str.isdigit():
            await slack_bot.post_message(channel_id, "Usage: `list db <number>` (e.g. `list db 3`)", thread_ts=thread_ts)
            return
        conn_num = int(conn_num_str)
        # Ensure connections are loaded
        if not _discovered_connections:
            from agent.tools import _get_project_dir, CODE_BASE_DIR
            project_dir = _get_project_dir()
            if str(project_dir) != str(CODE_BASE_DIR):
                config_info = _scan_project_config(project_dir)
                _discovered_connections.extend(config_info.get("connections_structured", []))
        if not _discovered_connections:
            await slack_bot.post_message(channel_id, "❌ No connections cached. Run `project <path>` first.", thread_ts=thread_ts)
            return
        if conn_num < 1 or conn_num > len(_discovered_connections):
            await slack_bot.post_message(channel_id, f"❌ Connection #{conn_num} not found. Available: 1-{len(_discovered_connections)}", thread_ts=thread_ts)
            return
        try:
            conn = _discovered_connections[conn_num - 1]
            parts = _parse_dsn(conn["dsn"])
            if not parts or not parts.get("host") or "*" in parts.get("host", ""):
                await slack_bot.post_message(
                    channel_id,
                    f"❌ Cannot parse DSN for connection #{conn_num} ({conn['label']}):\n`{conn['dsn'][:50]}...`\n"
                    f"Try: `db {conn_num}` to connect directly",
                    thread_ts=thread_ts,
                )
                return
            host_display = f"{parts['host']}:{parts['port']}" if parts['port'] else parts['host']

            if "mysql" in parts["scheme"] or "mysql" in conn["type"]:
                import pymysql
                admin_conn = pymysql.connect(host=parts["host"], port=int(parts["port"] or 3306), user=parts["user"], password=parts["password"], connect_timeout=5)
                cur = admin_conn.cursor()
                cur.execute("SHOW DATABASES")
                dbs = [row[0] for row in cur.fetchall() if row[0] not in ("information_schema", "mysql", "performance_schema", "sys")]
                admin_conn.close()
            else:
                import psycopg
                # Connect to 'postgres' default DB to list all databases
                admin_dsn = _build_dsn({**parts, "dbname": "postgres"})
                admin_conn = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
                cur = admin_conn.cursor()
                cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname")
                dbs = [row[0] for row in cur.fetchall()]
                admin_conn.close()

            _discovered_databases[conn_num] = dbs
            lines = [f"📋 *Databases on {host_display}* ({len(dbs)} found):\n"]
            for i, db in enumerate(dbs, 1):
                lines.append(f"{i}. `{db}`")
            lines.append(f"\nUse: `db {conn_num} <number>` to connect (e.g. `db {conn_num} 1`)")
            await slack_bot.post_message(channel_id, "\n".join(lines)[:3000], thread_ts=thread_ts)
        except Exception as e:
            await slack_bot.post_message(channel_id, f"❌ List DB error: {e}", thread_ts=thread_ts)
        return

    # Command: set project database ("db <dsn>", "db <number>", "db <conn> <db_num>", or "set db <dsn>")
    if msg_lower.startswith("db ") or msg_lower.startswith("set db "):
        dsn = msg.replace("set ", "", 1) if msg_lower.startswith("set ") else msg
        dsn = dsn.replace("db ", "", 1).strip()
        if not dsn:
            from agent.tools import get_project_db_dsn
            current = get_project_db_dsn()
            if current:
                await slack_bot.post_message(channel_id, f"📊 Current project DB: `{current.split('@')[-1] if '@' in current else current}`", thread_ts=thread_ts)
            elif _discovered_connections:
                lines = ["*📋 Available connections (from project scan):*"]
                for i, c in enumerate(_discovered_connections, 1):
                    lines.append(f"{i}. **{c['label']}** ({c['type']}): `{c['dsn'].split('@')[-1]}`")
                lines.append(f"\nUse: `db <number>` to select")
                await slack_bot.post_message(channel_id, "\n".join(lines), thread_ts=thread_ts)
            else:
                await slack_bot.post_message(
                    channel_id,
                    "Usage: `db postgresql://user:pass@host:port/dbname`\n"
                    "Or: `db <number>` (after `project` scan)\n"
                    "Or: `db` to show current",
                    thread_ts=thread_ts,
                )
            return

        # If dsn is "num num" → db <conn_num> <db_num> (pick database from connection)
        if re.match(r"^\d+\s+\d+$", dsn):
            parts_nums = dsn.split()
            conn_idx = int(parts_nums[0]) - 1
            db_idx = int(parts_nums[1]) - 1
            if not _discovered_connections:
                from agent.tools import _get_project_dir, CODE_BASE_DIR
                project_dir = _get_project_dir()
                if str(project_dir) != str(CODE_BASE_DIR):
                    config_info = _scan_project_config(project_dir)
                    _discovered_connections.extend(config_info.get("connections_structured", []))
            if not _discovered_connections:
                await slack_bot.post_message(channel_id, "❌ No connections found. Run `project <path>` first.", thread_ts=thread_ts)
                return
            if conn_idx < 0 or conn_idx >= len(_discovered_connections):
                await slack_bot.post_message(channel_id, f"❌ Connection #{parts_nums[0]} not found. Available: 1-{len(_discovered_connections)}", thread_ts=thread_ts)
                return
            conn_dsn = _discovered_connections[conn_idx]["dsn"]
            conn_num = int(parts_nums[0])
            # Get databases list (from cache or query)
            if conn_num not in _discovered_databases:
                # Auto-query
                try:
                    cp = _parse_dsn(conn_dsn)
                    if "mysql" in cp.get("scheme", ""):
                        import pymysql
                        mc = pymysql.connect(host=cp["host"], port=int(cp["port"] or 3306), user=cp["user"], password=cp["password"], connect_timeout=5)
                        cur = mc.cursor()
                        cur.execute("SHOW DATABASES")
                        dbs = [r[0] for r in cur.fetchall() if r[0] not in ("information_schema", "mysql", "performance_schema", "sys")]
                        mc.close()
                    else:
                        import psycopg
                        admin_dsn = _build_dsn({**cp, "dbname": "postgres"})
                        ac = psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5)
                        cur = ac.cursor()
                        cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname")
                        dbs = [r[0] for r in cur.fetchall()]
                        ac.close()
                    _discovered_databases[conn_num] = dbs
                except Exception as e:
                    await slack_bot.post_message(channel_id, f"❌ Cannot list databases: {e}", thread_ts=thread_ts)
                    return
            if db_idx < 0 or db_idx >= len(_discovered_databases[conn_num]):
                await slack_bot.post_message(channel_id, f"❌ DB #{parts_nums[1]} out of range. Available: 1-{len(_discovered_databases[conn_num])}", thread_ts=thread_ts)
                return
            # Build new DSN with selected database (no encode — _encode_dsn_password handles it)
            cp = _parse_dsn(conn_dsn)
            cp["dbname"] = _discovered_databases[conn_num][db_idx]
            dsn = _build_dsn(cp, encode=False)
        # If dsn is a single number, pick from discovered connections
        elif dsn.isdigit():
            if not _discovered_connections:
                # Try to rescan from active project
                from agent.tools import _get_project_dir, CODE_BASE_DIR
                project_dir = _get_project_dir()
                if str(project_dir) != str(CODE_BASE_DIR):
                    config_info = _scan_project_config(project_dir)
                    _discovered_connections.extend(config_info.get("connections_structured", []))
            if not _discovered_connections:
                await slack_bot.post_message(
                    channel_id,
                    "❌ No connections found. Run `project <path>` first to scan config.",
                    thread_ts=thread_ts,
                )
                return
            idx = int(dsn) - 1
            if 0 <= idx < len(_discovered_connections):
                dsn = _discovered_connections[idx]["dsn"]
            else:
                await slack_bot.post_message(
                    channel_id,
                    f"❌ Number out of range. Available: 1-{len(_discovered_connections)}",
                    thread_ts=thread_ts,
                )
                return

        try:
            from agent.tools import set_project_db_dsn
            import psycopg
            # URL-encode password to handle special chars
            encoded_dsn = _encode_dsn_password(dsn)
            # Test connection
            conn = psycopg.connect(encoded_dsn, autocommit=True, connect_timeout=5)
            conn.close()
            set_project_db_dsn(encoded_dsn)
            safe_dsn = encoded_dsn.split("@")[-1] if "@" in encoded_dsn else encoded_dsn
            await slack_bot.post_message(
                channel_id,
                f"✅ Project database set: `{safe_dsn}`\n"
                f"Bây giờ có thể hỏi về tables, schema, data...",
                thread_ts=thread_ts,
            )
        except Exception as e:
            await slack_bot.post_message(
                channel_id, f"❌ DB connection failed: {e}", thread_ts=thread_ts
            )
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

            # Scan config files
            config_info = _scan_project_config(project_dir)

            # Store discovered connections for `db <number>` shorthand
            _discovered_connections.clear()
            _discovered_connections.extend(config_info.get("connections_structured", []))

            # Auto-set DB if exactly 1 connection found
            db_auto_msg = ""
            if len(_discovered_connections) == 1:
                try:
                    from agent.tools import set_project_db_dsn
                    import psycopg
                    auto_dsn = _encode_dsn_password(_discovered_connections[0]["dsn"])
                    conn = psycopg.connect(auto_dsn, autocommit=True, connect_timeout=5)
                    conn.close()
                    set_project_db_dsn(auto_dsn)
                    safe = auto_dsn.split("@")[-1] if "@" in auto_dsn else auto_dsn
                    db_auto_msg = f"\n✅ DB auto-set: `{safe}`\n"
                except Exception:
                    pass  # Connection failed, user can use `db 1` later

            # Build response
            resp = (
                f"✅ Project: **{project_dir.name}**{rel_display}\n"
                f"Collection: `{collection}`\n\n"
                f"📄 Docs: {n_docs} files | 💻 Code: {n_code} files\n"
            )

            if config_info.get("files"):
                resp += f"\n{config_info['files']}\n"
            if config_info["connections"]:
                resp += config_info["connections"] + "\n"
            if db_auto_msg:
                resp += db_auto_msg
            elif len(_discovered_connections) > 1:
                resp += f"\n💡 Use `db 1`, `db 2`, ... to set database\n"
            if config_info["vars"]:
                resp += config_info["vars"]

            resp += (
                f"\n\nGửi `học tất cả` để index toàn bộ, hoặc:\n"
                f"- `reindex docs` — chỉ tài liệu\n"
                f"- `index code` — chỉ source code"
            )

            await slack_bot.post_message(channel_id, resp, thread_ts=thread_ts)
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
