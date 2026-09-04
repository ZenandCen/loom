"""Layer 4: Domain — business-specific middleware."""

import time
import unicodedata

from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import SystemMessage
from langgraph.config import get_config


def _sanitize_text(text: str) -> str:
    """Fix invalid unicode surrogates (e.g. from Vietnamese diacritics)."""
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        return text.encode("utf-8", errors="replace").decode("utf-8")


@wrap_model_call
def sanitize_output(request, handler):
    """Sanitize model response to prevent unicode issues in message history."""
    response = handler(request)
    if hasattr(response, "content") and isinstance(response.content, str):
        response.content = _sanitize_text(response.content)
    return response


@wrap_model_call
def inject_context(request, handler):
    """Inject dynamic context (user, date, environment) into system prompt."""
    config = get_config()
    user_id = config.get("configurable", {}).get("user_id", "anonymous")
    date_str = time.strftime("%Y-%m-%d %H:%M")

    context_block = (
        f"\n## Runtime Context\n"
        f"- User: {user_id}\n"
        f"- Date: {date_str}\n"
    )

    if request.system_message and hasattr(request.system_message, "content_blocks"):
        blocks = list(request.system_message.content_blocks)
        blocks.append({"type": "text", "text": context_block})
        new_sys = SystemMessage(content_blocks=blocks)
        request = request.override(system_message=new_sys)

    return handler(request)
