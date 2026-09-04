"""Layer 3: Observability — logging and tracing (non-blocking)."""

import time

from langchain.agents.middleware import (
    wrap_model_call,
    wrap_tool_call,
    before_agent,
    after_agent,
)
from langgraph.config import get_config


@before_agent
def log_session_start(state, runtime):
    """Log when agent session starts."""
    config = get_config()
    thread_id = config.get("configurable", {}).get("thread_id", "?")
    print(f"  [SESSION] ▶️  Started | thread={thread_id[:8]}...")
    return {}


@after_agent
def log_session_end(state, runtime):
    """Log when agent session completes."""
    n_msgs = len(state.get("messages", []))
    print(f"  [SESSION] ✅ Done | messages={n_msgs}")
    return {}


@wrap_model_call
def log_model_calls(request, handler):
    """Log model calls: prompt size, tools available, message count."""
    sys_msg = request.system_message
    prompt_chars = 0
    if sys_msg and hasattr(sys_msg, "content_blocks"):
        prompt_chars = sum(
            len(b.get("text", "")) for b in sys_msg.content_blocks if isinstance(b, dict)
        )
    tool_names = [t.name for t in request.tools] if request.tools else []
    n_msgs = len(request.messages)
    print(f"  [MODEL] prompt={prompt_chars:,}ch tools={tool_names} msgs={n_msgs}")
    return handler(request)


@wrap_tool_call
def log_tool_calls(request, handler):
    """Log every tool call with timing."""
    name = request.tool_call["name"]
    args = request.tool_call["args"]
    t0 = time.time()
    result = handler(request)
    elapsed = time.time() - t0
    print(f"  [TOOL] {name}({str(args)[:60]}) → {elapsed:.2f}s")
    return result
