"""Layer 2: Safety — budget limits and dangerous tool gating."""

import time

from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langgraph.config import get_config


class BudgetError(Exception):
    """Raised when model call budget is exceeded."""
    pass


DANGEROUS_TOOLS = {"send_email", "delete", "execute"}

MAX_MODEL_CALLS = 15


@wrap_model_call
def budget_guard(request, handler):
    """Limit max model calls per run. Raise BudgetError when exceeded."""
    config = get_config()
    call_count = config.get("configurable", {}).get("_model_call_count", 0) + 1

    if call_count > MAX_MODEL_CALLS:
        raise BudgetError(
            f"Budget exceeded: {MAX_MODEL_CALLS} model calls per run"
        )

    return handler(request)


@wrap_tool_call
def safety_check(request, handler):
    """Audit-log dangerous tools before execution."""
    tool_name = request.tool_call["name"]

    if tool_name in DANGEROUS_TOOLS:
        print(f"  [SAFETY] ⚠️  Dangerous tool: {tool_name} — audit logged")

    return handler(request)
