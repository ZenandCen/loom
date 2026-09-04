"""Compose: assemble all layers into a single agent.

This is the most important file — where everything gets wired together.
"""

from deepagents import create_deep_agent

from agent.config import build_backend, checkpointer, store
from agent.middleware import (
    budget_guard,
    inject_context,
    log_model_calls,
    log_session_end,
    log_session_start,
    log_tool_calls,
    safety_check,
    sanitize_output,
)
from agent.prompt import MAIN_AGENT_PROMPT
from agent.subagents import all_subagents
from agent.tools import all_tools
from utils.models import LLM, get_llm


def get_middleware_stack() -> list:
    """Return middleware list in execution order (layers 2-4)."""
    return [
        # Layer 2: Safety (gate everything)
        budget_guard,
        safety_check,
        # Layer 3: Observability (log, don't block)
        log_session_start,
        log_model_calls,
        log_tool_calls,
        log_session_end,
        # Layer 4: Domain (business logic)
        inject_context,
        sanitize_output,
    ]


def build_agent(llm: LLM = LLM.OPENAI):
    """Build the main deep agent with all layers.

    Args:
        llm: Which LLM to use. Default: LLM.OPENAI
    """
    model = get_llm(llm)

    agent = create_deep_agent(
        model=model,
        system_prompt=MAIN_AGENT_PROMPT,
        tools=all_tools,
        subagents=all_subagents,
        backend=build_backend(),
        store=store,
        checkpointer=checkpointer,
        middleware=get_middleware_stack(),
        interrupt_on={
            "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
            "delete": {"allowed_decisions": ["approve", "reject"]},
        },
    )

    return agent.with_config({"recursion_limit": 50})
