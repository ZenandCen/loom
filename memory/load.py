"""Load memory from store — use as a node in StateGraph wrapper (optional).

In Deep Agents, memory is handled natively via /memories/ path.
This module is for the optional StateGraph wrapper pattern.
"""

from langgraph.config import get_config

from agent.config import store


def load_memory(state: dict) -> dict:
    """Load relevant memories for the current user into state."""
    config = get_config()
    user_id = config.get("configurable", {}).get("user_id", "default")

    # Load user profile
    profile = store.get(("loom", user_id, "profile"), "current")
    if profile:
        state["user_profile"] = profile.value

    # Load recent context
    context_items = store.search(("loom", user_id), query=state.get("messages", [])[-1].content if state.get("messages") else "", limit=5)
    state["relevant_memories"] = [item.value["fact"] for item in context_items]

    return state
