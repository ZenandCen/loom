"""Load memory from store — use as a node in StateGraph wrapper (optional).

Uses hybrid search (vector + keyword) for semantic recall.
"""

from langgraph.config import get_config

from agent.config import store


def load_memory(state: dict) -> dict:
    """Load relevant memories for the current user into state (semantic search)."""
    config = get_config()
    user_id = config.get("configurable", {}).get("user_id", "default")

    # Load user profile
    profile = store.get(("loom", user_id, "profile"), "current")
    if profile:
        state["user_profile"] = profile.value

    # Semantic search for relevant context (vector + keyword hybrid)
    query = ""
    if state.get("messages"):
        last_msg = state["messages"][-1]
        query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    if query:
        context_items = store.search(("loom", user_id), query=query[:500], limit=5)
        state["relevant_memories"] = [item.value.get("fact", str(item.value)) for item in context_items]
    else:
        state["relevant_memories"] = []

    return state
