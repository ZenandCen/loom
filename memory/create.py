"""Create/save memory to store — use as a node in StateGraph wrapper (optional).

In Deep Agents, memory is handled natively via /memories/ path.
This module is for the optional StateGraph wrapper pattern.
"""

from langgraph.config import get_config

from agent.config import store
from utils.models import LLM, get_llm

model = get_llm(LLM.OPENAI)


def create_memory(state: dict) -> dict:
    """Extract and persist key facts from the conversation."""
    config = get_config()
    user_id = config.get("configurable", {}).get("user_id", "default")

    messages = state.get("messages", [])
    if len(messages) < 4:
        return {}

    # Ask LLM to extract memorable facts
    recent = messages[-10:]
    conversation = "\n".join(f"{m.type}: {m.content}" for m in recent)

    response = model.invoke([
        {
            "role": "system",
            "content": (
                "Extract any important facts, preferences, or context worth remembering "
                "long-term. Return as JSON array of strings. Return [] if nothing notable."
            ),
        },
        {"role": "user", "content": conversation},
    ])

    import json
    try:
        facts = json.loads(response.content)
    except (json.JSONDecodeError, ValueError):
        facts = []

    for fact in facts:
        if isinstance(fact, str) and len(fact) > 10:
            store.put(("loom", user_id, "episodic"), fact, {"fact": fact})

    return {"_memories_saved": len(facts)}
