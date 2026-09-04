"""loom — Entry point.

Usage:
    python main.py              # interactive mode (REPL)
    python main.py "question"   # one-shot + interactive
"""

import json
import sys
import unicodedata

from langgraph.types import Command
from langsmith import uuid7

from agent.build import build_agent
from utils.models import LLM


def sanitize(text: str) -> str:
    """Normalize unicode to fix Vietnamese diacritics encoding issues."""
    return unicodedata.normalize("NFC", text)


def handle_interrupt(agent, result, config):
    """Handle HITL interrupt and resume."""
    if not result.get("__interrupt__"):
        return result

    interrupt_info = result["__interrupt__"][0].value
    print(f"\n  [HITL] Approval needed:")
    for action in interrupt_info.get("action_requests", []):
        print(f"    Tool: {action['name']}")
        print(f"    Args: {json.dumps(action['args'], indent=6)}")

    decision = input("  Approve? [approve/edit/reject]: ").strip().lower()
    decision_type = "approve" if decision == "approve" else "reject"
    return agent.invoke(
        Command(resume={"decisions": [{"type": decision_type}]}),
        config=config,
    )


def run_turn(agent, query, config):
    """Run one agent turn."""
    query = sanitize(query)
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
        )
    except Exception as e:
        if "surrogates not allowed" in str(e) or "Unicode" in str(type(e)):
            print(f"\n  [ERROR] Unicode issue. Fresh thread...")
            config["configurable"]["thread_id"] = str(uuid7())
            result = agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config=config,
            )
        else:
            raise

    result = handle_interrupt(agent, result, config)

    content = sanitize(result["messages"][-1].content)
    print(f"\n{content}\n")
    return result


def interactive_loop(agent, config):
    """REPL loop — keep asking until exit."""
    print(f"  Interactive mode. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        run_turn(agent, query, config)


def main():
    query = " ".join(sys.argv[1:]) if sys.argv[1:] else None

    agent = build_agent()
    config = {
        "configurable": {
            "thread_id": str(uuid7()),
            "user_id": "alice",
        }
    }

    print(f"  thread={config['configurable']['thread_id'][:8]}...  user=alice  llm={LLM.OPENAI.value}")

    if query:
        print(f"\nYou > {query}\n")
        run_turn(agent, query, config)
        interactive_loop(agent, config)
    else:
        interactive_loop(agent, config)


if __name__ == "__main__":
    main()
