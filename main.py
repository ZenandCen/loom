"""loom — Multi-Agent Supervisor.

Usage:
    python main.py            # REPL (dev/debug)
    python main.py "question" # One-shot (dev)
    python main.py serve      # Production server (API + Agent + MCP)
"""

import json
import os
import sys
import unicodedata

from dotenv import load_dotenv
from langgraph.types import Command
from langsmith import uuid7

load_dotenv()

from agent.build import build_agent
from utils.models import LLM


def sanitize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# ─── REPL (dev) ───────────────────────────────────────────────────────────────


def handle_interrupt(agent, result, config):
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
    query = sanitize(query)
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
        )
    except Exception as e:
        if "surrogates not allowed" in str(e) or "Unicode" in str(type(e)):
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


def repl(query: str | None = None):
    """Interactive REPL (dev/debug)."""
    agent = build_agent()
    config = {
        "configurable": {
            "thread_id": str(uuid7()),
            "user_id": os.getenv("LOOM_DEFAULT_USER", "alice"),
        }
    }
    print(f"  thread={config['configurable']['thread_id'][:8]}... user=alice llm={LLM.OPENAI.value}")

    if query:
        print(f"\nYou > {query}\n")
        run_turn(agent, query, config)
    interactive_loop(agent, config)


# ─── Server (production) ──────────────────────────────────────────────────────


def serve():
    """Start the full server: API + Agent + optional MCP.

    Everything in 1 process. 1 port.
    """
    import uvicorn

    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("API_HOST", "0.0.0.0")

    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │  LOOM — Multi-Agent Supervisor                  │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  API:      http://localhost:{port}                 │")
    print(f"  │  Docs:     http://localhost:{port}/docs             │")
    print(f"  │  Health:   http://localhost:{port}/health           │")
    print(f"  │  Slack:    http://localhost:{port}/slack/webhook    │")
    print(f"  └─────────────────────────────────────────────────┘")

    uvicorn.run("api.server:app", host=host, port=port, reload=False)


# ─── Entry point ──────────────────────────────────────────────────────────────


def main():
    args = sys.argv[1:]

    if args and args[0] == "serve":
        serve()
    else:
        query = " ".join(args) if args else None
        repl(query)


if __name__ == "__main__":
    main()
