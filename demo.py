"""Demo tất cả 10 tính năng của Loom.

Usage:
    python demo.py
"""

import json
import time
from langgraph.types import Command
from langsmith import uuid7

from agent.build import build_agent
from agent.config import store
from utils.models import LLM, get_llm


def demo_1_llm_registry():
    print("\n" + "=" * 60)
    print("  1. LLM ENUM + REGISTRY")
    print("=" * 60)
    print(f"  Available LLMs: {[m.value for m in LLM]}")
    for llm in LLM:
        try:
            instance = get_llm(llm)
            print(f"    {llm.value:10s} → {type(instance).__name__}")
        except ValueError as e:
            print(f"    {llm.value:10s} → unavailable ({e})")


def demo_2_langsmith():
    print("\n" + "=" * 60)
    print("  2. LANGSMITH TRACING")
    print("=" * 60)
    import os
    if os.getenv("LANGSMITH_TRACING"):
        print(f"  Tracing: ON")
        print(f"  Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
        print(f"  URL: https://smith.langchain.com/projects/{os.getenv('LANGSMITH_PROJECT', 'default')}")
    else:
        print("  Tracing: OFF (set LANGSMITH_TRACING=true in .env)")


def demo_3_observability(agent, config):
    print("\n" + "=" * 60)
    print("  3. OBSERVABILITY MIDDLEWARE (watch the logs below)")
    print("=" * 60)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Say hello in 3 words"}]},
        config=config,
    )
    print(f"\n  → Response: {result['messages'][-1].content}")
    return config


def demo_4_filesystem(agent, config):
    print("\n" + "=" * 60)
    print("  4. DEEP AGENT + FILESYSTEM")
    print("=" * 60)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Write a file called /workspace/hello.txt with content 'Hello Loom'"}]},
        config=config,
    )
    print(f"\n  → {result['messages'][-1].content[:200]}")

    # Verify file exists
    from agent.config import WORKSPACE_DIR
    path = f"{WORKSPACE_DIR}/hello.txt"
    if __import__('os').path.exists(path):
        print(f"  ✓ File created: {path}")
        print(f"    Content: {open(path).read().strip()}")
    return config


def demo_5_subagent(agent, config):
    print("\n" + "=" * 60)
    print("  5. SUBAGENT DELEGATION")
    print("=" * 60)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Use the researcher subagent to find what year LangGraph was first released"}]},
        config=config,
    )
    print(f"\n  → {result['messages'][-1].content[:300]}")
    return config


def demo_6_memory_remember(agent, config):
    print("\n" + "=" * 60)
    print("  6. MEMORY — REMEMBER")
    print("=" * 60)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Remember that my name is Zen Chung and I am learning LangGraph. Category: preference"}]},
        config=config,
    )
    print(f"\n  → {result['messages'][-1].content[:200]}")
    return config


def demo_7_memory_recall(agent, config):
    print("\n" + "=" * 60)
    print("  7. MEMORY — RECALL (same thread)")
    print("=" * 60)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name? Check your memory."}]},
        config=config,
    )
    print(f"\n  → {result['messages'][-1].content[:200]}")
    return config


def demo_8_hitl(agent, config):
    print("\n" + "=" * 60)
    print("  8. HITL — HUMAN IN THE LOOP (auto-approve)")
    print("=" * 60)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Send an email to boss@company.com with subject 'Weekly Report' and content 'All tasks completed.'"}]},
        config=config,
    )

    if result.get("__interrupt__"):
        interrupt_info = result["__interrupt__"][0].value
        print(f"\n  ⚠️  INTERRUPT triggered!")
        for action in interrupt_info.get("action_requests", []):
            print(f"  Tool: {action['name']}")
            print(f"  Args: {json.dumps(action['args'], indent=4)}")
        print(f"\n  → Auto-approving...")

        result = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config,
        )
        print(f"  → After approval: {result['messages'][-1].content[:200]}")
    else:
        print(f"\n  → {result['messages'][-1].content[:200]}")
    return config


def demo_9_multi_llm():
    print("\n" + "=" * 60)
    print("  9. MULTI-LLM (build different agents)")
    print("=" * 60)
    for llm in LLM:
        try:
            agent = build_agent(llm=llm)
            print(f"  ✓ {llm.value:10s} → agent built successfully")
        except Exception as e:
            print(f"  ✗ {llm.value:10s} → {e}")


def demo_10_safety(agent, config):
    print("\n" + "=" * 60)
    print("  10. SAFETY MIDDLEWARE + HITL (budget + audit + block)")
    print("=" * 60)
    print("  Budget: max 15 model calls per run")
    print("  Audit: dangerous tools logged (send_email, delete, execute)")
    print("  Block: HITL interrupt on dangerous tools")
    print("\n  (Watch for [SAFETY] log + INTERRUPT when agent uses delete)")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Delete the file /workspace/hello.txt"}]},
        config=config,
    )

    if result.get("__interrupt__"):
        interrupt_info = result["__interrupt__"][0].value
        print(f"\n  ⚠️  INTERRUPT triggered!")
        for action in interrupt_info.get("action_requests", []):
            print(f"  Tool: {action['name']}")
            print(f"  Args: {json.dumps(action['args'], indent=4)}")
            print(f"\n  → Reject...")

            result = agent.invoke(
                Command(resume={"decisions": [{"type": "reject"}]}),
                config=config,
            )
            print(f"  → After rejection: {result['messages'][-1].content[:200]}")
    else:
        print(f"\n  → {result['messages'][-1].content[:200]}")


def main():
    print("""
    ┌────────────────────────────────────────────────────────────┐
    │              LOOM — FEATURE DEMO (10/10)                    │
    └────────────────────────────────────────────────────────────┘
    """)

    t0 = time.time()
    agent = build_agent()
    config = {
        "configurable": {
            "thread_id": str(uuid7()),
            "user_id": "demo_user",
        }
    }

    demo_1_llm_registry()
    demo_2_langsmith()
    config = demo_3_observability(agent, config)
    config = demo_4_filesystem(agent, config)
    config = demo_5_subagent(agent, config)
    config = demo_6_memory_remember(agent, config)
    config = demo_7_memory_recall(agent, config)
    config = demo_8_hitl(agent, config)
    demo_9_multi_llm()
    demo_10_safety(agent, config)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  ✓ All 10 features demoed in {elapsed:.1f}s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
