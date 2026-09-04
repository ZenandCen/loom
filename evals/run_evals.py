"""Run evaluations against the loom agent.

Usage:
    python -m evals.run_evals
"""

import json
import sys

from langsmith import uuid7

from agent.build import build_agent
from evals.dataset import DATASET
from evals.evaluators import evaluate_with_llm


def run_evals(limit: int | None = None):
    """Run all eval cases and report results."""
    agent = build_agent()
    cases = DATASET[:limit] if limit else DATASET

    print(f"\n{'='*60}")
    print(f"  🧪 Running {len(cases)} eval cases")
    print(f"{'='*60}\n")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {case['category']}: {case['input'][:50]}...")

        config = {
            "configurable": {
                "thread_id": str(uuid7()),
                "user_id": "eval_test",
            }
        }

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": case["input"]}]},
                config=config,
            )

            output = result["messages"][-1].content if result.get("messages") else ""
            evaluation = evaluate_with_llm(case["input"], output, case["expected_behavior"])

            results.append({
                "case": i,
                "category": case["category"],
                "input": case["input"],
                "score": evaluation.get("score", 0),
                "passed": evaluation.get("passed", False),
                "reason": evaluation.get("reason", ""),
            })

            status = "✅" if evaluation.get("passed") else "❌"
            print(f"         {status} score={evaluation.get('score', '?')}/10")

        except Exception as e:
            results.append({
                "case": i,
                "category": case["category"],
                "input": case["input"],
                "score": 0,
                "passed": False,
                "reason": f"Error: {e}",
            })
            print(f"         ❌ Error: {e}")

    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    avg_score = sum(r["score"] for r in results) / total if total else 0

    print(f"\n{'='*60}")
    print(f"  📊 Results: {passed}/{total} passed | avg score: {avg_score:.1f}/10")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_evals(limit=limit)
