"""Evaluators — LLM-as-judge and rule-based checks."""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from utils.models import LLM, get_llm

model = get_llm(LLM.OPENAI)


def evaluate_with_llm(input: str, output: str, expected: str) -> dict:
    """LLM-as-judge: evaluate if the agent's output meets expectations."""
    response = model.invoke([
        SystemMessage(content=(
            "You are an evaluator. Given a user input, the expected behavior, "
            "and the actual agent output, score the response.\n\n"
            "Return JSON: {\"score\": 0-10, \"passed\": bool, \"reason\": str}"
        )),
        HumanMessage(content=(
            f"Input: {input}\n"
            f"Expected behavior: {expected}\n"
            f"Actual output: {output[:2000]}"
        )),
    ])

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"score": 0, "passed": False, "reason": "Failed to parse evaluation"}


def check_tool_usage(messages: list, expected_tools: list[str]) -> dict:
    """Rule-based: check if expected tools were used."""
    used_tools = set()
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                used_tools.add(tc["name"])

    missing = set(expected_tools) - used_tools
    return {
        "passed": len(missing) == 0,
        "used_tools": list(used_tools),
        "missing_tools": list(missing),
    }
