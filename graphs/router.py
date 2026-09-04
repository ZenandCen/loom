"""Triage / routing logic — classify user intent before routing."""

from typing import Literal

from pydantic import BaseModel, Field

from utils.models import LLM, get_llm

model = get_llm(LLM.OPENAI)


class TriageSchema(BaseModel):
    """Structured output for triage classification."""

    action: Literal["research", "email", "general", "end"] = Field(
        description="The type of task requested"
    )
    reasoning: str = Field(description="Brief explanation of classification")
    urgency: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="How urgent this task is",
    )


def triage_node(state: dict) -> dict:
    """Classify user request into a routing category."""
    llm = model.with_structured_output(TriageSchema)
    last_message = state["messages"][-1].content

    result = llm.invoke([
        {
            "role": "system",
            "content": (
                "Classify the user request.\n"
                "- research: information seeking, fact-finding, analysis\n"
                "- email: communication tasks (send, draft, reply)\n"
                "- general: anything else (chitchat, questions, tasks)\n"
                "- end: goodbye, thanks, no action needed\n"
            ),
        },
        {"role": "user", "content": last_message},
    ])

    return {
        "classification": f"route_to_{result.action}" if result.action != "end" else "end",
        "urgency": result.urgency,
        "triage_reasoning": result.reasoning,
    }
