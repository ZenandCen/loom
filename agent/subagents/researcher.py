"""Research subagent — LLM agent for web research tasks."""

from agent.prompt import RESEARCHER_PROMPT
from agent.tools import tavily_search

research_subagent = {
    "name": "researcher",
    "description": (
        "Research a topic using web search. "
        "Use for fact-finding, comparative analysis, and information gathering."
    ),
    "system_prompt": RESEARCHER_PROMPT,
    "tools": [tavily_search],
}
