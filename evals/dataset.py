"""Eval dataset — test cases for agent evaluation."""

DATASET = [
    {
        "input": "What are the latest features in LangGraph 2025?",
        "expected_behavior": "Uses researcher subagent or tavily_search, returns factual info with sources",
        "category": "research",
    },
    {
        "input": "Write a report about AI agents and save it to /workspace/ai_agents.md",
        "expected_behavior": "Researches topic, writes structured report, saves to /workspace/",
        "category": "report",
    },
    {
        "input": "Send an email to bob@example.com with subject 'Meeting' and content 'Let's sync tomorrow'",
        "expected_behavior": "Triggers HITL interrupt for send_email approval",
        "category": "hitl",
    },
    {
        "input": "Remember that I prefer Python over JavaScript",
        "expected_behavior": "Uses remember tool, stores in /memories/ or store",
        "category": "memory",
    },
    {
        "input": "What's my preferred programming language?",
        "expected_behavior": "Uses recall tool, retrieves from memory",
        "category": "memory",
    },
    {
        "input": "Hello! How are you doing today?",
        "expected_behavior": "Responds conversationally without calling tools",
        "category": "chitchat",
    },
    {
        "input": "Research the top 5 LLM frameworks and generate a comparison report",
        "expected_behavior": "Uses report-pipeline subagent for structured output",
        "category": "pipeline",
    },
]
