from agent.subagents.researcher import research_subagent
from agent.subagents.workflow_pipeline import report_subagent
from agent.subagents.rag_agent import rag_subagent
from agent.subagents.code_agent import code_subagent
from agent.subagents.db_agent import db_subagent

all_subagents = [
    research_subagent,
    report_subagent,
    rag_subagent,
    code_subagent,
    db_subagent,
]

__all__ = [
    "research_subagent",
    "report_subagent",
    "rag_subagent",
    "code_subagent",
    "db_subagent",
    "all_subagents",
]
