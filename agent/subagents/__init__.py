from agent.subagents.researcher import research_subagent
from agent.subagents.workflow_pipeline import report_subagent

all_subagents = [research_subagent, report_subagent]

__all__ = ["research_subagent", "report_subagent", "all_subagents"]
