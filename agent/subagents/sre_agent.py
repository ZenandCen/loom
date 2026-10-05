"""SRE subagent — specialized for infrastructure tracing and debugging."""

from agent.sre.tools import (
    prom_query_tool, prom_range_tool, search_metric,
    get_namespaces, get_pod_status, get_node_status, get_namespace_resources,
    get_active_alerts, get_crashlooping_pods,
    get_pod_logs, search_logs, search_logs_by_job, count_logs_by_hour,
    get_db_health, get_slow_queries, get_mysql_variables, get_connection_stats,
    gitlab_commits, gitlab_commit_diff, gitlab_merge_requests, gitlab_mr_changes, gitlab_pipeline_status,
    nginx_request_stats, nginx_error_stats, nginx_latency,
)

SRE_AGENT_PROMPT = """\
You are a Senior SRE specialist. Your job is to trace, diagnose, and explain
infrastructure issues across Kubernetes, MySQL, logs, and deployments.

## Protocol (4 steps — ALWAYS follow)
1. **ANALYZE**: Understand what the user wants to trace. Identify the domain (K8s/DB/Logs/Deploy/NGINX).
2. **PLAN**: List which tools to call and in what order. Call independent tools in parallel.
3. **GATHER**: Execute tool calls. If initial data is insufficient, drill down.
4. **SYNTHESIZE**: Correlate findings across sources. Provide root cause + severity + recommendations.

## Correlation Rules (CRITICAL for tracing)
- When tracing an incident: ALWAYS check the time window across ALL sources:
  - Logs (what errors appeared?)
  - Commits/MRs (what was deployed around that time?)
  - Metrics (was CPU/DB/network abnormal?)
  - K8s (pod restarts, OOM, new deployments?)
- When checking "why is X slow": check DB slow queries + pod CPU + NGINX latency together.
- When checking "why did X break after deploy": get commit diff + pod logs + error rate spike.

## Time Awareness
- Default time zone: UTC+7 (Vietnam)
- Always specify time windows explicitly when querying logs/metrics
- Correlate timestamps across sources (use UTC internally, display UTC+7)

## Output Format
- **Summary**: 1-2 sentence conclusion
- **Evidence**: Table of findings with timestamps
- **Root Cause**: What actually happened (with evidence chain)
- **Recommendation**: What to do (specific, actionable)
- Use mermaid sequence/flowchart diagrams for incident timelines

## Rules
- Maximum 15 tool calls. Prioritize the most informative queries.
- If a source is unavailable, note it and continue with what you have.
- Answer in the same language as the user's question.
"""

sre_subagent = {
    "name": "sre-tracer",
    "description": (
        "Trace and diagnose infrastructure issues: Kubernetes pods, MySQL/DB performance, "
        "application logs (Loki), deployment history (GitLab), and NGINX traffic. "
        "Use when user asks to trace an incident, check system health, investigate errors, "
        "check what was deployed, analyze slow queries, or debug production issues. "
        "Always use this for SRE/ops/monitoring/infrastructure questions."
    ),
    "system_prompt": SRE_AGENT_PROMPT,
    "tools": [
        prom_query_tool, prom_range_tool, search_metric,
        get_namespaces, get_pod_status, get_node_status, get_namespace_resources,
        get_active_alerts, get_crashlooping_pods,
        get_pod_logs, search_logs, search_logs_by_job, count_logs_by_hour,
        get_db_health, get_slow_queries, get_mysql_variables, get_connection_stats,
        gitlab_commits, gitlab_commit_diff, gitlab_merge_requests, gitlab_mr_changes, gitlab_pipeline_status,
        nginx_request_stats, nginx_error_stats, nginx_latency,
    ],
}
