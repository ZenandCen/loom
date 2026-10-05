"""SRE tools for the agent — Prometheus, Loki, K8s, PMM/MySQL, GitLab, NGINX.

Each tool is a @tool-decorated function returning a string.
The SRE agent uses these to trace incidents, check health, and correlate across sources.
"""

import time
from datetime import datetime, timezone, timedelta

from langchain_core.tools import tool

from agent.sre.config import (
    PROM_URL, LOKI_URL, GRAFANA_AUTH, K8S_NAMESPACE,
    PMM_URL, GITLAB_URL, GITLAB_TOKEN,
)
from agent.sre.prometheus import prom_query, prom_range_query, loki_query
from agent.sre.pmm import pmm_query
from agent.sre import gitlab as gl
from agent.sre import nginx as ngx

UTC7 = timezone(timedelta(hours=7))


# ─── Prometheus / Metrics ──────────────────────────────────────────────────────


@tool(parse_docstring=True)
def prom_query_tool(promql: str) -> str:
    """Run a PromQL instant query against Prometheus. Returns current metric values.

    Args:
        promql: The PromQL expression to evaluate (e.g. 'up', 'rate(http_requests_total[5m])')
    """
    results = prom_query(promql)
    if not results:
        return "No data returned."
    lines = []
    for r in results[:50]:
        metric_labels = " ".join(f'{k}="{v}"' for k, v in sorted(r.get("metric", {}).items()))
        val = r["value"][1]
        lines.append(f"{metric_labels} => {val}")
    output = "\n".join(lines)
    return output[:10000] if len(output) > 10000 else output


@tool(parse_docstring=True)
def prom_range_tool(promql: str, duration: str = "1h") -> str:
    """Run a PromQL range query to see trends over time.

    Args:
        promql: The PromQL expression (e.g. 'rate(container_cpu_usage_seconds_total[5m])')
        duration: Time window — 30m, 1h, 3h, 6h, 24h (default: 1h)
    """
    results = prom_range_query(promql, duration=duration)
    if not results:
        return f"No data for '{promql}' in last {duration}."
    lines = [f"Range: last {duration} | {len(results)} series\n"]
    for r in results[:20]:
        metric_labels = " ".join(f'{k}="{v}"' for k, v in sorted(r.get("metric", {}).items()))
        values = [float(v[1]) for v in r.get("values", [])]
        if values:
            avg = sum(values) / len(values)
            mx = max(values)
            mn = min(values)
            lines.append(f"{metric_labels}: avg={avg:.3f} min={mn:.3f} max={mx:.3f}")
        else:
            lines.append(f"{metric_labels}: (no values)")
    output = "\n".join(lines)
    return output[:10000] if len(output) > 10000 else output


@tool(parse_docstring=True)
def search_metric(keyword: str) -> str:
    """Search Prometheus metric names by keyword (partial match).

    Args:
        keyword: Keyword to search for in metric names
    """
    results = prom_query('__name__=~".*"' if not keyword else f'__name__=~".*{keyword}.*"')
    names = sorted({r["metric"].get("__name__", "") for r in results if r["metric"].get("__name__")})
    if not names:
        matched = [n for n in names if keyword.lower() in n.lower()]
        return f"No metrics matching '{keyword}'."
    matched = [n for n in names if keyword.lower() in n.lower()] if keyword else names
    if not matched:
        return f"No metrics matching '{keyword}'."
    return f"Found {len(matched)} metrics:\n" + "\n".join(matched[:50])


# ─── K8s / Pods / Nodes ────────────────────────────────────────────────────────


@tool(parse_docstring=True)
def get_namespaces() -> str:
    """List all Kubernetes namespaces with pod counts."""
    results = prom_query("count by (namespace) (kube_pod_info)")
    if not results:
        return "No namespace data found (Prometheus may not have kube-state-metrics)."
    lines = []
    for r in sorted(results, key=lambda x: x["metric"].get("namespace", "")):
        ns = r["metric"].get("namespace", "unknown")
        count = int(float(r["value"][1]))
        lines.append(f"{ns}: {count} pods")
    return f"{len(lines)} namespaces:\n" + "\n".join(lines)


@tool(parse_docstring=True)
def get_pod_status(namespace: str = "") -> str:
    """Get status of all pods in a namespace (or all namespaces).

    Args:
        namespace: Namespace name (empty = all namespaces)
    """
    ns = namespace or K8S_NAMESPACE
    if ns:
        query = f'kube_pod_status_phase{{namespace="{ns}"}} == 1'
    else:
        query = "kube_pod_status_phase == 1"
    results = prom_query(query)
    if not results:
        return f"No pods found in namespace '{ns or 'all'}'."
    by_phase: dict[str, list[str]] = {}
    for r in results:
        phase = r["metric"].get("phase", "Unknown")
        ns_val = r["metric"].get("namespace", "")
        pod = r["metric"].get("pod", "")
        by_phase.setdefault(phase, []).append(f"{ns_val}/{pod}" if not ns else pod)
    lines = [f"Pod status in '{ns or 'all namespaces'}':\n"]
    for phase in ["Running", "Pending", "Failed", "Succeeded", "Unknown"]:
        pods = by_phase.get(phase, [])
        if pods:
            lines.append(f"\n{phase} ({len(pods)}):")
            for p in pods[:30]:
                lines.append(f"  {p}")
    return "\n".join(lines)


@tool(parse_docstring=True)
def get_node_status() -> str:
    """Get Ready/NotReady status of all cluster nodes."""
    results = prom_query('kube_node_status_condition{condition="Ready"} == 1')
    if not results:
        return "No node data found."
    lines = []
    for r in sorted(results, key=lambda x: x["metric"].get("node", "")):
        node = r["metric"].get("node", "unknown")
        status = r["value"][1]
        lines.append(f"{node}: {'Ready' if status == '1' else 'NotReady'}")
    return f"{len(lines)} nodes:\n" + "\n".join(lines)


@tool(parse_docstring=True)
def get_namespace_resources(namespace: str = "") -> str:
    """Get CPU and memory usage for all pods in a namespace.

    Args:
        namespace: Namespace name (default: K8S_NAMESPACE env var)
    """
    ns = namespace or K8S_NAMESPACE
    if not ns:
        return "Error: No namespace specified and K8S_NAMESPACE not set."
    cpu_r = prom_query(f'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{ns}",container!=""}}[5m]))')
    mem_r = prom_query(f'sum by (pod) (container_memory_usage_bytes{{namespace="{ns}",container!=""}})')
    cpu = {r["metric"].get("pod", "?"): float(r["value"][1]) for r in cpu_r}
    mem = {r["metric"].get("pod", "?"): float(r["value"][1]) for r in mem_r}
    pods = sorted(set(cpu) | set(mem))
    if not pods:
        return f"No resource data for namespace '{ns}'."
    lines = [f"Resources in '{ns}' ({len(pods)} pods):\n"]
    lines.append(f"{'Pod':<60} {'CPU(cores)':>10} {'Mem(MB)':>10}")
    lines.append("-" * 82)
    for p in pods[:50]:
        c = cpu.get(p, 0)
        m = mem.get(p, 0) / 1024 / 1024
        lines.append(f"{p:<60} {c:>10.4f} {m:>10.1f}")
    return "\n".join(lines)


@tool(parse_docstring=True)
def get_active_alerts() -> str:
    """Get currently firing alerts from Prometheus/Alertmanager."""
    results = prom_query("ALERTS")
    if not results:
        return "No active alerts (or ALERTS metric not available). Try query_metric with specific alert rules."
    lines = []
    for r in results:
        name = r["metric"].get("alertname", "unknown")
        sev = r["metric"].get("severity", "warning")
        ns = r["metric"].get("namespace", "")
        instance = r["metric"].get("instance", r["metric"].get("pod", ""))
        state = r["value"][1]
        if float(state) > 0:
            lines.append(f"[{sev.upper()}] {name} | ns={ns} | {instance}")
    if not lines:
        return "No firing alerts."
    return f"{len(lines)} active alerts:\n" + "\n".join(lines)


@tool(parse_docstring=True)
def get_crashlooping_pods() -> str:
    """Find pods with high restart counts (CrashLoopBackOff detection)."""
    results = prom_query("increase(kube_pod_container_status_restarts_total[1h]) > 3")
    if not results:
        return "No pods with >3 restarts in the last hour."
    lines = []
    for r in results:
        ns = r["metric"].get("namespace", "")
        pod = r["metric"].get("pod", "")
        container = r["metric"].get("container", "")
        count = int(float(r["value"][1]))
        lines.append(f"[CRITICAL] {ns}/{pod} (container: {container}) — {count} restarts in 1h")
    return f"{len(lines)} crashlooping pod(s):\n" + "\n".join(lines)


# ─── Loki / Logs ───────────────────────────────────────────────────────────────


@tool(parse_docstring=True)
def get_pod_logs(namespace: str, pod: str, lines: int = 50) -> str:
    """Get recent log lines from a specific pod (via Loki).

    Args:
        namespace: Kubernetes namespace
        pod: Pod name
        lines: Number of log lines to fetch (default: 50)
    """
    logql = f'{{namespace="{namespace}",pod="{pod}"}}'
    entries = loki_query(logql, lines=lines, hours_back=1)
    if not entries:
        return f"No logs found for pod '{pod}' in '{namespace}' (last 1h)."
    return f"Logs for {namespace}/{pod} (last {len(entries)} lines):\n" + "\n".join(line for _, line in entries)


@tool(parse_docstring=True)
def search_logs(namespace: str, keyword: str, lines: int = 30, hours_back: int = 1) -> str:
    """Search logs in a namespace for a keyword (via Loki).

    Args:
        namespace: Kubernetes namespace to search in
        keyword: Keyword or pattern to search for
        lines: Max lines to return (default: 30)
        hours_back: How many hours back to search (default: 1)
    """
    logql = f'{{namespace="{namespace}"}} |= "{keyword}"'
    entries = loki_query(logql, lines=lines, hours_back=hours_back)
    if not entries:
        return f"No logs matching '{keyword}' in namespace '{namespace}' (last {hours_back}h)."
    return f"Found {len(entries)} log lines matching '{keyword}' in '{namespace}' (last {hours_back}h):\n" + "\n".join(line for _, line in entries)


@tool(parse_docstring=True)
def search_logs_by_job(job: str, keyword: str, hours_back: int = 24, limit: int = 50) -> str:
    """Search Loki logs by job label and keyword. Use when you know the Loki job name.

    Args:
        job: Loki job label (e.g. 'fho-prod/dapp-api-non-chain')
        keyword: Keyword to search for in log lines
        hours_back: How many hours back to search (default: 24)
        limit: Max log lines to return (default: 50)
    """
    logql = f'{{job="{job}"}} |= "{keyword}"'
    entries = loki_query(logql, lines=limit, hours_back=hours_back)
    if not entries:
        return f"No logs: job='{job}' keyword='{keyword}' last {hours_back}h"
    from datetime import datetime
    lines = [f"Found {len(entries)} logs | job='{job}' | '{keyword}' | last {hours_back}h (UTC+7):\n"]
    for ts, line in entries[:limit]:
        lines.append(f"[{ts}] {line[:300]}")
    return "\n".join(lines)


@tool(parse_docstring=True)
def count_logs_by_hour(job: str, keyword: str, days_back: int = 7) -> str:
    """Count matching log occurrences per hour over N days. Useful for finding time patterns.

    Args:
        job: Loki job label
        keyword: Keyword to search for
        days_back: How many days back to analyze (default: 7)
    """
    if not LOKI_URL:
        return "Loki not configured."
    end_ns = int(time.time() * 1e9)
    start_ns = end_ns - int(days_back * 86400 * 1e9)
    logql = f'{{job="{job}"}} |= "{keyword}"'
    try:
        import requests
        r = requests.get(
            f"{LOKI_URL}/query_range",
            auth=GRAFANA_AUTH,
            params={"query": logql, "limit": 5000, "start": str(start_ns), "end": str(end_ns), "direction": "forward"},
            timeout=60,
        )
        r.raise_for_status()
        results = r.json().get("data", {}).get("result", [])
    except Exception as e:
        return f"Error querying Loki: {e}"

    hour_counts: dict[int, int] = {}
    for stream in results:
        for ts_ns, _ in stream.get("values", []):
            h = datetime.fromtimestamp(int(ts_ns) // 1e9, tz=UTC7).hour
            hour_counts[h] = hour_counts.get(h, 0) + 1

    if not hour_counts:
        return f"No logs found for job='{job}' keyword='{keyword}' in last {days_back}d."
    out = [f"Hourly distribution (UTC+7) | job='{job}' | '{keyword}' | last {days_back}d:\n"]
    max_cnt = max(hour_counts.values())
    for h in range(24):
        cnt = hour_counts.get(h, 0)
        if cnt:
            bar = "#" * min(cnt // max(1, max_cnt // 40), 50)
            out.append(f"  {h:02d}:00  {cnt:>5d}  {bar}")
    return "\n".join(out)


# ─── PMM / MySQL ───────────────────────────────────────────────────────────────


@tool(parse_docstring=True)
def get_db_health() -> str:
    """Get overall MySQL cluster health: connections, QPS, slow queries, replication lag, buffer hit ratio."""
    if not PMM_URL:
        return "PMM not configured. Set PMM_URL, PMM_USER, PMM_PASS env vars."

    sections = []

    up = pmm_query("mysql_up")
    if up:
        sections.append("### MySQL Instances:")
        for r in up:
            svc = r["metric"].get("service_name", "?")
            status = "UP" if float(r["value"][1]) > 0.5 else "DOWN"
            sections.append(f"  {svc}: {status}")

    conn = pmm_query("mysql_global_status_threads_connected")
    if conn:
        sections.append("\n### Connections:")
        for r in conn:
            svc = r["metric"].get("service_name", "?")
            val = int(float(r["value"][1]))
            sections.append(f"  {svc}: {val} threads")

    qps = pmm_query("rate(mysql_global_status_questions[5m])")
    if qps:
        sections.append("\n### QPS (5m rate):")
        for r in qps:
            svc = r["metric"].get("service_name", "?")
            val = float(r["value"][1])
            sections.append(f"  {svc}: {val:.1f} queries/s")

    slow = pmm_query("rate(mysql_global_status_slow_queries[5m])")
    if slow:
        sections.append("\n### Slow Queries (5m rate):")
        for r in slow:
            svc = r["metric"].get("service_name", "?")
            val = float(r["value"][1])
            sections.append(f"  {svc}: {val:.2f} slow queries/s")

    lag = pmm_query("mysql_slave_status_seconds_behind_master")
    if lag:
        sections.append("\n### Replication Lag:")
        for r in lag:
            svc = r["metric"].get("service_name", "?")
            val = float(r["value"][1])
            status = "CRITICAL" if val > 30 else "WARNING" if val > 5 else "OK"
            sections.append(f"  {svc}: {val:.1f}s [{status}]")

    buffer = pmm_query("1 - (mysql_global_status_innodb_buffer_pool_reads - mysql_global_status_innodb_buffer_pool_read_requests) / (mysql_global_status_innodb_buffer_pool_read_requests + 1)")
    if buffer:
        sections.append("\n### InnoDB Buffer Hit Ratio:")
        for r in buffer:
            svc = r["metric"].get("service_name", "?")
            val = float(r["value"][1]) * 100
            sections.append(f"  {svc}: {val:.2f}%")

    if not sections:
        return "No MySQL metrics available from PMM."
    return "MySQL Health Summary:\n\n" + "\n".join(sections)


@tool(parse_docstring=True)
def get_slow_queries(limit: int = 5, node: str = "") -> str:
    """Get top slow queries from PMM with execution stats.

    Args:
        limit: Number of slow queries to return (default: 5)
        node: MySQL service name to filter (empty = all)
    """
    if not PMM_URL:
        return "PMM not configured."

    query = 'topk(10, sum by (service_name, digest, fingerprint) (rate(pmm_query_digest_sum[5m])) / (sum by (service_name, digest, fingerprint) (rate(pmm_query_digest_count[5m])) + 1e-9))'
    results = pmm_query(query)
    if not results:
        return "No slow query data from PMM."

    lines = [f"Top {limit} slow queries (by avg execution time):\n"]
    count = 0
    for r in results:
        if node and r["metric"].get("service_name") != node:
            continue
        svc = r["metric"].get("service_name", "?")
        digest = r["metric"].get("digest", "?")[:12]
        fp = r["metric"].get("fingerprint", "?")[:80]
        avg_time = float(r["value"][1])
        lines.append(f"  {count+1}. [{svc}] digest={digest}")
        lines.append(f"     avg={avg_time*1000:.1f}ms  fingerprint: {fp[:60]}")
        count += 1
        if count >= limit:
            break
    if count == 0:
        return "No slow queries found."
    return "\n".join(lines)


@tool(parse_docstring=True)
def get_mysql_variables(node: str = "") -> str:
    """Get key MySQL configuration variables (timeouts, connections, buffer pool).

    Args:
        node: MySQL service name to query (empty = all)
    """
    if not PMM_URL:
        return "PMM not configured."

    var_names = [
        "max_connections", "wait_timeout", "interactive_timeout",
        "innodb_buffer_pool_size", "slow_query_log", "long_query_time",
        "innodb_io_capacity", "innodb_log_file_size",
    ]
    lines = ["MySQL Configuration Variables:\n"]
    for var in var_names:
        results = pmm_query(f'mysql_global_variables{{variable_name="{var}"}}')
        for r in results:
            svc = r["metric"].get("service_name", "?")
            if node and svc != node:
                continue
            val = r["value"][1]
            lines.append(f"  {svc}: {var} = {val}")
    if len(lines) == 1:
        return "No MySQL variables found."
    return "\n".join(lines)


@tool(parse_docstring=True)
def get_connection_stats(node: str = "") -> str:
    """Get MySQL connection statistics: threads, aborted connections, etc.

    Args:
        node: MySQL service name to query (empty = all)
    """
    if not PMM_URL:
        return "PMM not configured."

    stats_queries = [
        ("Threads Connected", "mysql_global_status_threads_connected"),
        ("Threads Running", "mysql_global_status_threads_running"),
        ("Aborted Clients", "mysql_global_status_aborted_clients"),
        ("Aborted Connects", "mysql_global_status_aborted_connects"),
        ("Connection Errors", "rate(mysql_global_status_aborted_connects[5m])"),
    ]
    lines = ["MySQL Connection Stats:\n"]
    for label, query in stats_queries:
        results = pmm_query(query)
        for r in results:
            svc = r["metric"].get("service_name", "?")
            if node and svc != node:
                continue
            val = r["value"][1]
            try:
                val = f"{float(val):.2f}"
            except (ValueError, TypeError):
                pass
            lines.append(f"  {svc}: {label} = {val}")
    return "\n".join(lines)


# ─── GitLab ────────────────────────────────────────────────────────────────────


@tool(parse_docstring=True)
def gitlab_commits(repo: str, branch: str = "main", limit: int = 10, since_hours: int = 24) -> str:
    """Get recent commits from a GitLab repository.

    Args:
        repo: GitLab project path (e.g. 'group/project-name')
        branch: Branch name (default: main)
        limit: Max commits to return (default: 10)
        since_hours: Only commits from the last N hours (default: 24)
    """
    if not GITLAB_URL or not GITLAB_TOKEN:
        return "GitLab not configured. Set GITLAB_URL and GITLAB_TOKEN env vars."
    commits = gl.get_commits(repo, branch=branch, limit=limit, since_hours=since_hours)
    if not commits:
        return f"No commits in {repo} ({branch}) in last {since_hours}h."
    lines = [f"Recent commits in {repo} ({branch}), last {since_hours}h:\n"]
    for c in commits:
        lines.append(f"  {c['sha']} | {c['date']} | {c['author']}")
        lines.append(f"    {c['message']}")
    return "\n".join(lines)


@tool(parse_docstring=True)
def gitlab_commit_diff(repo: str, sha: str) -> str:
    """Get the diff (changed files) for a specific commit.

    Args:
        repo: GitLab project path (e.g. 'group/project-name')
        sha: Commit SHA (short or full)
    """
    if not GITLAB_URL or not GITLAB_TOKEN:
        return "GitLab not configured."
    return gl.get_commit_detail(repo, sha)


@tool(parse_docstring=True)
def gitlab_merge_requests(repo: str, state: str = "merged", limit: int = 5) -> str:
    """Get recent merge requests from a GitLab repository.

    Args:
        repo: GitLab project path (e.g. 'group/project-name')
        state: MR state — open, merged, closed (default: merged)
        limit: Max MRs to return (default: 5)
    """
    if not GITLAB_URL or not GITLAB_TOKEN:
        return "GitLab not configured."
    mrs = gl.get_merge_requests(repo, state=state, limit=limit)
    if not mrs:
        return f"No {state} MRs in {repo}."
    lines = [f"Recent {state} MRs in {repo}:\n"]
    for mr in mrs:
        lines.append(f"  !{mr['iid']} | {mr['updated']} | {mr['author']}")
        lines.append(f"    {mr['title']}")
        lines.append(f"    {mr['source_branch']} → {mr['target_branch']}")
    return "\n".join(lines)


@tool(parse_docstring=True)
def gitlab_mr_changes(repo: str, mr_iid: int) -> str:
    """Get the code changes (diff) for a specific merge request.

    Args:
        repo: GitLab project path (e.g. 'group/project-name')
        mr_iid: Merge request IID number (e.g. 42)
    """
    if not GITLAB_URL or not GITLAB_TOKEN:
        return "GitLab not configured."
    return gl.get_mr_changes(repo, mr_iid)


@tool(parse_docstring=True)
def gitlab_pipeline_status(repo: str, ref: str = "main") -> str:
    """Get recent CI/CD pipeline status for a branch.

    Args:
        repo: GitLab project path (e.g. 'group/project-name')
        ref: Branch or tag name (default: main)
    """
    if not GITLAB_URL or not GITLAB_TOKEN:
        return "GitLab not configured."
    pipelines = gl.get_pipeline_status(repo, ref=ref)
    if not pipelines:
        return f"No pipelines found for {repo} ({ref})."
    lines = [f"Recent pipelines in {repo} ({ref}):\n"]
    for p in pipelines:
        status_icon = "✓" if p["status"] == "success" else "✗" if p["status"] in ("failed", "canceled") else "⏳"
        lines.append(f"  {status_icon} #{p['id']} | {p['status']} | {p['sha']} | {p['updated']}")
    return "\n".join(lines)


# ─── NGINX ─────────────────────────────────────────────────────────────────────


@tool(parse_docstring=True)
def nginx_request_stats(hours: float = 1, top_n: int = 20) -> str:
    """Get NGINX request stats: top paths by request count.

    Args:
        hours: How many hours back to analyze (default: 1)
        top_n: Number of top paths to show (default: 20)
    """
    if not LOKI_URL:
        return "Loki not configured."
    return ngx.get_request_stats(hours=hours, top_n=top_n)


@tool(parse_docstring=True)
def nginx_error_stats(hours: float = 1) -> str:
    """Get NGINX 4xx/5xx error breakdown by upstream and path.

    Args:
        hours: How many hours back to analyze (default: 1)
    """
    if not LOKI_URL:
        return "Loki not configured."
    return ngx.get_error_stats(hours=hours)


@tool(parse_docstring=True)
def nginx_latency(hours: float = 1, path_filter: str = "") -> str:
    """Get NGINX upstream latency stats (avg/max) by path.

    Args:
        hours: How many hours back to analyze (default: 1)
        path_filter: Filter to specific path (e.g. '/api/loyalty')
    """
    if not LOKI_URL:
        return "Loki not configured."
    return ngx.get_latency_stats(hours=hours, path_filter=path_filter)


# ─── Aggregated List ───────────────────────────────────────────────────────────

ALL_SRE_TOOLS = [
    prom_query_tool, prom_range_tool, search_metric,
    get_namespaces, get_pod_status, get_node_status, get_namespace_resources,
    get_active_alerts, get_crashlooping_pods,
    get_pod_logs, search_logs, search_logs_by_job, count_logs_by_hour,
    get_db_health, get_slow_queries, get_mysql_variables, get_connection_stats,
    gitlab_commits, gitlab_commit_diff, gitlab_merge_requests, gitlab_mr_changes, gitlab_pipeline_status,
    nginx_request_stats, nginx_error_stats, nginx_latency,
]
