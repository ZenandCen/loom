"""NGINX access-log aggregation via Loki.

Parses NGINX ingress logs and provides request counts, error breakdowns,
and latency stats with automatic window-bisection retry when Loki's
max-series-per-query cap is hit."""

import json
import re
import time

import requests

from agent.sre.config import LOKI_URL, GRAFANA_AUTH

_NGINX_RE = re.compile(
    r'"(?P<method>[A-Z]+) (?P<path>[^ ?]+)[^"]*" '
    r'(?P<status>\d{3}) \d+ "[^"]*" "[^"]*" \d+ (?P<up_time>[\d.]+|-)'
    r'(?: \[(?P<upstream>[^\]]+)\])?'
)

_LOKI_REGEXP = (
    r'"[A-Z]+ (?P<path>/[^ ?]+)[^"]*" (?P<status>\d{3}) \d+ "[^"]*" "[^"]*" \d+'
    r' (?P<up_lat>[\d.]+)(?: \[(?P<upstream>[^\]]+)\])?'
)


def loki_count_over_time(
    job: str,
    hours: float = 1,
    error_only: bool = False,
) -> dict[str, int]:
    """Count NGINX requests grouped by (upstream, path) over a time window.

    Uses Loki count_over_time with regex extraction. Auto-bisects the window
    when Loki rejects the query for exceeding max-series-per-query.

    Returns dict: {(upstream, path): count}
    """
    if not LOKI_URL:
        return {}
    end_s = int(time.time())
    start_s = end_s - int(hours * 3600)
    return _count_window(job, start_s, end_s, error_only)


def _count_window(job: str, st: int, et: int, error_only: bool) -> dict[str, int]:
    dur = et - st
    if dur <= 0:
        return {}
    error_filter = '| status =~ "4..|5.."' if error_only else ""
    logql = (
        'sum by (upstream, path) ('
        f'  count_over_time('
        f'    {{job="{job}"}}'
        f'    | json | stream="stdout"'
        f'    | line_format "{{{{.log}}}}"'
        f'    | regexp `{_LOKI_REGEXP}`'
        f'{error_filter}'
        f'    [{dur}s]'
        f'  )'
        f')'
    )
    try:
        r = requests.get(
            f"{LOKI_URL}/query_range",
            auth=GRAFANA_AUTH,
            params={"query": logql, "start": str(st), "end": str(et), "step": str(max(dur, 60))},
            timeout=60,
        )
        if r.status_code == 400 and "maximum of series" in r.text and dur > 900:
            mid = st + dur // 2
            left = _count_window(job, st, mid, error_only)
            right = _count_window(job, mid, et, error_only)
            for k, v in right.items():
                left[k] = left.get(k, 0) + v
            return left
        if not r.ok:
            return {}
        d = r.json()
        if d.get("status") != "success":
            return {}
        result: dict[str, int] = {}
        for stream in d["data"]["result"]:
            up = stream["metric"].get("upstream", "?")
            path = stream["metric"].get("path", "?")
            cnt = sum(int(float(v)) for _, v in stream.get("values", []))
            key = f"{up}|{path}"
            result[key] = result.get(key, 0) + cnt
        return result
    except Exception:
        return {}


def get_request_stats(hours: float = 1, top_n: int = 20) -> str:
    """Get top request counts by upstream/path."""
    from agent.sre.config import K8S_NAMESPACE

    jobs = _get_nginx_jobs()
    all_counts: dict[str, int] = {}
    for job in jobs:
        counts = loki_count_over_time(job, hours=hours)
        for k, v in counts.items():
            all_counts[k] = all_counts.get(k, 0) + v

    if not all_counts:
        return f"No NGINX request data in last {hours}h."

    sorted_items = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    total = sum(all_counts.values())
    lines = [f"NGINX Request Stats (last {hours}h) — {total} total requests, top {top_n}:\n"]
    lines.append(f"{'Count':>8} {'Pct':>6}  Upstream | Path")
    lines.append("-" * 70)
    for key, cnt in sorted_items:
        up, path = key.split("|", 1)
        pct = cnt / total * 100 if total else 0
        lines.append(f"{cnt:>8} {pct:>5.1f}%  {up} | {path}")
    return "\n".join(lines)


def get_error_stats(hours: float = 1, top_n: int = 20) -> str:
    """Get 4xx/5xx error breakdown by upstream/path/status."""
    jobs = _get_nginx_jobs()
    all_counts: dict[str, int] = {}
    for job in jobs:
        counts = loki_count_over_time(job, hours=hours, error_only=True)
        for k, v in counts.items():
            all_counts[k] = all_counts.get(k, 0) + v

    if not all_counts:
        return f"No 4xx/5xx errors in last {hours}h."

    sorted_items = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    total_errors = sum(all_counts.values())
    lines = [f"NGINX Error Stats (4xx/5xx, last {hours}h) — {total_errors} total errors:\n"]
    lines.append(f"{'Count':>8}  Upstream | Path")
    lines.append("-" * 60)
    for key, cnt in sorted_items:
        up, path = key.split("|", 1)
        lines.append(f"{cnt:>8}  {up} | {path}")
    return "\n".join(lines)


def get_latency_stats(hours: float = 1, path_filter: str = "") -> str:
    """Get average upstream latency by path from NGINX logs."""
    if not LOKI_URL:
        return "Loki not configured."

    end_ns = int(time.time() * 1e9)
    start_ns = end_ns - int(hours * 3600 * 1e9)
    jobs = _get_nginx_jobs()

    latencies: dict[str, list[float]] = {}
    for job in jobs:
        logql = f'{{job="{job}"}}'
        try:
            r = requests.get(
                f"{LOKI_URL}/query_range",
                auth=GRAFANA_AUTH,
                params={"query": logql, "limit": 2000, "start": str(start_ns), "end": str(end_ns), "direction": "backward"},
                timeout=30,
            )
            if not r.ok:
                continue
            for stream in r.json().get("data", {}).get("result", []):
                for _, line in stream.get("values", []):
                    parsed = _parse_nginx_line(line)
                    if parsed and (not path_filter or path_filter in parsed["path"]):
                        key = f'{parsed["upstream"]}|{parsed["path"]}'
                        latencies.setdefault(key, []).append(parsed["up_time"])
        except Exception:
            continue

    if not latencies:
        return f"No NGINX latency data in last {hours}h."

    stats = []
    for key, values in latencies.items():
        avg_ms = sum(values) / len(values) * 1000
        max_ms = max(values) * 1000
        stats.append((key, avg_ms, max_ms, len(values)))

    stats.sort(key=lambda x: x[1], reverse=True)
    top = stats[:15]
    lines = [f"NGINX Latency (avg upstream time, last {hours}h), top 15 slowest:\n"]
    lines.append(f"{'Avg(ms)':>8} {'Max(ms)':>8} {'Count':>7}  Upstream | Path")
    lines.append("-" * 70)
    for key, avg_ms, max_ms, cnt in top:
        up, path = key.split("|", 1)
        lines.append(f"{avg_ms:>8.0f} {max_ms:>8.0f} {cnt:>7}  {up} | {path}")
    return "\n".join(lines)


def _get_nginx_jobs() -> list[str]:
    """Return the NGINX ingress job labels to query."""
    from agent.sre.config import K8S_NAMESPACE
    ns = K8S_NAMESPACE or "ingress-nginx"
    return [f"{ns}/ingress-nginx-controller", f"{ns}/ingress-nginx-internal-controller"]


def _parse_nginx_line(raw: str) -> dict | None:
    """Parse a single NGINX access log line (JSON-wrapped or plain)."""
    try:
        obj = json.loads(raw)
        if obj.get("stream") != "stdout":
            return None
        log = obj.get("log", "")
    except Exception:
        log = raw
    m = _NGINX_RE.search(log)
    if not m:
        return None
    g = m.groupdict()
    try:
        up_t = float(g["up_time"])
    except (ValueError, TypeError):
        up_t = 0.0
    return {
        "method": g["method"],
        "path": g["path"],
        "status": int(g["status"]),
        "up_time": up_t,
        "upstream": g.get("upstream", "?"),
    }
