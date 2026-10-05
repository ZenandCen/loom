"""Prometheus and Loki query helpers via Grafana datasource proxy."""

import time

import requests

from agent.sre.config import PROM_URL, LOKI_URL, GRAFANA_AUTH

_TIMEOUT = 20


def prom_query(promql: str, at: int | None = None) -> list[dict]:
    """Run a PromQL instant query. Returns list of result dicts."""
    if not PROM_URL:
        return []
    params: dict = {"query": promql}
    if at:
        params["time"] = at
    try:
        r = requests.get(f"{PROM_URL}/query", auth=GRAFANA_AUTH, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json().get("data", {}).get("result", [])
    except Exception:
        return []


def prom_range_query(promql: str, duration: str = "1h", step: str = "5m") -> list[dict]:
    """Run a PromQL range query. duration: 30m, 1h, 3h, 6h, 24h."""
    if not PROM_URL:
        return []
    durations = {"30m": 1800, "1h": 3600, "3h": 10800, "6h": 21600, "24h": 86400, "48h": 172800}
    end = int(time.time())
    start = end - durations.get(duration, 3600)
    try:
        r = requests.get(
            f"{PROM_URL}/query_range",
            auth=GRAFANA_AUTH,
            params={"query": promql, "start": start, "end": end, "step": step},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("data", {}).get("result", [])
    except Exception:
        return []


def loki_query(logql: str, lines: int = 100, hours_back: float = 1.0) -> list[tuple[str, str]]:
    """Query Loki for log lines. Returns list of (timestamp_str, line) tuples, newest first."""
    if not LOKI_URL:
        return []
    end_ns = int(time.time() * 1e9)
    start_ns = end_ns - int(hours_back * 3600 * 1e9)
    try:
        r = requests.get(
            f"{LOKI_URL}/query_range",
            auth=GRAFANA_AUTH,
            params={
                "query": logql,
                "limit": lines,
                "start": str(start_ns),
                "end": str(end_ns),
                "direction": "backward",
            },
            timeout=30,
        )
        r.raise_for_status()
        results = r.json().get("data", {}).get("result", [])
        entries = []
        for stream in results:
            for ts_ns, line in stream.get("values", []):
                entries.append((int(ts_ns), line))
        entries.sort(key=lambda x: x[0], reverse=True)
        return [(f"{int(ts)//1_000_000_000}", line) for ts, line in entries[:lines]]
    except Exception:
        return []
