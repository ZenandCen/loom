"""SRE module configuration — Grafana/Prometheus/Loki/PMM/GitLab endpoints."""

import os

GRAFANA_BASE = os.getenv("GRAFANA_BASE_URL", "")
GRAFANA_USER = os.getenv("GRAFANA_USER", "")
GRAFANA_PASSWORD = os.getenv("GRAFANA_PASSWORD", "")
PROM_DS_ID = os.getenv("PROM_DATASOURCE_ID", "1")
LOKI_DS_ID = os.getenv("LOKI_DATASOURCE_ID", "2")

PROM_URL = f"{GRAFANA_BASE}/api/datasources/proxy/{PROM_DS_ID}/api/v1" if GRAFANA_BASE else ""
LOKI_URL = f"{GRAFANA_BASE}/api/datasources/proxy/{LOKI_DS_ID}/loki/api/v1" if GRAFANA_BASE else ""
GRAFANA_AUTH = (GRAFANA_USER, GRAFANA_PASSWORD)

PMM_URL = os.getenv("PMM_URL", "")
PMM_USER = os.getenv("PMM_USER", "")
PMM_PASS = os.getenv("PMM_PASS", "")

GITLAB_URL = os.getenv("GITLAB_URL", "")
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN", "")

K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "")
