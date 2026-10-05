"""PMM (Percona Monitoring and Management) session handling.

PMM uses cookie-based login (its datasource proxy doesn't accept basic auth),
so we keep a single module-level session and re-login lazily on 401."""

import requests

from agent.sre.config import PMM_URL, PMM_USER, PMM_PASS

_pmm_session: requests.Session | None = None


def _pmm_login() -> requests.Session | None:
    global _pmm_session
    if not PMM_URL or not PMM_PASS:
        return None
    try:
        sess = requests.Session()
        r = sess.post(
            f"{PMM_URL}/graph/login",
            json={"user": PMM_USER, "password": PMM_PASS},
            timeout=8,
        )
        if r.ok and "error" not in r.text.lower():
            _pmm_session = sess
            return sess
    except Exception:
        pass
    return None


def get_session() -> requests.Session | None:
    global _pmm_session
    if _pmm_session is None:
        _pmm_login()
    return _pmm_session


def pmm_query(promql: str, ds_id: int = 1) -> list[dict]:
    """Query PMM datasource proxy (requires session cookie auth)."""
    global _pmm_session
    if not PMM_URL:
        return []
    if _pmm_session is None:
        _pmm_login()
    if _pmm_session is None:
        return []
    url = f"{PMM_URL}/graph/api/datasources/proxy/{ds_id}/api/v1/query"
    try:
        r = _pmm_session.get(url, params={"query": promql}, timeout=15)
        if r.status_code == 401:
            _pmm_session = None
            _pmm_login()
            if _pmm_session:
                r = _pmm_session.get(url, params={"query": promql}, timeout=15)
        if r.ok:
            return r.json().get("data", {}).get("result", [])
    except Exception:
        pass
    return []
