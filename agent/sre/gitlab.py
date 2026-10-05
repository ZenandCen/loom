"""GitLab API client for self-hosted instances.

Uses plain `requests` with a Personal Access Token (api scope).
Only the endpoints needed for SRE tracing are implemented."""

import requests

from agent.sre.config import GITLAB_URL, GITLAB_TOKEN

_TIMEOUT = 20


def _headers() -> dict:
    return {"PRIVATE-TOKEN": GITLAB_TOKEN}


def _url(path: str) -> str:
    return f"{GITLAB_URL}/api/v4{path}"


def _get(path: str, params: dict | None = None) -> dict | list | None:
    if not GITLAB_URL or not GITLAB_TOKEN:
        return None
    try:
        r = requests.get(_url(path), headers=_headers(), params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_commits(repo: str, branch: str = "main", limit: int = 10, since_hours: int = 24) -> list[dict]:
    """Get recent commits from a repository.

    Args:
        repo: Project path (e.g. 'group/project' or 'group/sub/project')
        branch: Branch name
        limit: Max commits to return
        since_hours: Only commits newer than this many hours
    """
    from datetime import datetime, timedelta, timezone

    since = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = _get(f"/projects/{requests.utils.quote(repo, safe='')}/repository/commits", {
        "ref_name": branch,
        "per_page": limit,
        "since": since,
    })
    if not data:
        return []
    return [
        {
            "sha": c["short_id"],
            "full_sha": c["id"],
            "author": c.get("author_name", ""),
            "date": c.get("committed_date", ""),
            "message": c["title"] if "title" in c else c.get("message", "").split("\n")[0],
        }
        for c in data
    ]


def get_commit_detail(repo: str, sha: str) -> str:
    """Get commit diff (changed files + diff content)."""
    data = _get(f"/projects/{requests.utils.quote(repo, safe='')}/repository/commits/{sha}/diff")
    if not data:
        return f"Error: Could not fetch diff for commit {sha[:8]} in {repo}"
    lines = []
    for d in data:
        lines.append(f"--- {d.get('old_path', d.get('new_path', '?'))}")
        lines.append(f"+++ {d.get('new_path', d.get('old_path', '?'))}")
        diff = d.get("diff", "")
        if len(diff) > 2000:
            diff = diff[:2000] + "\n... [truncated]"
        lines.append(diff)
        lines.append("")
    result = "\n".join(lines)
    return result[:15000] if len(result) > 15000 else result


def get_merge_requests(repo: str, state: str = "merged", limit: int = 5) -> list[dict]:
    """Get recent merge requests.

    Args:
        repo: Project path
        state: open | merged | closed
        limit: Max MRs to return
    """
    data = _get(f"/projects/{requests.utils.quote(repo, safe='')}/merge_requests", {
        "state": state,
        "per_page": limit,
        "order_by": "updated_at",
        "sort": "desc",
    })
    if not data:
        return []
    return [
        {
            "iid": mr["iid"],
            "title": mr["title"],
            "author": mr.get("author", {}).get("name", ""),
            "state": mr["state"],
            "updated": mr.get("updated_at", ""),
            "source_branch": mr.get("source_branch", ""),
            "target_branch": mr.get("target_branch", ""),
        }
        for mr in data
    ]


def get_mr_changes(repo: str, mr_iid: int) -> str:
    """Get the changes (diffs) for a merge request."""
    data = _get(f"/projects/{requests.utils.quote(repo, safe='')}/merge_requests/{mr_iid}/changes")
    if not data:
        return f"Error: Could not fetch changes for MR !{mr_iid} in {repo}"
    changes = data.get("changes", [])
    lines = []
    for c in changes:
        lines.append(f"--- {c.get('old_path', '?')}")
        lines.append(f"+++ {c.get('new_path', '?')}")
        diff = c.get("diff", "")
        if len(diff) > 1500:
            diff = diff[:1500] + "\n... [truncated]"
        lines.append(diff)
        lines.append("")
    result = f"MR !{mr_iid}: {data.get('title', '?')} ({len(changes)} files changed)\n\n"
    result += "\n".join(lines)
    return result[:20000] if len(result) > 20000 else result


def get_pipeline_status(repo: str, ref: str = "main") -> list[dict]:
    """Get recent CI/CD pipeline status for a branch."""
    data = _get(f"/projects/{requests.utils.quote(repo, safe='')}/pipelines", {
        "ref": ref,
        "per_page": 5,
    })
    if not data:
        return []
    return [
        {
            "id": p["id"],
            "status": p["status"],
            "ref": p.get("ref", ""),
            "sha": p.get("sha", "")[:8],
            "created": p.get("created_at", ""),
            "updated": p.get("updated_at", ""),
        }
        for p in data
    ]


def list_projects(group: str = "") -> list[dict]:
    """List projects in a group (or all accessible if group is empty)."""
    params: dict = {"per_page": 30, "simple": True, "order_by": "last_activity_at"}
    if group:
        data = _get(f"/groups/{requests.utils.quote(group, safe='')}/projects", params)
    else:
        data = _get("/projects", params)
    if not data:
        return []
    return [
        {"path": p["path_with_namespace"], "name": p["name"], "last_activity": p.get("last_activity_at", "")}
        for p in data
    ]
