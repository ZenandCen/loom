"""Tool definitions for the loom agent.

Tools are the agent's hands — they interact with the external world.
Each tool is a pure function wrapped in LangChain's @tool decorator.
"""

import os
from pathlib import Path

from langchain_core.tools import tool
from tavily import TavilyClient

from agent.config import store
from rag.tool import all_rag_tools

tavily_client = TavilyClient()

# Code base: parent directory containing all projects
CODE_BASE_DIR = Path(os.getenv("LOOM_CODE_DIR", "."))

# Active project (switchable at runtime)
_active_project: Path | None = None


def _get_project_dir() -> Path:
    """Return the active project directory, or CODE_BASE_DIR if none set."""
    if _active_project and _active_project.exists():
        return _active_project
    return CODE_BASE_DIR


@tool(parse_docstring=True)
def tavily_search(query: str) -> str:
    """Search the web for information.

    Args:
        query: Search query to execute
    """
    results = tavily_client.search(query, max_results=5)
    return "\n\n".join(
        f"**{r['title']}**\n{r['url']}\n{r['content']}" for r in results["results"]
    )


@tool(parse_docstring=True)
def send_email(to: str, subject: str, content: str) -> str:
    """Send an email to a recipient.

    Args:
        to: Recipient email address
        subject: Email subject line
        content: Email body content
    """
    return f"Email sent to {to}: '{subject}'"


@tool(parse_docstring=True)
def remember(fact: str, category: str = "preference") -> str:
    """Save a fact to long-term memory for future sessions.

    Args:
        fact: The fact or preference to remember
        category: Category — preference, context, or rule
    """
    from langgraph.config import get_config

    config = get_config()
    user_id = config.get("configurable", {}).get("user_id", "default")
    store.put(("loom", user_id, category), fact, {"fact": fact, "category": category})
    return f"Remembered ({category}): {fact}"


@tool(parse_docstring=True)
def recall(query: str) -> str:
    """Search long-term memory for relevant facts.

    Args:
        query: What to recall from memory
    """
    from langgraph.config import get_config

    config = get_config()
    user_id = config.get("configurable", {}).get("user_id", "default")
    results = store.search(("loom", user_id), query=query)
    if not results:
        return "No relevant memories found."
    return "\n".join(item.value["fact"] for item in results)


@tool(parse_docstring=True)
def set_project(path: str) -> str:
    """Switch the active project. All subsequent file operations use this project as root.

    Args:
        path: Project name or relative path from base dir (e.g. 'Learning/loom' or 'Projects/my-app')
    """
    global _active_project
    project_path = (CODE_BASE_DIR / path).resolve()
    if not str(project_path).startswith(str(CODE_BASE_DIR.resolve())):
        return "Error: Access denied."
    if not project_path.exists() or not project_path.is_dir():
        return f"Error: Project '{path}' not found."
    _active_project = project_path
    return f"Active project set to: {project_path.name} ({project_path})"


@tool(parse_docstring=True)
def read_file(path: str) -> str:
    """Read the contents of a source code file in the active project.

    Args:
        path: Relative path to the file (e.g. 'src/main.py')
    """
    project_dir = _get_project_dir()
    file_path = (project_dir / path).resolve()
    if not str(file_path).startswith(str(CODE_BASE_DIR.resolve())):
        return "Error: Access denied. Path must be within the code base."
    if not file_path.exists():
        return f"Error: File '{path}' not found in '{project_dir.name}'."
    if not file_path.is_file():
        return f"Error: '{path}' is not a file."
    try:
        content = file_path.read_text(encoding="utf-8")
        if len(content) > 50000:
            return content[:50000] + f"\n\n... [truncated, file is {len(content)} chars total]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


@tool(parse_docstring=True)
def list_directory(path: str = ".") -> str:
    """List files and directories in the active project.

    Args:
        path: Relative directory path (default: project root)
    """
    project_dir = _get_project_dir()
    dir_path = (project_dir / path).resolve()
    if not str(dir_path).startswith(str(CODE_BASE_DIR.resolve())):
        return "Error: Access denied."
    if not dir_path.exists() or not dir_path.is_dir():
        return f"Error: Directory '{path}' not found in '{project_dir.name}'."
    entries = []
    for entry in sorted(dir_path.iterdir()):
        if entry.name.startswith("."):
            continue
        prefix = "📁" if entry.is_dir() else "📄"
        entries.append(f"{prefix} {entry.name}")
    return f"📂 {project_dir.name}/{path if path != '.' else ''}:\n" + ("\n".join(entries) if entries else "(empty)")


@tool(parse_docstring=True)
def search_code(pattern: str, path: str = ".") -> str:
    """Search for a pattern in source code (grep-like, regex supported).

    Args:
        pattern: Text or regex pattern to search for
        path: Directory to search in (default: active project root)
    """
    project_dir = _get_project_dir()
    dir_path = (project_dir / path).resolve()
    if not str(dir_path).startswith(str(CODE_BASE_DIR.resolve())):
        return "Error: Access denied."
    if not dir_path.exists():
        return f"Error: Directory '{path}' not found."

    import re

    results = []
    skip_dirs = {".git", ".venv", "__pycache__", "node_modules", "chroma_data", ".idea"}
    skip_exts = {".png", ".jpg", ".jpeg", ".pdf", ".zip", ".pyc", ".so", ".bin", ".ico", ".wasm"}

    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if Path(f).suffix in skip_exts:
                continue
            fpath = Path(root) / f
            try:
                lines = fpath.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            rel = fpath.relative_to(project_dir)
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    results.append(f"{rel}:{i}: {line.strip()}")
                    if len(results) > 50:
                        return "\n".join(results) + f"\n\n... [truncated at 50 matches]"
    return f"Search in '{project_dir.name}/{path}':\n" + ("\n".join(results) if results else "No matches found.")


@tool(parse_docstring=True)
def list_projects() -> str:
    """List all available projects in the code base directory.

    Returns:
        List of project directories with their top-level contents.
    """
    entries = []
    for entry in sorted(CODE_BASE_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if entry.name in ("workspace", "data", "chroma_data"):
            continue
        # Show top-level file count
        try:
            n_files = sum(1 for _ in entry.rglob("*") if _.is_file())
        except Exception:
            n_files = 0
        entries.append(f"📁 {entry.name}/ ({n_files} files)")
    return f"Code base: {CODE_BASE_DIR}\n\n" + ("\n".join(entries) if entries else "(no projects found)")


# --- Tools list (pass to agent) ---
all_tools = [
    tavily_search,
    send_email,
    remember,
    recall,
    set_project,
    list_projects,
    read_file,
    list_directory,
    search_code,
    *all_rag_tools,
]
