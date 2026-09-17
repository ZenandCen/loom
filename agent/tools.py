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
# Active project database DSN (separate from Loom's infrastructure DB)
_project_db_dsn: str = os.getenv("LOOM_PROJECT_DB_DSN", "")


def _get_project_dir() -> Path:
    """Return the active project directory, or CODE_BASE_DIR if none set."""
    if _active_project and _active_project.exists():
        return _active_project
    return CODE_BASE_DIR


def get_project_db_dsn() -> str:
    """Return the project database DSN (empty if not set)."""
    return _project_db_dsn


def set_project_db_dsn(dsn: str) -> str:
    """Set the project database DSN at runtime."""
    global _project_db_dsn
    _project_db_dsn = dsn
    return f"Project database set: {dsn.split('@')[-1] if '@' in dsn else dsn}"


def _get_query_dsn() -> str:
    """Get the DSN to use for project DB queries.

    If a project DB DSN is set, use it. Otherwise, if no project is active,
    fall back to Loom's infra DSN. If a project IS active but no project DSN,
    return empty (to avoid accidentally querying Loom's own tables).
    """
    if _project_db_dsn:
        return _project_db_dsn
    if _active_project:
        return ""  # Project active but no DB configured — refuse
    from agent.config import _pg_dsn
    return _pg_dsn


def get_active_collection() -> str:
    """Derive RAG collection name from the active project.

    e.g. project 'my-ai-project' → collection 'my_ai_project'
    Falls back to 'rag_kb' if no project is set.
    """
    from rag.config import get_rag_settings
    if _active_project:
        name = _active_project.name.lower().replace("-", "_").replace(" ", "_")
        return name
    return get_rag_settings().default_collection


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


def reset_project() -> str:
    """Clear the active project and its DB. Collection falls back to rag_kb (free mode)."""
    global _active_project, _project_db_dsn
    _active_project = None
    _project_db_dsn = ""
    return "Cleared project → rag_kb (free mode)."


@tool(parse_docstring=True)
def read_file(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read a source code file. Supports pagination for large files.

    Args:
        path: Relative path to the file (e.g. 'src/main.py')
        start_line: First line to read (1-based, 0 = from beginning)
        end_line: Last line to read (0 = to end)
    """
    project_dir = _get_project_dir()
    file_path = (project_dir / path).resolve()
    if not str(file_path).startswith(str(CODE_BASE_DIR.resolve())):
        return "Error: Access denied. Path must be within the code base."
    if not file_path.exists():
        file_path = (CODE_BASE_DIR / path).resolve()
        if not file_path.exists():
            return f"Error: File '{path}' not found in '{project_dir.name}'."
    if not file_path.is_file():
        return f"Error: '{path}' is not a file."
    try:
        # Detect binary files (xlsx, docx, pdf, etc.)
        suffix = file_path.suffix.lower()
        binary_exts = {".xlsx", ".xls", ".docx", ".pdf", ".pptx", ".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".woff", ".woff2", ".ttf", ".exe", ".dll", ".so", ".pyc"}
        if suffix in binary_exts:
            return (
                f"Cannot read binary file '{file_path.name}' as text.\n"
                f"Use `reindex_folder` to index this file into RAG, then `rag_query` to ask questions about it.\n"
                f"Example: reindex_folder('.') then rag_query('what tables are listed?')"
            )
        lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
        total = len(lines)
        s = max(start_line - 1, 0) if start_line > 0 else 0
        e = min(end_line, total) if end_line > 0 else total
        chunk = lines[s:e]
        content = "".join(chunk)
        header = f"[{file_path.name}: lines {s+1}-{e} of {total}]\n"
        return header + content
    except UnicodeDecodeError:
        return (
            f"Cannot read '{file_path.name}' as text (binary/encoded file).\n"
            f"Use `reindex_folder` to index this file into RAG, then `rag_query` to ask questions about it."
        )
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
def read_folder(path: str = ".", max_lines_per_file: int = 200, overlap: int = 20) -> str:
    """Read ALL files in a folder in parallel. Large files are chunked with overlap.

    Uses parallel workers to speed up reading. Each file is read in overlapping
    chunks to ensure no data is lost at boundaries.

    Args:
        path: Folder path relative to active project
        max_lines_per_file: Max lines per chunk before paginating (default 200)
        overlap: Lines of overlap between chunks to prevent data loss (default 20)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    project_dir = _get_project_dir()
    dir_path = (project_dir / path).resolve()
    if not dir_path.exists():
        dir_path = (CODE_BASE_DIR / path).resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        return f"Error: Directory '{path}' not found."

    skip_dirs = {".git", ".venv", "__pycache__", "node_modules", "chroma_data"}
    skip_exts = {".png", ".jpg", ".jpeg", ".pdf", ".zip", ".pyc", ".so", ".bin", ".ico", ".wasm", ".xlsx", ".xls", ".docx"}

    # Collect all files
    files = []
    for f in sorted(dir_path.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            if f.suffix.lower() not in skip_exts:
                parts = f.parts
                if not any(p in skip_dirs for p in parts):
                    files.append(f)

    if not files:
        return f"No readable files in '{path}'."

    def _read_one_file(fpath: Path) -> str:
        """Read a single file with pagination + overlap."""
        try:
            lines = fpath.read_text(encoding="utf-8").splitlines(keepends=True)
        except Exception:
            return f"--- {fpath.name} (binary/unreadable) ---"

        rel = fpath.relative_to(dir_path)
        total = len(lines)

        if total <= max_lines_per_file:
            return f"--- {rel} ({total} lines) ---\n{''.join(lines)}"

        # Paginate with overlap
        chunks = []
        pos = 0
        chunk_num = 0
        while pos < total:
            end = min(pos + max_lines_per_file, total)
            chunk_lines = lines[pos:end]
            chunk_num += 1
            chunks.append(f"[{rel} chunk {chunk_num}: lines {pos+1}-{end}/{total}]\n{''.join(chunk_lines)}")
            if end >= total:
                break
            # Next chunk starts WITHIN current chunk (overlap)
            pos = end - overlap

        return "\n\n".join(chunks)

    # Read files in parallel
    results = [f"--- {f.relative_to(dir_path)} ---\n(read failed)" for f in files]
    with ThreadPoolExecutor(max_workers=min(8, len(files))) as executor:
        futures = {executor.submit(_read_one_file, f): i for i, f in enumerate(files)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = f"--- {files[idx].name} ---\nError: {e}"

    # Limit total output
    output = "\n\n".join(results)
    if len(output) > 80000:
        output = output[:80000] + f"\n\n[OUTPUT LIMIT: {len(output)}+ chars. {len(files)} files total. Read specific files individually for full content.]"
    return f"📂 {path}/ — {len(files)} files (parallel read, overlap={overlap}):\n\n{output}"


@tool(parse_docstring=True)
def check_indexed(path: str, collection: str = "") -> str:
    """Check if a file or folder has been indexed into the RAG vector store.

    Returns the number of chunks found for the given path in the vector store.
    Use this BEFORE deciding whether to index or query.

    Args:
        path: File or folder path to check (relative to active project)
        collection: Override collection name (default: active project's collection)
    """
    project_dir = _get_project_dir()
    check_path = (project_dir / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()

    collection = collection or get_active_collection()
    try:
        from rag.indexing import get_vectorstore
        vs = get_vectorstore(collection)

        # Search for documents with matching source metadata
        # Use a broad query to see if ANY chunks from this path exist
        results = vs.similarity_search_with_relevance_scores("document", k=50)
        if not results:
            return f"Not indexed: No chunks found in '{collection}' for '{path}'. You should use `reindex_folder` or `reindex_file` first."

        # Filter results that match our path
        matched_sources = set()
        for doc, score in results:
            source = doc.metadata.get("source", "")
            if str(check_path) in source or path in source:
                matched_sources.add(source.split("/")[-1])

        if matched_sources:
            return f"Indexed: Found chunks from {len(matched_sources)} file(s) in '{collection}': {', '.join(sorted(matched_sources)[:10])}. You can use `rag_query` to ask questions."
        else:
            return f"Not indexed: No chunks matching '{path}' in '{collection}'. You should use `reindex_folder` or `reindex_file` first."
    except Exception as e:
        return f"Error checking index: {e}"


@tool(parse_docstring=True)
def reindex_file(path: str, collection: str = "") -> str:
    """Embed and index a document file into the RAG vector store.

    Args:
        path: Path to the file (relative to active project)
        collection: Override collection name (default: active project's collection)
    """
    project_dir = _get_project_dir()
    file_path = (project_dir / path).resolve()
    if not file_path.exists():
        file_path = (CODE_BASE_DIR / path).resolve()
    if not file_path.exists():
        return f"Error: File '{path}' not found."

    from rag.indexing import load_document, get_vectorstore
    from rag.chunking import ChunkingConfig, chunk_documents

    collection = collection or get_active_collection()
    try:
        docs = load_document(file_path)
        if not docs:
            return f"Error: Cannot extract content from '{path}' (unsupported or empty)."
        chunks = chunk_documents(docs, ChunkingConfig())
        vs = get_vectorstore(collection)
        vs.add_documents(chunks)
        return f"Indexed '{file_path.name}' → {len(chunks)} chunks in '{collection}'."
    except Exception as e:
        return f"Reindex failed: {e}"


@tool(parse_docstring=True)
def reindex_folder(path: str, collection: str = "") -> str:
    """Index ALL supported documents in a folder into the RAG vector store.

    Recursively scans the folder for supported files (.pdf, .md, .txt, .html, .docx, .csv, .xlsx),
    loads them, chunks them, embeds them, and stores them in the vector DB.
    Use this instead of calling reindex_file multiple times for a folder of documents.

    Args:
        path: Folder path (relative to active project, or absolute)
        collection: Override collection name (default: active project's collection)
    """
    project_dir = _get_project_dir()
    folder_path = (project_dir / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not folder_path.exists():
        folder_path = (CODE_BASE_DIR / path).resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        return f"Error: Folder '{path}' not found."

    from rag.indexing import load_documents, get_vectorstore
    from rag.chunking import ChunkingConfig, chunk_documents

    collection = collection or get_active_collection()
    try:
        docs = load_documents(folder_path)
        if not docs:
            return f"No supported documents found in '{folder_path}'. Supported: .pdf, .md, .txt, .html, .docx, .csv, .xlsx"
        chunks = chunk_documents(docs, ChunkingConfig())
        vs = get_vectorstore(collection)
        vs.add_documents(chunks)
        files = set(d.metadata.get("source", "?").split("/")[-1] for d in docs)
        return f"Indexed {len(files)} files → {len(chunks)} chunks in '{collection}' collection.\nFiles: {', '.join(sorted(files)[:15])}"
    except Exception as e:
        return f"Reindex folder failed: {e}"


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


@tool(parse_docstring=True)
def query_database(sql: str) -> str:
    """Run a SQL query against the project's PostgreSQL database.

    Only SELECT queries are allowed. Results are limited to 10 rows.

    Args:
        sql: SQL query to execute (SELECT only)
    """
    import re as _re

    # Safety: only allow SELECT
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
        return "Error: Only SELECT queries are allowed."
    # Block dangerous keywords
    dangerous = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
        "TRUNCATE", "GRANT", "REVOKE", "INTO",
        "FOR UPDATE", "FOR SHARE", "FOR NO KEY UPDATE",
    ]
    for kw in dangerous:
        if _re.search(rf"\b{kw}\b", stripped):
            return f"Error: '{kw}' is not allowed. Only read-only SELECT queries."
    # Block dangerous PostgreSQL functions
    dangerous_funcs = [
        "pg_terminate_backend", "pg_cancel_backend",
        "lo_import", "lo_export", "pg_read_file", "pg_read_binary_file",
        "pg_terminate", "dblink", "dblink_connect",
        "pg_reload_conf", "pg_rotate_logfile",
        "pg_ls_dir", "pg_stat_file",
    ]
    for func in dangerous_funcs:
        if func.upper() in stripped:
            return f"Error: Function '{func}' is not allowed."
    # Block COPY, EXECUTE, SET, DO, VACUUM, REINDEX, CLUSTER
    blocked_cmds = ["COPY", "EXECUTE", "SET ", "DO $$", "VACUUM", "REINDEX", "CLUSTER", "REFRESH"]
    for cmd in blocked_cmds:
        if stripped.startswith(cmd):
            return f"Error: '{cmd.strip()}' is not allowed."

    dsn = _get_query_dsn()
    if not dsn:
        return "Error: No project database configured. Use `db <dsn>` to set it."

    import psycopg

    try:
        conn = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
        cur = conn.cursor()
        # Set statement timeout to 30 seconds
        cur.execute("SET statement_timeout = 30000")
        # Add LIMIT if not present
        if "LIMIT" not in stripped:
            sql = sql.rstrip().rstrip(";") + " LIMIT 1000"
        cur.execute(sql)
        rows = cur.fetchmany(10)  # Fetch at most 10 rows
        cols = [desc[0] for desc in cur.description] if cur.description else []
        total_estimate = len(rows)
        conn.close()

        if not rows:
            return "Query returned 0 rows."

        shown = rows

        # Format as table
        max_widths = [len(c) for c in cols]
        str_rows = []
        for row in shown:
            str_row = [str(v) if v is not None else "NULL" for v in row]
            str_rows.append(str_row)
            for i, v in enumerate(str_row):
                max_widths[i] = max(max_widths[i], len(v))

        header = " | ".join(c.ljust(max_widths[i]) for i, c in enumerate(cols))
        separator = "-+-".join("-" * w for w in max_widths)
        body = "\n".join(" | ".join(v.ljust(max_widths[i]) for i, v in enumerate(r)) for r in str_rows)

        result = f"{header}\n{separator}\n{body}"
        if len(rows) > limit:
            result += f"\n... showing {limit}/{len(rows)} rows. Use LIMIT/OFFSET for more."
        return result

    except Exception as e:
        return f"Query error: {e}"


@tool(parse_docstring=True)
def list_tables() -> str:
    """List all tables in the project's PostgreSQL database with their columns.

    Returns:
        List of tables with column names and types.
    """
    dsn = _get_query_dsn()
    if not dsn:
        return "Error: No project database configured. Use `db <dsn>` in Slack to set it."

    import psycopg

    try:
        conn = psycopg.connect(dsn, autocommit=True)
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """)
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return "No tables found in 'public' schema."

        # Full: all tables with all columns (max 500 tables)
        tables: dict[str, list[str]] = {}
        for table, col, dtype in rows:
            tables.setdefault(table, []).append(f"  {col} ({dtype})")

        total_tables = len(tables)
        total_cols = len(rows)
        max_tables = 500

        result = [f"Database: {total_tables} tables, {total_cols} columns total"]
        result.append("")

        shown = 0
        for table in sorted(tables.keys()):
            if shown >= max_tables:
                break
            shown += 1
            result.append(f"📋 {table} ({len(tables[table])} cols):")
            result.extend(tables[table])
            result.append("")

        if shown < total_tables:
            result.append(f"... {total_tables - shown} more tables not shown")
            result.append("Use `describe_table('<name>')` for specific tables.")

        return "\n".join(result)

    except Exception as e:
        return f"Error: {e}"


@tool(parse_docstring=True)
def describe_table(table_name: str) -> str:
    """Show full column details for a specific table.

    Args:
        table_name: Name of the table to describe
    """
    dsn = _get_query_dsn()
    if not dsn:
        return "Error: No project database configured. Use `db <dsn>` in Slack to set it."

    import psycopg

    try:
        conn = psycopg.connect(dsn, autocommit=True)
        cur = conn.cursor()
        cur.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return f"Table '{table_name}' not found in 'public' schema."

        result = [f"📋 {table_name} ({len(rows)} columns):"]
        for col, dtype, nullable, default in rows:
            null_mark = "" if nullable == "NO" else "?"
            default_str = f" = {default}" if default else ""
            result.append(f"  {col}{null_mark} ({dtype}){default_str}")

        # Also get row count estimate
        try:
            conn = psycopg.connect(dsn, autocommit=True)
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM \"{table_name}\"")
            count = cur.fetchone()[0]
            conn.close()
            result.append(f"\n📊 Approx rows: {count:,}")
        except Exception:
            pass

        return "\n".join(result)

    except Exception as e:
        return f"Error: {e}"


# --- Tools list (pass to agent) ---
all_tools = [
    tavily_search,
    send_email,
    remember,
    recall,
    set_project,
    list_projects,
    read_file,
    read_folder,
    list_directory,
    search_code,
    reindex_file,
    reindex_folder,
    check_indexed,
    query_database,
    list_tables,
    *all_rag_tools,
]
