"""Code indexing: AST-based chunking for Python + regex-based for TypeScript.

Each function or class becomes a separate Document chunk with rich metadata,
enabling semantic retrieval of "what does this code do?" questions.
"""

import ast
import logging
import re
import uuid
import warnings
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

warnings.filterwarnings("ignore", category=SyntaxWarning)

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 4000
SKIP_DIRS = {".venv", "venv", "__pycache__", "node_modules", ".git", "dist", "build", ".tox"}
CODE_EXTENSIONS = {".py", ".ts", ".tsx"}


def _get_source_lines(source: str) -> list[str]:
    return source.splitlines()


def _extract_docstring(node: ast.AST) -> str:
    if ast.get_docstring(node):
        return ast.get_docstring(node).strip()
    return ""


def _format_signature(node, lines: list[str]) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = []
        for a in node.args.args:
            annotation = f": {ast.unparse(a.annotation)}" if a.annotation else ""
            args.append(f"{a.arg}{annotation}")
        for a in node.args.kwonlyargs:
            annotation = f": {ast.unparse(a.annotation)}" if a.annotation else ""
            args.append(f"{a.arg}{annotation}")
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        return f"{prefix}def {node.name}({', '.join(args)}){ret}"
    elif isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
        return f"class {node.name}({bases})"
    return ""


def _extract_body(node: ast.AST, lines: list[str], max_lines: int = 15) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return ""
    body = node.body
    if not body:
        return ""
    start = body[0].lineno - 1
    end = node.end_lineno if node.end_lineno else start + max_lines
    chunk_lines = []
    for i in range(start, min(end, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line in ("pass", "..."):
            continue
        chunk_lines.append(line)
        if len(chunk_lines) >= max_lines:
            break
    return "\n".join(chunk_lines)


def _chunk_function(node, source: str, lines: list[str], file_path: str, class_name: str = "") -> Document:
    docstring = _extract_docstring(node)
    signature = _format_signature(node, lines)
    body = _extract_body(node, lines)
    parts = [f"File: {file_path}"]
    if class_name:
        parts.append(f"Class: {class_name}")
    parts.append(f"Function: {node.name}")
    if docstring:
        parts.append(f'"{docstring}"')
    parts.append(signature)
    if body:
        parts.append(body)
    content = "\n".join(parts)
    if len(content) > MAX_CHUNK_CHARS:
        content = content[:MAX_CHUNK_CHARS] + "\n... (truncated)"
    metadata = {
        "source": file_path,
        "type": "code",
        "language": "python",
        "function": node.name,
        "class": class_name,
        "docstring": docstring[:200] if docstring else "",
    }
    return Document(page_content=content, metadata=metadata)


def _chunk_class(node, source: str, lines: list[str], file_path: str) -> list[Document]:
    chunks: list[Document] = []
    class_docstring = _extract_docstring(node)
    signature = _format_signature(node, lines)
    class_parts = [f"File: {file_path}"]
    class_parts.append(f"Class: {node.name}")
    if class_docstring:
        class_parts.append(f'"{class_docstring}"')
    class_parts.append(signature)
    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if methods:
        method_list = "\n".join(f"  - {_format_signature(m, lines)}" for m in methods[:20])
        class_parts.append(f"Methods:\n{method_list}")
    class_content = "\n".join(class_parts)
    if len(class_content) > MAX_CHUNK_CHARS:
        class_content = class_content[:MAX_CHUNK_CHARS]
    class_metadata = {
        "source": file_path,
        "type": "code",
        "language": "python",
        "class": node.name,
        "function": "",
        "docstring": class_docstring[:200] if class_docstring else "",
    }
    chunks.append(Document(page_content=class_content, metadata=class_metadata))
    for method in methods:
        if method.name.startswith("__") and method.name not in ("__init__", "__aenter__", "__aexit__"):
            continue
        chunk = _chunk_function(method, source, lines, file_path, class_name=node.name)
        chunks.append(chunk)
    return chunks


def parse_python_file(file_path: Path) -> list[Document]:
    """Parse a single Python file into Document chunks using AST."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError) as e:
        logger.warning(f"Cannot read {file_path}: {e}")
        return []
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return []
    lines = _get_source_lines(source)
    chunks: list[Document] = []
    module_docstring = _extract_docstring(tree)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(_chunk_function(node, source, lines, str(file_path)))
        elif isinstance(node, ast.ClassDef):
            chunks.extend(_chunk_class(node, source, lines, str(file_path)))
    if not chunks and source.strip():
        content = f"File: {file_path}\n"
        if module_docstring:
            content += f'"{module_docstring}"\n'
        content += "\n".join(lines[:30])
        if len(content) > MAX_CHUNK_CHARS:
            content = content[:MAX_CHUNK_CHARS]
        chunks.append(Document(
            page_content=content,
            metadata={"source": str(file_path), "type": "code", "language": "python", "class": "", "function": "module", "docstring": module_docstring[:200]},
        ))
    return chunks


# --- TypeScript parsing (regex-based) ---

TS_PATTERNS = [
    (re.compile(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*[\w<>\[\],\s|]+)?\s*\{'), "function"),
    (re.compile(r'(?:export\s+)?(?:default\s+)?class\s+(\w+)(?:\s+extends\s+[\w<>,\s]+)?\s*\{'), "class"),
    (re.compile(r'(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+[\w<>,\s]+)?\s*\{'), "interface"),
    (re.compile(r'(?:export\s+)?type\s+(\w+)\s*=\s*'), "type"),
    (re.compile(r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*(?::\s*[\w<>\[\],\s|]+)?\s*=>'), "arrow_fn"),
    (re.compile(r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function\s*)?\(([^)]*)\)\s*\{'), "fn_expr"),
]


def _extract_jsdoc(lines: list[str], def_line: int) -> str:
    """Extract JSDoc comment above a definition."""
    doc_lines = []
    i = def_line - 1
    while i >= 0:
        line = lines[i].strip()
        if line.startswith("*/") or (line.startswith("*") and not line.startswith("**/")):
            doc_lines.insert(0, line.lstrip("*/ ").rstrip())
            i -= 1
        elif line.endswith("/**") or line.startswith("/**"):
            doc_lines.insert(0, line.lstrip("/* ").rstrip())
            i -= 1
            while i >= 0:
                l = lines[i].strip()
                if l.endswith("*/"):
                    break
                doc_lines.insert(0, l.lstrip("*/ ").rstrip())
                i -= 1
            break
        else:
            break
    return " ".join(l for l in doc_lines if l)


def _find_block_end(lines: list[str], start_line: int, max_lines: int = 20) -> int:
    """Find where a block ends by counting braces."""
    depth = 0
    started = False
    for i in range(start_line, min(start_line + 200, len(lines))):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
                if started and depth == 0:
                    return i
    return start_line + max_lines


def parse_ts_file(file_path: Path) -> list[Document]:
    """Parse a TypeScript file into Document chunks using regex patterns."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError) as e:
        logger.warning(f"Cannot read {file_path}: {e}")
        return []
    lines = _get_source_lines(source)
    chunks: list[Document] = []
    seen_lines: set[int] = set()

    for pattern, kind in TS_PATTERNS:
        for match in pattern.finditer(source):
            name = match.group(1)
            args = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
            match_start = match.start()
            line_num = source[:match_start].count("\n")
            if line_num in seen_lines:
                continue
            seen_lines.add(line_num)

            jsdoc = _extract_jsdoc(lines, line_num)
            signature_line = lines[line_num].strip() if line_num < len(lines) else ""

            if kind == "function":
                end_line = _find_block_end(lines, line_num)
                body_lines = lines[line_num + 1: min(line_num + 15, end_line + 1)]
                body = "\n".join(l.strip() for l in body_lines if l.strip() and not l.strip().startswith("//"))
            elif kind == "class":
                end_line = _find_block_end(lines, line_num)
                methods = []
                for i in range(line_num + 1, min(end_line + 1, len(lines))):
                    m = re.match(r'\s+(?:public|private|protected|static|async|readonly|get|set|\s)*(?:\w+)\s*\([^)]*\)', lines[i])
                    if m:
                        methods.append(lines[i].strip())
                        if len(methods) >= 15:
                            break
                body = "Methods:\n" + "\n".join(f"  {m}" for m in methods) if methods else ""
            elif kind == "interface":
                end_line = _find_block_end(lines, line_num)
                props = lines[line_num + 1: min(end_line + 1, line_num + 20)]
                body = "\n".join(l.strip() for l in props if l.strip() and not l.strip() == "}")
            elif kind == "type":
                body = lines[line_num].strip()
            else:
                end_line = _find_block_end(lines, line_num)
                body_lines = lines[line_num + 1: min(line_num + 12, end_line + 1)]
                body = "\n".join(l.strip() for l in body_lines if l.strip())

            parts = [f"File: {file_path}"]
            parts.append(f"{'Class' if kind == 'class' else 'Type' if kind == 'type' else 'Interface' if kind == 'interface' else 'Function'}: {name}")
            if jsdoc:
                parts.append(f'"{jsdoc}"')
            parts.append(signature_line)
            if body:
                parts.append(body)
            content = "\n".join(parts)
            if len(content) > MAX_CHUNK_CHARS:
                content = content[:MAX_CHUNK_CHARS] + "\n... (truncated)"
            chunks.append(Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "type": "code",
                    "language": "typescript",
                    "function": name,
                    "class": name if kind == "class" else "",
                    "docstring": jsdoc[:200] if jsdoc else "",
                },
            ))

    if not chunks and source.strip():
        content = f"File: {file_path}\n" + "\n".join(lines[:30])
        if len(content) > MAX_CHUNK_CHARS:
            content = content[:MAX_CHUNK_CHARS]
        chunks.append(Document(
            page_content=content,
            metadata={"source": str(file_path), "type": "code", "language": "typescript", "class": "", "function": "module", "docstring": ""},
        ))
    return chunks


def index_code_directory(
    input_dir: Path,
    collection_name: str,
    max_files: int = 300,
) -> int:
    """Index all Python + TypeScript files in a directory into the RAG vector store.

    Scans the entire directory tree (recursive), skipping technical folders.
    """
    from rag.indexing import get_vectorstore

    code_files = []
    for f in sorted(input_dir.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in CODE_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in f.parts):
            continue
        if f.name.startswith("test_") or f.name.startswith("__"):
            continue
        code_files.append(f)
        if len(code_files) >= max_files:
            break

    if not code_files:
        logger.warning(f"No code files found in {input_dir}")
        return 0

    n_py = sum(1 for f in code_files if f.suffix == ".py")
    n_ts = sum(1 for f in code_files if f.suffix in (".ts", ".tsx"))
    logger.info(f"Found {len(code_files)} code files ({n_py} Python, {n_ts} TypeScript) in {input_dir}")

    all_chunks: list[Document] = []
    parent_records: list[dict] = []
    for code_file in code_files:
        if code_file.suffix == ".py":
            chunks = parse_python_file(code_file)
        else:
            chunks = parse_ts_file(code_file)
        if chunks:
            parent_id = str(uuid.uuid4())
            for chunk in chunks:
                chunk.metadata["parent_id"] = parent_id
            all_chunks.extend(chunks)
            # Store full file content as parent
            try:
                file_content = code_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                file_content = ""
            parent_records.append({
                "id": parent_id,
                "content": file_content,
                "source": str(code_file),
            })

    if not all_chunks:
        return 0

    # Store parent records (full file content) for parent expansion
    if parent_records:
        from rag.parents import store_parents
        store_parents(parent_records, collection_name)

    vs = get_vectorstore(collection_name)
    vs.add_documents(all_chunks)
    logger.info(f"Indexed {len(all_chunks)} code chunks from {len(code_files)} files into '{collection_name}' ({len(parent_records)} parents)")
    return len(all_chunks)
