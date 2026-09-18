"""Synthesizer node — combines worker results into final answer with diagram."""

import logging
import re

from agent.team.state import TeamState
from agent.team.schemas import TeamOutput, WorkerResult, WorkerName, WorkerStatus, DiagramType
from utils.models import LLM, get_llm, get_backbone_llm

logger = logging.getLogger(__name__)

# Backbone answer model with automatic fallback: qwen -> gemini -> gemma(local).
_synth_model = get_backbone_llm()

SYNTH_PROMPT = """\
You are a synthesis engine. Combine results from multiple specialist workers into ONE comprehensive, detailed answer.

CRITICAL — Detail level:
- Be THOROUGH and COMPREHENSIVE. The user should be able to read your answer and fully understand the topic without needing to ask follow-up questions.
- Include ALL relevant specifics: file names, function names, table names, column names, config keys, API endpoints, exact values, code snippets where helpful.
- Do NOT summarize away details. If a worker found 5 related tables, list ALL 5 with their purposes. If there are 3 related functions, explain each one.
- Do NOT use vague language like "some components", "related modules", "various tables". Name them explicitly.
- If information is incomplete or a worker returned an error, state exactly what is missing and why.
- Structure with clear headers (##, ###) for each major section. Use tables when comparing items.
- Use bullet points for lists, but expand each bullet with enough context to be self-contained.
- If a worker returned empty/error, note it briefly and continue with available data.
- Length: as long as needed to be complete. Do NOT artificially truncate.
- Match the user's language (Vietnamese if they write in Vietnamese).

ANALYSIS DEPTH:
- Holistic View: Show how components connect. Don't just list isolated facts — explain relationships, data flow, and dependencies between them.
- Architecture Alignment: When suggesting changes or solutions, respect the EXISTING architecture and patterns. Do NOT introduce new flow types or patterns that contradict the codebase.
- Reuse Awareness: When discussing code, always check if similar utilities/patterns already exist. Reference them. "A small duplicate beats a wrong abstraction."
- Specific Metrics: When analyzing code complexity, provide concrete numbers (file LOC, function count, call depth, number of dependencies).

SOLUTION STRUCTURE (when the user asks for a solution/approach/design):
- Present 2-3 viable approaches with one-line trade-offs each.
- Mark your recommendation and explain WHY (1-2 sentences).
- For each approach, note: what changes, what stays the same, risk level, rollback difficulty.
- If the change affects multiple services/modules, list ALL affected areas explicitly.

ISSUE CLASSIFICATION (when identifying problems/gaps/risks):
- 🔴 **Blocking**: Must fix before proceeding. System will break or data will be lost.
- 🟡 **Important**: Should fix soon. Degraded functionality or technical debt.
- 🟢 **Nice-to-have**: Improvement opportunity. No urgency.
- For each issue: file/location, what's wrong, impact, recommended fix.

REQUIRED ENDING STRUCTURE (always include these sections at the end):
1. `## Key Takeaways` — 3-5 bullet points summarizing the most important facts.
2. `## Suggested Next Steps` — 2-3 concrete, actionable items the user can do next. Be specific (name the file, table, or command).
3. Mermaid diagram (if architecture/flow was discussed).

DIAGRAM RULES (Mermaid) — CRITICAL:
- When explaining architecture/flow/process, include a Mermaid diagram in a ```mermaid code block at the very END (after Key Takeaways and Next Steps).
- NEVER use ASCII art, box-drawing characters (─, │, ┌, ┐, └, ┘, ──→), or text-based diagrams. ONLY valid Mermaid syntax.
- Node IDs: alphanumeric + underscore ONLY (e.g. `A`, `minio_store`, `db_1`). NEVER use spaces, hyphens, or Vietnamese diacritics in IDs.
- Node labels: ALWAYS wrap in double quotes if they contain spaces, special chars, or Vietnamese: `A["MinIO Files"]`, `B["FTEL Source"]`
- Valid diagram types (choose ONE and use it correctly):
  * `flowchart TD` — for architecture/data flow (top-down). Edges: `A --> B`, `A -->|label| B`
  * `sequenceDiagram` — for request/response flow. Syntax: `participant A as Label`, `A->>B: message`
  * `erDiagram` — for DB relationships. Syntax: `TABLE1 ||--o{ TABLE2 : "has"`
  * `classDiagram` — for code structure
- Example of CORRECT flowchart:
  ```mermaid
  flowchart TD
      A["MinIO Storage"] --> B["OCR / Vision"]
      B --> C["Chunk & Index"]
      C --> D["PostgreSQL RAG"]
      E["FTEL Source"] --> F["Reconcile by Hour"]
      F --> G["Weekly Tables"]
      H["Member Data"] --> I["Compare / Upsert"]
      I --> J["DM Mismatch Table"]
  ```
- Example of WRONG (NEVER do this):
  ```
  [MinIO] ──→ [Process] ──→ [DB]
  ```
- Max 15 nodes. Keep labels short (2-4 words). Use Vietnamese in labels (quoted), English in IDs.
- If you are not confident the Mermaid will be valid, it is better to OMIT the diagram than to produce broken syntax.
"""


def _extract_diagram(text: str) -> tuple[str, str]:
    """Extract mermaid diagram from text. Returns (text_without_diagram, diagram)."""
    match = re.search(r'```mermaid\n(.*?)```', text, re.DOTALL)
    if match:
        diagram = match.group(1).strip()
        text_without = (text[:match.start()] + text[match.end():]).strip()
        return text_without, diagram
    return text, ""


def _clean_ascii_art(text: str) -> str:
    """Remove lines that look like ASCII art diagrams (box-drawing chars, arrow chains)."""
    box_chars = set("─│┌┐└┘├┤┬┴┼←→↑↓↔↕")
    lines = text.split("\n")
    cleaned = []
    skipping_block = False
    block_count = 0

    for line in lines:
        has_box = any(c in line for c in box_chars)
        # Detect lines with multiple bracketed items connected by arrows (ASCII flow)
        is_ascii_flow = bool(re.search(r'\[[^\]]+\]\s*[─→\-]{2,}\s*\[[^\]]+\]', line))

        if has_box or is_ascii_flow:
            block_count += 1
            if block_count <= 3:  # Only skip first few consecutive lines (the diagram)
                skipping_block = True
                continue
        else:
            skipping_block = False
            block_count = 0

        if not skipping_block:
            cleaned.append(line)

    # Collapse multiple blank lines
    result = re.sub(r'\n{3,}', '\n\n', "\n".join(cleaned))
    return result.strip()


def _validate_mermaid(diagram: str) -> str:
    """Validate and fix common Mermaid syntax errors. Returns cleaned diagram or empty string if unfixable."""
    if not diagram:
        return ""

    # Reject ASCII art (box-drawing characters)
    box_chars = set("─│┌┐└┘├┤┬┴┼←→↑↓")
    if any(c in diagram for c in box_chars):
        logger.warning("[MERMAID] Rejected ASCII art diagram (box-drawing chars detected)")
        return ""

    # Must start with a valid diagram type
    valid_starts = ("flowchart", "graph", "sequenceDiagram", "erDiagram", "classDiagram", "stateDiagram")
    first_line = diagram.split("\n")[0].strip()
    if not any(first_line.startswith(s) for s in valid_starts):
        logger.warning(f"[MERMAID] No valid diagram type header: '{first_line}'")
        return ""

    lines = diagram.split("\n")
    fixed_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed_lines.append("")
            continue

        # Fix: unquoted labels with spaces or special chars in flowchart/graph
        # Pattern: A[Label with spaces] → A["Label with spaces"]
        if re.match(r'^\s*\w+\[', stripped) and not re.match(r'^\s*\w+\["', stripped):
            # Has [ but not ["
            m = re.match(r'^(\s*\w+)\[([^\]"]+)\]$', stripped)
            if m:
                node_id, label = m.group(1), m.group(2)
                if " " in label or any(c in label for c in "()&{}"):
                    fixed_lines.append(f'{node_id}["{label}"]')
                    continue
        fixed_lines.append(line)

    result = "\n".join(fixed_lines)

    # Fix: replace common invalid arrows
    result = result.replace("──→", "-->")
    result = result.replace("──->", "-->")
    result = result.replace("→", "-->")
    result = result.replace("↑", "---")
    result = result.replace("↓", "---")

    # Fix: "graph TD" is valid but prefer "flowchart TD"
    if result.strip().startswith("graph "):
        result = "flowchart " + result.strip()[6:]

    return result.strip()


def _classify_diagram(diagram: str) -> DiagramType:
    """Classify diagram type from content."""
    if diagram.startswith("sequenceDiagram"):
        return DiagramType.SEQUENCE
    if diagram.startswith("erDiagram"):
        return DiagramType.ER
    if diagram.startswith("classDiagram"):
        return DiagramType.CLASS
    if diagram.startswith("flowchart") or diagram.startswith("graph"):
        return DiagramType.FLOWCHART
    if diagram.startswith("stateDiagram"):
        return DiagramType.ACTIVITY
    return DiagramType.FLOWCHART


def synthesize_node(state: TeamState) -> dict:
    """Combine all worker results into final synthesis."""
    query = state.get("user_query", "")
    history = state.get("history", "")
    results = {}
    for key, label in [
        ("rag_result", "Document Analysis"),
        ("code_result", "Code Exploration"),
        ("db_result", "Database Query"),
        ("web_result", "Web Research"),
    ]:
        val = state.get(key, "")
        if val:
            results[label] = val

    if not results:
        # No workers dispatched — answer directly (greeting, chit-chat, simple Q&A,
        # or a request the planner decided needs no tools).
        history_block = f"\n\nPrevious conversation:\n{history}" if history else ""
        direct_prompt = (
            "You are a knowledgeable, thorough AI assistant. Respond directly to the user's message.\n"
            "CRITICAL: If the user asked a real question, you MUST answer it DIRECTLY and COMPREHENSIVELY using your own knowledge. "
            "NEVER reply with a greeting like 'Hi, how can I help?' when a genuine question is present.\n"
            "- If it is ONLY a greeting or thanks (no real question), respond warmly in one sentence.\n"
            "- If it is a question, answer it THOROUGHLY — be detailed, specific, and complete. Include concrete examples, names, steps. Do NOT be vague or overly brief.\n"
            "- If the request is genuinely unclear, ask ONE short clarifying question.\n"
            "- Match the user's language (Vietnamese if they write in Vietnamese).\n"
            "- Be as detailed as needed to fully answer the question. Do NOT truncate or generalize.\n"
            f"{history_block}\n\n"
            f"User: {query}\n\nAssistant:"
        )
        try:
            response = _synth_model.invoke([
                {"role": "system", "content": "You are a thorough, knowledgeable AI assistant who gives detailed, specific, comprehensive answers."},
                {"role": "user", "content": direct_prompt},
            ])
            synthesis = response.content if isinstance(response.content, str) else str(response.content)
            synthesis = synthesis.strip()
            if not synthesis:
                return {"synthesis": "Mình ở đây để hỗ trợ! Bạn muốn hỏi gì?"}
            logger.info(f"[SYNTH] Direct response (no workers): {len(synthesis)} chars")
            return {"synthesis": synthesis}
        except Exception as e:
            logger.warning(f"Direct response failed, using fallback: {e}")
            return {"synthesis": "Mình ở đây để hỗ trợ! Bạn muốn hỏi gì?"}

    sections = []
    for label, content in results.items():
        sections.append(f"## {label}\n{content}")
    combined = "\n\n".join(sections)

    # Free mode (no project, rag_kb): bias toward combining RAG + web, and favor
    # modern, minimal-change, low-risk solutions that don't casually restructure.
    bias = ""
    if not state.get("project", ""):
        bias = (
            "\n\nLưu ý (chế độ chung / rag_kb): tổng hợp cả ngữ cảnh nội bộ (RAG) và nghiên cứu web thành MỘT phương án nhất quán. "
            "Đề xuất giải pháp HIỆN ĐẠI, TỐI GIẢN, ÍT đập code, an toàn; KHÔNG tự ý thay đổi cấu trúc DB hay kiến trúc hiện có. "
            "Ưu tiên phương án ít thay đổi nhất, dễ triển khai và dễ rollback nhất."
        )

    prompt = (
        f"User asked: {query}\n\n"
        f"Worker results:\n\n{combined}\n\n"
        f"Now synthesize into ONE coherent answer for the user:{bias}"
    )

    response = _synth_model.invoke([
        {"role": "system", "content": SYNTH_PROMPT},
        {"role": "user", "content": prompt},
    ])

    synthesis = response.content if isinstance(response.content, str) else str(response.content)
    text, diagram = _extract_diagram(synthesis)
    if diagram:
        diagram = _validate_mermaid(diagram)
    text = _clean_ascii_art(text)
    logger.info(f"[SYNTH] Final answer: {len(synthesis)} chars, diagram={'yes' if diagram else 'no'}")
    return {"synthesis": text, "diagram": diagram}


def build_team_output(state: TeamState) -> TeamOutput:
    """Build structured TeamOutput from final state (called by API layer)."""
    workers: list[WorkerResult] = []

    worker_map = {
        "rag_result": (WorkerName.RAG, "rag_sources"),
        "code_result": (WorkerName.CODE, "code_sources"),
        "db_result": (WorkerName.DB, "db_sources"),
        "web_result": (WorkerName.WEB, "web_sources"),
    }

    for key, (name, src_key) in worker_map.items():
        content = state.get(key, "")
        if content:
            workers.append(WorkerResult(
                worker=name,
                status=WorkerStatus.OK,
                content=content,
                sources=state.get(src_key, []),
                chars=len(content),
            ))

    diagram = state.get("diagram", "")
    return TeamOutput(
        synthesis=state.get("synthesis", ""),
        plan=state.get("plan", ""),
        workers=workers,
        diagram=diagram,
        diagram_type=_classify_diagram(diagram) if diagram else DiagramType.NONE,
        total_chars=len(state.get("synthesis", "")) + len(diagram),
    )
