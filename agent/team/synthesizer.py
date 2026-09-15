"""Synthesizer node — combines worker results into final answer with diagram."""

import logging
import re

from agent.team.state import TeamState
from agent.team.schemas import TeamOutput, WorkerResult, WorkerName, WorkerStatus, DiagramType
from utils.models import LLM, get_llm

logger = logging.getLogger(__name__)

_synth_model = get_llm(LLM.OPENAI)

SYNTH_PROMPT = """\
You are a synthesis engine. Combine results from multiple specialist workers into ONE coherent answer.

Rules:
- Organize by topic, not by worker
- Remove redundancy across workers
- If explaining architecture/workflow/flow, INCLUDE a Mermaid diagram in a separate ```mermaid block at the END
- Diagram types: graph TD (architecture), sequenceDiagram (flow), erDiagram (DB), flowchart TD (process), classDiagram (code structure)
- Max 8-12 nodes in diagram. Vietnamese labels if user speaks Vietnamese.
- End with 2-3 sentence summary
- If a worker returned empty/error, note it briefly and continue with available data
- Be concise but complete. Use bullet points and headers for structure.
- Put the mermaid diagram in its own code block at the very end, after all text.
"""


def _extract_diagram(text: str) -> tuple[str, str]:
    """Extract mermaid diagram from text. Returns (text_without_diagram, diagram)."""
    match = re.search(r'```mermaid\n(.*?)```', text, re.DOTALL)
    if match:
        diagram = match.group(1).strip()
        text_without = (text[:match.start()] + text[match.end():]).strip()
        return text_without, diagram
    return text, ""


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
        return {"synthesis": "No worker results to synthesize. The task may not have required any workers."}

    sections = []
    for label, content in results.items():
        sections.append(f"## {label}\n{content}")
    combined = "\n\n".join(sections)

    prompt = (
        f"User asked: {query}\n\n"
        f"Worker results:\n\n{combined}\n\n"
        f"Now synthesize into ONE coherent answer for the user:"
    )

    response = _synth_model.invoke([
        {"role": "system", "content": SYNTH_PROMPT},
        {"role": "user", "content": prompt},
    ])

    synthesis = response.content if isinstance(response.content, str) else str(response.content)
    text, diagram = _extract_diagram(synthesis)
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
