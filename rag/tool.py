"""RAG tool — exposes the RAG pipeline as a callable tool for the loom agent.

The main agent can call this tool to query the knowledge base.
This is the integration point between the RAG module and the agent system.
"""

import logging

from langchain_core.tools import tool

from rag.enums import PipelineLevel
from rag.pipeline import run_rag

logger = logging.getLogger(__name__)


@tool(parse_docstring=True)
def rag_query(query: str, pipeline_level: str = "adaptive") -> str:
    """Query the knowledge base using RAG (Retrieval-Augmented Generation).

    Searches indexed documents and generates a grounded answer with sources.
    Use this for questions that require information from your local knowledge base.

    Args:
        query: The question to answer using the knowledge base.
        pipeline_level: Pipeline complexity — "basic", "adaptive", "self_rag", or "multi_source".
    """
    try:
        level = PipelineLevel(pipeline_level)
    except ValueError:
        level = PipelineLevel.ADAPTIVE

    result = run_rag(query, level=level)

    # Format the response for the agent
    answer = result.get("generation", "No answer generated.")
    sources = result.get("sources", [])
    docs_count = result.get("documents_count", 0)

    response = f"{answer}"
    if sources:
        response += f"\n\nSources ({docs_count} documents):\n" + "\n".join(f"  - {s}" for s in sources)

    return response


# Export for registration in the agent's tool list
all_rag_tools = [rag_query]
