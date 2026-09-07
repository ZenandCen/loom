"""RAGAS-based evaluation of the RAG pipeline.

Measures 4 key quality metrics:
    - Faithfulness: Is the answer grounded in retrieved context?
    - Answer Relevance: Does the answer address the question?
    - Context Precision: Are the most relevant docs ranked highest?
    - Context Recall: Does the context contain all needed information?

Each metric scores 0-1. Overall score is the arithmetic mean.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from rag.config import get_rag_settings
from rag.enums import PipelineLevel
from rag.pipeline import get_pipeline
from rag.retrieval import get_retriever

logger = logging.getLogger(__name__)


def load_eval_dataset(path: Optional[str] = None) -> list[dict]:
    """Load evaluation dataset from a JSON file.

    Expected format:
    [
        {
            "question": "What is X?",
            "ground_truth": "X is ...",
            "context": "Optional pre-retrieved context"
        }
    ]

    Args:
        path: Path to JSON file (default: settings.eval_dataset_path).

    Returns:
        List of evaluation sample dicts.
    """
    settings = get_rag_settings()
    path = path or settings.eval_dataset_path
    dataset_path = Path(path)

    if not dataset_path.exists():
        logger.warning(f"Eval dataset not found: {path}")
        return []

    with open(dataset_path) as f:
        return json.load(f)


def run_evaluation(
    dataset: Optional[list[dict]] = None,
    pipeline_level: PipelineLevel = PipelineLevel.ADAPTIVE,
    collection_name: Optional[str] = None,
) -> dict:
    """Run RAGAS evaluation on a question dataset.

    For each question:
        1. Run the pipeline to get the answer
        2. Retrieve context independently
        3. Score with RAGAS metrics

    Args:
        dataset: List of {question, ground_truth, context} dicts.
        pipeline_level: Which pipeline to evaluate.
        collection_name: Vector store collection.

    Returns:
        Dictionary with metric scores (0-1) and overall_score.
    """
    settings = get_rag_settings()
    if dataset is None:
        dataset = load_eval_dataset()

    if not dataset:
        logger.warning("No evaluation data available")
        return {}

    # Lazy import — RAGAS is a heavy dependency
    try:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import (
            AnswerRelevance,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError:
        logger.error("RAGAS not installed. Run: pip install ragas")
        return {"error": "ragas not installed"}

    # Build evaluation samples
    pipeline = get_pipeline(pipeline_level)
    retriever = get_retriever(collection_name)

    samples = []
    for item in dataset:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        # Get answer from the pipeline
        result = pipeline.invoke({"question": question, "query_rewrite_count": 0})
        answer = result.get("generation", "")

        # Get context from retriever (independent retrieval for context metrics)
        docs = retriever.invoke(question)
        context = [d.page_content for d in docs]

        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=context,
                response=answer,
                reference=ground_truth,
            )
        )

    # Run RAGAS evaluation
    eval_dataset = EvaluationDataset(samples=samples)
    metrics = {
        "faithfulness": Faithfulness(),
        "answer_relevance": AnswerRelevance(),
        "context_precision": ContextPrecision(),
        "context_recall": ContextRecall(),
    }

    result = evaluate(dataset=eval_dataset, metrics=metrics)
    df = result.to_pandas()

    scores = {
        "faithfulness": float(df["faithfulness"].mean()),
        "answer_relevance": float(df["answer_relevance"].mean()),
        "context_precision": float(df["context_precision"].mean()),
        "context_recall": float(df["context_recall"].mean()),
    }
    scores["overall_score"] = sum(scores.values()) / len(scores)

    return scores


def print_report(scores: dict) -> None:
    """Display a formatted terminal report with bar charts.

    Args:
        scores: Dictionary of metric scores from run_evaluation().
    """
    if not scores:
        print("No evaluation results.")
        return

    print("\n" + "=" * 50)
    print("  RAGAS Evaluation Report")
    print("=" * 50)
    for metric, score in scores.items():
        if metric == "overall_score":
            continue
        bar = "█" * int(score * 20) + "░" * int((1 - score) * 20)
        print(f"  {metric:<25} {bar} {score:.3f}")
    print("=" * 50)
    print(f"  {'OVERALL':<25} {scores.get('overall_score', 0):.3f}")
    print("=" * 50 + "\n")
