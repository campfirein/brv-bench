"""Evaluate command — measure retrieval quality against ground truth."""

import json
import time
from pathlib import Path

from tqdm import tqdm

from brv_bench.adapters.base import RetrievalAdapter
from brv_bench.metrics.base import Metric
from brv_bench.types import (
    BenchmarkDataset,
    BenchmarkReport,
    GroundTruthEntry,
    QueryExecution,
)


def _pair_to_dict(
    execution: QueryExecution,
    entry: GroundTruthEntry,
) -> dict:
    """Serialize a single (execution, ground_truth) pair."""
    return {
        "query": entry.query,
        "category": entry.category,
        "expected_doc_ids": list(entry.expected_doc_ids),
        "expected_answer": entry.expected_answer,
        "answer": execution.answer,
        "result_doc_ids": [r.path for r in execution.results],
        "duration_ms": execution.duration_ms,
    }


async def run_queries(
    adapter: RetrievalAdapter,
    entries: tuple[GroundTruthEntry, ...],
    limit: int,
    output_path: Path | None = None,
) -> list[tuple[QueryExecution, GroundTruthEntry]]:
    """Execute all ground-truth queries through the adapter.

    Results are saved incrementally to output_path (if provided)
    so partial progress survives crashes.

    Returns a list of (execution_result, ground_truth) pairs.
    """
    pairs: list[tuple[QueryExecution, GroundTruthEntry]] = []

    # Resume from existing partial results
    start_idx = 0
    if output_path and output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        existing = data.get("pairs", [])
        start_idx = len(existing)
        if start_idx > 0:
            # Reconstruct pairs from saved data (for metric computation)
            for saved in existing:
                gt = entries[len(pairs)]
                qe = QueryExecution(
                    query=saved["query"],
                    results=tuple(),
                    total_found=len(saved.get("result_doc_ids", [])),
                    duration_ms=saved["duration_ms"],
                    answer=saved.get("answer"),
                )
                pairs.append((qe, gt))

    remaining = entries[start_idx:]
    for entry in tqdm(
        remaining,
        desc="Querying",
        unit="query",
        initial=start_idx,
        total=len(entries),
    ):
        execution = await adapter.query(entry.query, limit)
        pairs.append((execution, entry))

        if output_path:
            _save_partial(output_path, pairs)

    return pairs


def _save_partial(
    output_path: Path,
    pairs: list[tuple[QueryExecution, GroundTruthEntry]],
) -> None:
    """Save current pairs to disk for crash recovery."""
    data = {
        "status": "in_progress",
        "completed": len(pairs),
        "pairs": [
            _pair_to_dict(qe, gt) for qe, gt in pairs
        ],
    }
    output_path.write_text(json.dumps(data, indent=2))


def compute_metrics(
    metrics: list[Metric],
    pairs: list[tuple[QueryExecution, GroundTruthEntry]],
) -> list:
    """Run all metrics over the query-execution pairs."""
    results = []
    for metric in metrics:
        results.extend(metric.compute(pairs))
    return results


async def evaluate(
    adapter: RetrievalAdapter,
    dataset: BenchmarkDataset,
    metrics: list[Metric],
    limit: int = 10,
    output_path: Path | None = None,
) -> BenchmarkReport:
    """Run the full evaluation pipeline.

    1. Setup adapter
    2. Reset (cold start)
    3. Run queries (saved incrementally to output_path)
    4. Compute metrics
    5. Save final report to output_path
    6. Teardown adapter

    Args:
        adapter: Retrieval backend to benchmark.
        dataset: Benchmark dataset with corpus and ground truth.
        metrics: Metrics to compute.
        limit: Max results per query.
        output_path: Optional path to save results JSON.

    Returns:
        BenchmarkReport with all computed metrics.
    """
    await adapter.setup()
    try:
        await adapter.reset()

        start = time.perf_counter()
        pairs = await run_queries(
            adapter, dataset.entries, limit, output_path
        )
        duration_ms = (time.perf_counter() - start) * 1000

        metric_results = compute_metrics(metrics, pairs)

        report = BenchmarkReport(
            name=dataset.name,
            context_tree_docs=len(dataset.corpus),
            query_count=len(dataset.entries),
            duration_ms=duration_ms,
            metrics=tuple(metric_results),
        )

        if output_path:
            _save_report(output_path, report, pairs)

        return report
    finally:
        await adapter.teardown()


def _save_report(
    output_path: Path,
    report: BenchmarkReport,
    pairs: list[tuple[QueryExecution, GroundTruthEntry]],
) -> None:
    """Save the final report with metrics and per-query results."""
    data = {
        "status": "completed",
        "benchmark": report.name,
        "context_tree_docs": report.context_tree_docs,
        "query_count": report.query_count,
        "duration_ms": report.duration_ms,
        "metrics": {
            m.name: {
                "label": m.label,
                "value": m.value,
                "unit": m.unit,
            }
            for m in report.metrics
        },
        "pairs": [
            _pair_to_dict(qe, gt) for qe, gt in pairs
        ],
    }
    output_path.write_text(json.dumps(data, indent=2))
