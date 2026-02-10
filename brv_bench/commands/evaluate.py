"""Evaluate command — measure retrieval quality against ground truth."""

import time

from brv_bench.adapters.base import RetrievalAdapter
from brv_bench.metrics.base import Metric
from brv_bench.types import (
    BenchmarkDataset,
    BenchmarkReport,
    GroundTruthEntry,
    QueryExecution,
)


async def run_queries(
    adapter: RetrievalAdapter,
    entries: tuple[GroundTruthEntry, ...],
    limit: int,
) -> list[tuple[QueryExecution, GroundTruthEntry]]:
    """Execute all ground-truth queries through the adapter.

    Returns a list of (execution_result, ground_truth) pairs.
    """
    pairs: list[tuple[QueryExecution, GroundTruthEntry]] = []
    for entry in entries:
        execution = await adapter.query(entry.query, limit)
        pairs.append((execution, entry))
    return pairs


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
) -> BenchmarkReport:
    """Run the full evaluation pipeline.

    1. Setup adapter
    2. Reset (cold start)
    3. Run queries
    4. Compute metrics
    5. Teardown adapter

    Args:
        adapter: Retrieval backend to benchmark.
        dataset: Benchmark dataset with corpus and ground truth.
        metrics: Metrics to compute.
        limit: Max results per query.

    Returns:
        BenchmarkReport with all computed metrics.
    """
    await adapter.setup()
    try:
        await adapter.reset()

        start = time.perf_counter()
        pairs = await run_queries(
            adapter, dataset.entries, limit
        )
        duration_ms = (time.perf_counter() - start) * 1000

        metric_results = compute_metrics(metrics, pairs)

        return BenchmarkReport(
            name=dataset.name,
            context_tree_docs=len(dataset.corpus),
            query_count=len(dataset.entries),
            duration_ms=duration_ms,
            metrics=tuple(metric_results),
        )
    finally:
        await adapter.teardown()
