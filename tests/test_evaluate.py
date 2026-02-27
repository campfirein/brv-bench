"""Tests for the evaluate command."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from brv_bench.adapters.base import RetrievalAdapter
from brv_bench.commands.evaluate import (
    compute_category_breakdown,
    compute_metrics,
    evaluate,
    run_queries,
)
from brv_bench.metrics import default_metrics
from brv_bench.metrics.precision import PrecisionAtK
from brv_bench.types import (
    BenchmarkDataset,
    BenchmarkReport,
    CorpusDocument,
    GroundTruthEntry,
    QueryExecution,
    SearchResult,
)

# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------


def _make_adapter(
    results_map: dict[str, QueryExecution],
) -> RetrievalAdapter:
    """Create a mock adapter with pre-defined results."""
    adapter = AsyncMock(spec=RetrievalAdapter)
    adapter.name = "mock"
    adapter.supports_warm_latency = False

    async def mock_query(query: str, limit: int) -> QueryExecution:
        return results_map.get(
            query,
            QueryExecution(
                query=query,
                results=(),
                total_found=0,
                duration_ms=1.0,
            ),
        )

    adapter.query.side_effect = mock_query
    return adapter


def _perfect_result(query: str, docs: tuple[str, ...]) -> QueryExecution:
    return QueryExecution(
        query=query,
        results=tuple(
            SearchResult(
                path=d,
                title=d,
                score=1.0,
                excerpt=f"excerpt for {d}",
            )
            for d in docs
        ),
        total_found=len(docs),
        duration_ms=5.0,
    )


CORPUS = (
    CorpusDocument(doc_id="auth/oauth.md", content="OAuth details"),
    CorpusDocument(doc_id="db/schema.md", content="DB schema"),
    CorpusDocument(
        doc_id="db/migrations.md",
        content="DB migrations",
    ),
)

DATASET = BenchmarkDataset(
    name="test-dataset",
    corpus=CORPUS,
    entries=(
        GroundTruthEntry(
            query="How does auth work?",
            expected_doc_ids=("auth/oauth.md",),
            category="natural-language",
        ),
        GroundTruthEntry(
            query="Database schema",
            expected_doc_ids=(
                "db/schema.md",
                "db/migrations.md",
            ),
            category="exact",
        ),
    ),
)


# ----------------------------------------------------------------
# run_queries
# ----------------------------------------------------------------


class TestRunQueries:
    def test_returns_pairs_for_each_entry(self):
        results_map = {
            "How does auth work?": _perfect_result(
                "How does auth work?", ("auth/oauth.md",)
            ),
            "Database schema": _perfect_result(
                "Database schema",
                ("db/schema.md", "db/migrations.md"),
            ),
        }
        adapter = _make_adapter(results_map)
        pairs = asyncio.run(run_queries(adapter, DATASET.entries, limit=10))

        assert len(pairs) == 2
        assert pairs[0][1].query == "How does auth work?"
        assert pairs[1][1].query == "Database schema"

    def test_query_called_with_correct_limit(self):
        adapter = _make_adapter({})
        entries = (GroundTruthEntry(query="test", expected_doc_ids=("a.md",)),)
        asyncio.run(run_queries(adapter, entries, limit=5))
        adapter.query.assert_called_once_with("test", 5)

    def test_empty_entries(self):
        adapter = _make_adapter({})
        pairs = asyncio.run(run_queries(adapter, (), limit=10))
        assert pairs == []


# ----------------------------------------------------------------
# compute_metrics
# ----------------------------------------------------------------


class TestComputeMetrics:
    def test_returns_results_from_all_metrics(self):
        pairs = [
            (
                _perfect_result("q1", ("a.md",)),
                GroundTruthEntry(query="q1", expected_doc_ids=("a.md",)),
            ),
        ]
        metrics = [PrecisionAtK(5), PrecisionAtK(10)]
        results = compute_metrics(metrics, pairs)

        assert len(results) == 2
        assert results[0].name == "precision@5"
        assert results[1].name == "precision@10"

    def test_empty_metrics(self):
        pairs = [
            (
                _perfect_result("q1", ("a.md",)),
                GroundTruthEntry(query="q1", expected_doc_ids=("a.md",)),
            ),
        ]
        results = compute_metrics([], pairs)
        assert results == []

    def test_empty_pairs(self):
        results = compute_metrics([PrecisionAtK(5)], [])
        assert len(results) == 1
        assert results[0].value == 0.0


# ----------------------------------------------------------------
# compute_category_breakdown
# ----------------------------------------------------------------


class TestComputeCategoryBreakdown:
    def test_groups_by_category(self):
        pairs = [
            (
                _perfect_result("q1", ("a.md",)),
                GroundTruthEntry(
                    query="q1",
                    expected_doc_ids=("a.md",),
                    category="single-hop",
                ),
            ),
            (
                _perfect_result("q2", ("b.md",)),
                GroundTruthEntry(
                    query="q2",
                    expected_doc_ids=("b.md",),
                    category="temporal",
                ),
            ),
            (
                _perfect_result("q3", ("c.md",)),
                GroundTruthEntry(
                    query="q3",
                    expected_doc_ids=("c.md",),
                    category="single-hop",
                ),
            ),
        ]
        breakdown = compute_category_breakdown([PrecisionAtK(5)], pairs)

        assert len(breakdown) == 2
        # Alphabetically sorted
        assert breakdown[0].category == "single-hop"
        assert breakdown[0].query_count == 2
        assert breakdown[1].category == "temporal"
        assert breakdown[1].query_count == 1

    def test_metrics_computed_per_group(self):
        pairs = [
            # single-hop: perfect hit
            (
                _perfect_result("q1", ("a.md",)),
                GroundTruthEntry(
                    query="q1",
                    expected_doc_ids=("a.md",),
                    category="single-hop",
                ),
            ),
            # temporal: zero hit
            (
                _perfect_result("q2", ("x.md",)),
                GroundTruthEntry(
                    query="q2",
                    expected_doc_ids=("a.md",),
                    category="temporal",
                ),
            ),
        ]
        breakdown = compute_category_breakdown([PrecisionAtK(5)], pairs)

        single_hop = breakdown[0]
        assert single_hop.category == "single-hop"
        assert single_hop.metrics[0].value == 1.0

        temporal = breakdown[1]
        assert temporal.category == "temporal"
        assert temporal.metrics[0].value == 0.0

    def test_empty_pairs(self):
        breakdown = compute_category_breakdown([PrecisionAtK(5)], [])
        assert breakdown == ()

    def test_single_category(self):
        pairs = [
            (
                _perfect_result("q1", ("a.md",)),
                GroundTruthEntry(
                    query="q1",
                    expected_doc_ids=("a.md",),
                    category="multi-hop",
                ),
            ),
        ]
        breakdown = compute_category_breakdown([PrecisionAtK(5)], pairs)
        assert len(breakdown) == 1
        assert breakdown[0].category == "multi-hop"
        assert breakdown[0].query_count == 1


# ----------------------------------------------------------------
# evaluate (full pipeline)
# ----------------------------------------------------------------


class TestEvaluate:
    def test_full_pipeline(self):
        results_map = {
            "How does auth work?": _perfect_result(
                "How does auth work?", ("auth/oauth.md",)
            ),
            "Database schema": _perfect_result(
                "Database schema",
                ("db/schema.md", "db/migrations.md"),
            ),
        }
        adapter = _make_adapter(results_map)
        metrics = [PrecisionAtK(5)]

        report = asyncio.run(evaluate(adapter, DATASET, metrics, limit=10))

        assert isinstance(report, BenchmarkReport)
        assert report.name == "test-dataset"
        assert report.query_count == 2
        assert report.context_tree_docs == 3
        assert report.duration_ms > 0
        assert len(report.metrics) == 1
        assert report.metrics[0].name == "precision@5"
        assert report.metrics[0].value == 1.0

        # Category breakdown populated from DATASET categories
        assert len(report.category_breakdown) == 2
        cats = {cr.category for cr in report.category_breakdown}
        assert cats == {"exact", "natural-language"}

    def test_calls_setup_reset_teardown(self):
        adapter = _make_adapter({})
        dataset = BenchmarkDataset(name="empty", corpus=(), entries=())
        metrics = [PrecisionAtK(5)]

        asyncio.run(evaluate(adapter, dataset, metrics))

        adapter.setup.assert_awaited_once()
        adapter.reset.assert_awaited_once()
        adapter.teardown.assert_awaited_once()

    def test_teardown_called_on_error(self):
        adapter = _make_adapter({})
        adapter.reset.side_effect = RuntimeError("reset failed")
        dataset = BenchmarkDataset(name="err", corpus=(), entries=())

        with pytest.raises(RuntimeError, match="reset failed"):
            asyncio.run(evaluate(adapter, dataset, [PrecisionAtK(5)]))

        adapter.teardown.assert_awaited_once()

    def test_with_default_metrics(self):
        results_map = {
            "How does auth work?": _perfect_result(
                "How does auth work?", ("auth/oauth.md",)
            ),
            "Database schema": _perfect_result(
                "Database schema",
                ("db/schema.md", "db/migrations.md"),
            ),
        }
        adapter = _make_adapter(results_map)

        report = asyncio.run(
            evaluate(adapter, DATASET, default_metrics(), limit=10)
        )

        assert report.query_count == 2
        assert len(report.metrics) == 8
        metric_names = {m.name for m in report.metrics}
        assert "precision@5" in metric_names
        assert "recall@10" in metric_names
        assert "mrr" in metric_names
        assert "cold-latency" in metric_names
