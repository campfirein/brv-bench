"""Tests for brv_bench.types — all shared dataclasses."""

import pytest

from brv_bench.types import (
    BenchmarkReport,
    GroundTruthDataset,
    GroundTruthEntry,
    MetricResult,
    Percentiles,
    QueryExecution,
    SearchResult,
)

# =============================================================================


class TestGroundTruthEntry:
    def test_create_with_required_fields(self):
        entry = GroundTruthEntry(
            query="How does auth work?",
            expected_docs=("auth/oauth.md",),
        )
        assert entry.query == "How does auth work?"
        assert entry.expected_docs == ("auth/oauth.md",)
        assert entry.category == "unspecified"

    def test_create_with_category(self):
        entry = GroundTruthEntry(
            query="OAuth 2.0",
            expected_docs=("auth/oauth.md",),
            category="exact",
        )
        assert entry.category == "exact"

    def test_multiple_expected_docs(self):
        entry = GroundTruthEntry(
            query="authentication",
            expected_docs=("auth/oauth.md", "auth/jwt.md"),
        )
        assert len(entry.expected_docs) == 2

    def test_frozen(self):
        entry = GroundTruthEntry(query="test", expected_docs=())
        with pytest.raises(AttributeError):
            entry.query = "modified"


# =============================================================================


class TestGroundTruthDataset:
    def test_create(self):
        entries = (
            GroundTruthEntry(query="q1", expected_docs=("a.md",)),
            GroundTruthEntry(query="q2", expected_docs=("b.md",)),
        )
        dataset = GroundTruthDataset(name="test-dataset", entries=entries)
        assert dataset.name == "test-dataset"
        assert len(dataset.entries) == 2

    def test_empty_dataset(self):
        dataset = GroundTruthDataset(name="empty", entries=())
        assert len(dataset.entries) == 0


# =============================================================================


class TestSearchResult:
    def test_create(self):
        result = SearchResult(
            path="auth/oauth.md",
            title="OAuth 2.0",
            score=12.5,
            excerpt="OAuth flow uses PKCE...",
        )
        assert result.path == "auth/oauth.md"
        assert result.score == 12.5

    def test_frozen(self):
        result = SearchResult(path="a.md", title="A", score=1.0, excerpt="")
        with pytest.raises(AttributeError):
            result.score = 99.0


# =============================================================================


class TestQueryExecution:
    def test_create(self):
        results = (
            SearchResult(path="a.md", title="A", score=10.0, excerpt="..."),
            SearchResult(path="b.md", title="B", score=5.0, excerpt="..."),
        )
        execution = QueryExecution(
            query="test query",
            results=results,
            total_found=2,
            duration_ms=150.5,
        )
        assert execution.query == "test query"
        assert len(execution.results) == 2
        assert execution.duration_ms == 150.5

    def test_empty_results(self):
        execution = QueryExecution(
            query="no results",
            results=(),
            total_found=0,
            duration_ms=10.0,
        )
        assert len(execution.results) == 0


# =============================================================================


class TestMetricResult:
    def test_create_without_percentiles(self):
        result = MetricResult(
            name="precision@5",
            label="Precision@5",
            value=0.82,
            unit="ratio",
        )
        assert result.value == 0.82
        assert result.percentiles is None

    def test_create_with_percentiles(self):
        result = MetricResult(
            name="cold-latency",
            label="Cold Latency",
            value=120.5,
            unit="ms",
            percentiles=Percentiles(p50=100.0, p95=200.0, p99=350.0),
        )
        assert result.percentiles is not None
        assert result.percentiles.p50 == 100.0
        assert result.percentiles.p99 == 350.0


# =============================================================================


class TestBenchmarkReport:
    def test_create(self):
        report = BenchmarkReport(
            name="test-report",
            context_tree_docs=47,
            query_count=30,
            duration_ms=5000.0,
            metrics=(
                MetricResult(
                    name="mrr", label="MRR", value=0.76, unit="ratio"
                ),
            ),
        )
        assert report.name == "test-report"
        assert report.context_tree_docs == 47
        assert len(report.metrics) == 1
