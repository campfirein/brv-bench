"""Tests for brv_bench.types."""

import pytest

from brv_bench.types import (
    BenchmarkDataset,
    BenchmarkReport,
    CorpusDocument,
    GroundTruthEntry,
    MetricResult,
    Percentiles,
    QueryExecution,
    SearchResult,
)

# ----------------------------------------------------------------
# CorpusDocument
# ----------------------------------------------------------------


class TestCorpusDocument:
    def test_create(self):
        doc = CorpusDocument(
            doc_id="auth_oauth",
            content="OAuth implementation details",
            source="session_1",
        )
        assert doc.doc_id == "auth_oauth"
        assert doc.content == "OAuth implementation details"
        assert doc.source == "session_1"

    def test_default_source(self):
        doc = CorpusDocument(doc_id="x", content="content")
        assert doc.source == ""

    def test_frozen(self):
        doc = CorpusDocument(doc_id="x", content="content")
        with pytest.raises(AttributeError):
            doc.doc_id = "y"  # type: ignore[misc]


# ----------------------------------------------------------------
# GroundTruthEntry
# ----------------------------------------------------------------


class TestGroundTruthEntry:
    def test_create_with_required_fields(self):
        entry = GroundTruthEntry(
            query="How does auth work?",
            expected_doc_ids=("auth/oauth.md",),
        )
        assert entry.query == "How does auth work?"
        assert entry.expected_doc_ids == ("auth/oauth.md",)
        assert entry.category == "unspecified"
        assert entry.expected_answer is None

    def test_create_with_all_fields(self):
        entry = GroundTruthEntry(
            query="How does auth work?",
            expected_doc_ids=("auth/oauth.md",),
            category="single-hop",
            expected_answer="Uses OAuth 2.0 with PKCE",
        )
        assert entry.category == "single-hop"
        assert entry.expected_answer == "Uses OAuth 2.0 with PKCE"

    def test_multiple_expected_doc_ids(self):
        entry = GroundTruthEntry(
            query="Auth",
            expected_doc_ids=("auth/oauth.md", "auth/jwt.md"),
        )
        assert len(entry.expected_doc_ids) == 2

    def test_frozen(self):
        entry = GroundTruthEntry(query="test", expected_doc_ids=())
        with pytest.raises(AttributeError):
            entry.query = "new"  # type: ignore[misc]


# ----------------------------------------------------------------
# BenchmarkDataset
# ----------------------------------------------------------------


class TestBenchmarkDataset:
    def test_create(self):
        corpus = (
            CorpusDocument(doc_id="a", content="aaa"),
            CorpusDocument(doc_id="b", content="bbb"),
        )
        entries = (GroundTruthEntry(query="q1", expected_doc_ids=("a",)),)
        dataset = BenchmarkDataset(name="test", corpus=corpus, entries=entries)
        assert dataset.name == "test"
        assert len(dataset.corpus) == 2
        assert len(dataset.entries) == 1

    def test_empty_dataset(self):
        dataset = BenchmarkDataset(name="empty", corpus=(), entries=())
        assert len(dataset.corpus) == 0
        assert len(dataset.entries) == 0


# ----------------------------------------------------------------
# SearchResult
# ----------------------------------------------------------------


class TestSearchResult:
    def test_create(self):
        r = SearchResult(path="a.md", title="A", score=0.9, excerpt="...")
        assert r.path == "a.md"
        assert r.score == 0.9

    def test_frozen(self):
        r = SearchResult(path="a.md", title="A", score=0.9, excerpt="...")
        with pytest.raises(AttributeError):
            r.path = "b.md"  # type: ignore[misc]


# ----------------------------------------------------------------
# QueryExecution
# ----------------------------------------------------------------


class TestQueryExecution:
    def test_create(self):
        qe = QueryExecution(
            query="test",
            results=(
                SearchResult(
                    path="a.md",
                    title="A",
                    score=0.9,
                    excerpt="...",
                ),
            ),
            total_found=1,
            duration_ms=5.0,
        )
        assert qe.query == "test"
        assert len(qe.results) == 1

    def test_empty_results(self):
        qe = QueryExecution(
            query="test",
            results=(),
            total_found=0,
            duration_ms=1.0,
        )
        assert qe.total_found == 0


# ----------------------------------------------------------------
# MetricResult + Percentiles
# ----------------------------------------------------------------


class TestMetricResult:
    def test_create_without_percentiles(self):
        mr = MetricResult(
            name="precision@5",
            label="Precision@5",
            value=0.82,
            unit="ratio",
        )
        assert mr.percentiles is None

    def test_create_with_percentiles(self):
        mr = MetricResult(
            name="cold-latency",
            label="Cold Latency",
            value=1.2,
            unit="ms",
            percentiles=Percentiles(p50=1.1, p95=2.3, p99=3.1),
        )
        assert mr.percentiles is not None
        assert mr.percentiles.p50 == 1.1


# ----------------------------------------------------------------
# BenchmarkReport
# ----------------------------------------------------------------


class TestBenchmarkReport:
    def test_create(self):
        report = BenchmarkReport(
            name="test",
            memory_system="brv-cli",
            context_tree_docs=47,
            query_count=30,
            duration_ms=5000.0,
            metrics=(),
        )
        assert report.name == "test"
        assert report.memory_system == "brv-cli"
        assert report.context_tree_docs == 47
