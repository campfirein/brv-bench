"""Tests for brv_bench.metrics — all P0 metric implementations."""

import pytest

from brv_bench.metrics import (
    LatencyMetric,
    MeanReciprocalRank,
    PrecisionAtK,
    RecallAtK,
    ResultDiversity,
    default_metrics,
)
from brv_bench.types import GroundTruthEntry, QueryExecution, SearchResult

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _result(path: str, score: float = 1.0, excerpt: str = "") -> SearchResult:
    return SearchResult(path=path, title=path, score=score, excerpt=excerpt)


def _execution(
    query: str,
    paths: list[str],
    duration_ms: float = 100.0,
    excerpts: list[str] | None = None,
) -> QueryExecution:
    if excerpts is None:
        excerpts = [""] * len(paths)
    results = tuple(
        _result(p, score=10.0 - i, excerpt=excerpts[i]) for i, p in enumerate(paths)
    )
    return QueryExecution(
        query=query,
        results=results,
        total_found=len(results),
        duration_ms=duration_ms,
    )


def _truth(
    query: str, expected: list[str], category: str = "exact"
) -> GroundTruthEntry:
    return GroundTruthEntry(
        query=query, expected_docs=tuple(expected), category=category
    )


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_perfect_precision(self):
        pairs = [
            (_execution("q", ["a.md", "b.md"]), _truth("q", ["a.md", "b.md"])),
        ]
        results = PrecisionAtK(5).compute(pairs)
        assert len(results) == 1
        assert results[0].name == "precision@5"
        assert results[0].value == 1.0

    def test_zero_precision(self):
        pairs = [
            (_execution("q", ["x.md", "y.md"]), _truth("q", ["a.md"])),
        ]
        results = PrecisionAtK(5).compute(pairs)
        assert results[0].value == 0.0

    def test_partial_precision(self):
        pairs = [
            (
                _execution("q", ["a.md", "x.md", "y.md", "z.md"]),
                _truth("q", ["a.md", "b.md"]),
            ),
        ]
        results = PrecisionAtK(5).compute(pairs)
        # 1 hit / min(5, 2) = 1/2 = 0.5
        assert results[0].value == 0.5

    def test_k_limits_results(self):
        # 5 results, only the 6th would be relevant — but K=5 cuts it off
        pairs = [
            (
                _execution("q", ["x1.md", "x2.md", "x3.md", "x4.md", "x5.md", "a.md"]),
                _truth("q", ["a.md"]),
            ),
        ]
        results = PrecisionAtK(5).compute(pairs)
        assert results[0].value == 0.0

    def test_empty_pairs(self):
        results = PrecisionAtK(5).compute([])
        assert results[0].value == 0.0

    def test_multiple_queries_averaged(self):
        pairs = [
            (_execution("q1", ["a.md"]), _truth("q1", ["a.md"])),  # 1.0
            (_execution("q2", ["x.md"]), _truth("q2", ["a.md"])),  # 0.0
        ]
        results = PrecisionAtK(5).compute(pairs)
        assert results[0].value == 0.5


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_recall(self):
        pairs = [
            (_execution("q", ["a.md", "b.md", "x.md"]), _truth("q", ["a.md", "b.md"])),
        ]
        results = RecallAtK(5).compute(pairs)
        assert results[0].value == 1.0

    def test_zero_recall(self):
        pairs = [
            (_execution("q", ["x.md"]), _truth("q", ["a.md", "b.md"])),
        ]
        results = RecallAtK(5).compute(pairs)
        assert results[0].value == 0.0

    def test_partial_recall(self):
        pairs = [
            (_execution("q", ["a.md", "x.md"]), _truth("q", ["a.md", "b.md"])),
        ]
        results = RecallAtK(5).compute(pairs)
        # 1 found / 2 relevant = 0.5
        assert results[0].value == 0.5

    def test_empty_expected_docs(self):
        pairs = [
            (_execution("q", ["a.md"]), _truth("q", [])),
        ]
        results = RecallAtK(5).compute(pairs)
        assert results[0].value == 1.0  # vacuously complete

    def test_empty_pairs(self):
        results = RecallAtK(5).compute([])
        assert results[0].value == 0.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


class TestMeanReciprocalRank:
    def test_first_result_relevant(self):
        pairs = [
            (_execution("q", ["a.md", "x.md"]), _truth("q", ["a.md"])),
        ]
        results = MeanReciprocalRank().compute(pairs)
        assert results[0].value == 1.0

    def test_second_result_relevant(self):
        pairs = [
            (_execution("q", ["x.md", "a.md"]), _truth("q", ["a.md"])),
        ]
        results = MeanReciprocalRank().compute(pairs)
        assert results[0].value == 0.5

    def test_no_relevant_result(self):
        pairs = [
            (_execution("q", ["x.md", "y.md"]), _truth("q", ["a.md"])),
        ]
        results = MeanReciprocalRank().compute(pairs)
        assert results[0].value == 0.0

    def test_multiple_queries_averaged(self):
        pairs = [
            (_execution("q1", ["a.md"]), _truth("q1", ["a.md"])),  # RR = 1.0
            (_execution("q2", ["x.md", "a.md"]), _truth("q2", ["a.md"])),  # RR = 0.5
        ]
        results = MeanReciprocalRank().compute(pairs)
        assert results[0].value == 0.75

    def test_empty_pairs(self):
        results = MeanReciprocalRank().compute([])
        assert results[0].value == 0.0


# ---------------------------------------------------------------------------
# Result Diversity
# ---------------------------------------------------------------------------


class TestResultDiversity:
    def test_identical_excerpts(self):
        """Identical excerpts = zero diversity."""
        pairs = [
            (
                _execution(
                    "q",
                    ["a.md", "b.md"],
                    excerpts=["the same text here", "the same text here"],
                ),
                _truth("q", ["a.md"]),
            ),
        ]
        results = ResultDiversity(5).compute(pairs)
        assert results[0].value == pytest.approx(0.0, abs=0.01)

    def test_completely_different_excerpts(self):
        """No word overlap = maximum diversity."""
        pairs = [
            (
                _execution(
                    "q",
                    ["a.md", "b.md"],
                    excerpts=[
                        "authentication oauth tokens refresh",
                        "database migration schema indexes",
                    ],
                ),
                _truth("q", ["a.md"]),
            ),
        ]
        results = ResultDiversity(5).compute(pairs)
        assert results[0].value == 1.0

    def test_single_result(self):
        """Single result = maximally diverse (nothing to compare)."""
        pairs = [
            (_execution("q", ["a.md"], excerpts=["some text"]), _truth("q", ["a.md"])),
        ]
        results = ResultDiversity(5).compute(pairs)
        assert results[0].value == 1.0

    def test_empty_pairs(self):
        results = ResultDiversity(5).compute([])
        assert results[0].value == 0.0


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


class TestLatencyMetric:
    def test_single_query(self):
        pairs = [
            (_execution("q", ["a.md"], duration_ms=150.0), _truth("q", ["a.md"])),
        ]
        results = LatencyMetric("Cold Latency", "cold-latency").compute(pairs)
        assert results[0].value == 150.0
        assert results[0].unit == "ms"
        assert results[0].percentiles is not None
        assert results[0].percentiles.p50 == 150.0

    def test_multiple_queries_percentiles(self):
        durations = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        pairs = [
            (_execution(f"q{i}", ["a.md"], duration_ms=d), _truth(f"q{i}", ["a.md"]))
            for i, d in enumerate(durations)
        ]
        results = LatencyMetric("Cold Latency", "cold-latency").compute(pairs)
        assert results[0].value == 55.0  # mean
        assert results[0].percentiles is not None
        assert results[0].percentiles.p50 == 50.0
        assert results[0].percentiles.p95 == 100.0

    def test_empty_pairs(self):
        results = LatencyMetric("Cold Latency", "cold-latency").compute([])
        assert results[0].value == 0.0
        assert results[0].percentiles is not None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestDefaultMetrics:
    def test_returns_all_p0_metrics(self):
        metrics = default_metrics()
        ids = {m.id for m in metrics}
        assert "precision@5" in ids
        assert "precision@10" in ids
        assert "recall@5" in ids
        assert "recall@10" in ids
        assert "mrr" in ids
        assert "diversity@5" in ids
        assert "cold-latency" in ids

    def test_count(self):
        assert len(default_metrics()) == 7
