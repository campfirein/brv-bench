"""Tests for brv_bench.metrics."""

import pytest

from brv_bench.metrics import (
    LatencyMetric,
    MeanReciprocalRank,
    NDCGAtK,
    PrecisionAtK,
    RecallAtK,
    default_metrics,
    diagnostic_metrics,
    primary_metrics,
)
from brv_bench.types import (
    GroundTruthEntry,
    Percentiles,
    QueryExecution,
    SearchResult,
)

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _qe(
    query: str,
    paths: list[str],
    duration_ms: float = 5.0,
    answer: str | None = None,
) -> QueryExecution:
    return QueryExecution(
        query=query,
        results=tuple(
            SearchResult(
                path=p,
                title=p,
                score=1.0 - i * 0.1,
                excerpt=f"excerpt for {p}",
            )
            for i, p in enumerate(paths)
        ),
        total_found=len(paths),
        duration_ms=duration_ms,
        answer=answer,
    )


def _gt(
    query: str,
    expected: list[str],
    category: str = "unspecified",
    expected_answer: str | None = None,
) -> GroundTruthEntry:
    return GroundTruthEntry(
        query=query,
        expected_doc_ids=tuple(expected),
        category=category,
        expected_answer=expected_answer,
    )


# ----------------------------------------------------------------
# Precision@K
# ----------------------------------------------------------------


class TestPrecisionAtK:
    def test_perfect_precision(self):
        pairs = [(_qe("q", ["a.md", "b.md"]), _gt("q", ["a.md", "b.md"]))]
        [result] = PrecisionAtK(5).compute(pairs)
        assert result.value == 1.0

    def test_zero_precision(self):
        pairs = [(_qe("q", ["x.md", "y.md"]), _gt("q", ["a.md", "b.md"]))]
        [result] = PrecisionAtK(5).compute(pairs)
        assert result.value == 0.0

    def test_partial_precision(self):
        pairs = [(_qe("q", ["a.md", "x.md"]), _gt("q", ["a.md", "b.md"]))]
        [result] = PrecisionAtK(5).compute(pairs)
        assert result.value == 0.5

    def test_k_limits_results(self):
        pairs = [
            (
                _qe("q", ["a.md", "b.md", "c.md", "d.md", "e.md"]),
                _gt("q", ["a.md"]),
            )
        ]
        [result] = PrecisionAtK(2).compute(pairs)
        # top-2 = [a.md, b.md], 1 hit / 2 = 0.5
        assert result.value == 0.5

    def test_empty_pairs(self):
        [result] = PrecisionAtK(5).compute([])
        assert result.value == 0.0

    def test_multiple_queries_averaged(self):
        pairs = [
            (_qe("q1", ["a.md"]), _gt("q1", ["a.md"])),
            (_qe("q2", ["x.md"]), _gt("q2", ["a.md"])),
        ]
        [result] = PrecisionAtK(5).compute(pairs)
        assert result.value == 0.5


# ----------------------------------------------------------------
# Recall@K
# ----------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_recall(self):
        pairs = [(_qe("q", ["a.md", "b.md"]), _gt("q", ["a.md", "b.md"]))]
        [result] = RecallAtK(5).compute(pairs)
        assert result.value == 1.0

    def test_zero_recall(self):
        pairs = [(_qe("q", ["x.md"]), _gt("q", ["a.md", "b.md"]))]
        [result] = RecallAtK(5).compute(pairs)
        assert result.value == 0.0

    def test_partial_recall(self):
        pairs = [(_qe("q", ["a.md", "x.md"]), _gt("q", ["a.md", "b.md"]))]
        [result] = RecallAtK(5).compute(pairs)
        assert result.value == 0.5

    def test_empty_expected_doc_ids(self):
        pairs = [(_qe("q", ["a.md"]), _gt("q", []))]
        [result] = RecallAtK(5).compute(pairs)
        assert result.value == 1.0

    def test_empty_pairs(self):
        [result] = RecallAtK(5).compute([])
        assert result.value == 0.0


# ----------------------------------------------------------------
# MRR
# ----------------------------------------------------------------


class TestMeanReciprocalRank:
    def test_first_result_relevant(self):
        pairs = [(_qe("q", ["a.md", "b.md"]), _gt("q", ["a.md"]))]
        [result] = MeanReciprocalRank().compute(pairs)
        assert result.value == 1.0

    def test_second_result_relevant(self):
        pairs = [(_qe("q", ["x.md", "a.md"]), _gt("q", ["a.md"]))]
        [result] = MeanReciprocalRank().compute(pairs)
        assert result.value == 0.5

    def test_no_relevant_result(self):
        pairs = [(_qe("q", ["x.md", "y.md"]), _gt("q", ["a.md"]))]
        [result] = MeanReciprocalRank().compute(pairs)
        assert result.value == 0.0

    def test_multiple_queries_averaged(self):
        pairs = [
            (_qe("q1", ["a.md"]), _gt("q1", ["a.md"])),
            (_qe("q2", ["x.md", "a.md"]), _gt("q2", ["a.md"])),
        ]
        [result] = MeanReciprocalRank().compute(pairs)
        assert result.value == 0.75

    def test_empty_pairs(self):
        [result] = MeanReciprocalRank().compute([])
        assert result.value == 0.0


# ----------------------------------------------------------------
# NDCG@K
# ----------------------------------------------------------------


class TestNDCGAtK:
    def test_perfect_ranking(self):
        # Both relevant docs at positions 0 and 1
        pairs = [
            (_qe("q", ["a.md", "b.md", "x.md"]), _gt("q", ["a.md", "b.md"]))
        ]
        [result] = NDCGAtK(5).compute(pairs)
        assert result.value == pytest.approx(1.0)

    def test_zero_ndcg(self):
        # No relevant docs in results
        pairs = [(_qe("q", ["x.md", "y.md"]), _gt("q", ["a.md", "b.md"]))]
        [result] = NDCGAtK(5).compute(pairs)
        assert result.value == 0.0

    def test_imperfect_ranking(self):
        # Relevant doc at position 1 instead of 0
        # DCG  = 0/log2(2) + 1/log2(3) = 1/log2(3)
        # IDCG = 1/log2(2) = 1.0
        # NDCG = (1/log2(3)) / 1.0
        import math

        pairs = [(_qe("q", ["x.md", "a.md"]), _gt("q", ["a.md"]))]
        [result] = NDCGAtK(5).compute(pairs)
        assert result.value == pytest.approx(1.0 / math.log2(3))

    def test_k_limits_results(self):
        # Relevant doc at position 2, but K=2 cuts it off
        pairs = [(_qe("q", ["x.md", "y.md", "a.md"]), _gt("q", ["a.md"]))]
        [result] = NDCGAtK(2).compute(pairs)
        assert result.value == 0.0

    def test_multiple_queries_averaged(self):
        pairs = [
            # Perfect: NDCG = 1.0
            (_qe("q1", ["a.md"]), _gt("q1", ["a.md"])),
            # Zero: NDCG = 0.0
            (_qe("q2", ["x.md"]), _gt("q2", ["a.md"])),
        ]
        [result] = NDCGAtK(5).compute(pairs)
        assert result.value == pytest.approx(0.5)

    def test_empty_pairs(self):
        [result] = NDCGAtK(5).compute([])
        assert result.value == 0.0

    def test_no_relevant_docs_in_ground_truth(self):
        # IDCG = 0 → NDCG = 0
        pairs = [(_qe("q", ["a.md"]), _gt("q", []))]
        [result] = NDCGAtK(5).compute(pairs)
        assert result.value == 0.0

    def test_id_includes_k(self):
        assert NDCGAtK(5).id == "ndcg@5"
        assert NDCGAtK(10).id == "ndcg@10"


# ----------------------------------------------------------------
# LatencyMetric
# ----------------------------------------------------------------


class TestLatencyMetric:
    def test_single_query(self):
        pairs = [(_qe("q", ["a.md"], duration_ms=100.0), _gt("q", ["a.md"]))]
        [result] = LatencyMetric("Cold Latency", "cold-latency").compute(pairs)
        assert result.value == 0.1
        assert result.unit == "s"
        assert result.percentiles is not None
        assert result.percentiles.p50 == 0.1

    def test_multiple_queries_percentiles(self):
        pairs = [
            (_qe("q1", ["a.md"], duration_ms=float(i)), _gt("q1", ["a.md"]))
            for i in range(1, 101)
        ]
        [result] = LatencyMetric("Cold", "cold").compute(pairs)
        assert result.percentiles is not None
        assert result.percentiles.p50 == pytest.approx(0.050, abs=0.001)
        assert result.percentiles.p95 == pytest.approx(0.095, abs=0.001)

    def test_empty_pairs(self):
        [result] = LatencyMetric("Cold", "cold").compute([])
        assert result.value == 0.0
        assert result.percentiles == Percentiles(p50=0.0, p95=0.0, p99=0.0)


# ----------------------------------------------------------------
# default_metrics
# ----------------------------------------------------------------


class TestDefaultMetrics:
    def test_returns_all_p0_metrics(self):
        metrics = default_metrics()
        ids = {m.id for m in metrics}
        assert "precision@5" in ids
        assert "precision@10" in ids
        assert "recall@5" in ids
        assert "recall@10" in ids
        assert "ndcg@5" in ids
        assert "ndcg@10" in ids
        assert "mrr" in ids
        assert "cold-latency" in ids

    def test_count(self):
        assert len(default_metrics()) == 8


class TestPrimaryMetrics:
    def test_contains_retrieval_quality_metrics(self):
        ids = {m.id for m in primary_metrics()}
        assert "precision@5" in ids
        assert "precision@10" in ids
        assert "recall@5" in ids
        assert "recall@10" in ids
        assert "ndcg@5" in ids
        assert "ndcg@10" in ids
        assert "mrr" in ids

    def test_count(self):
        assert len(primary_metrics()) == 7

    def test_no_diagnostic_ids(self):
        primary_ids = {m.id for m in primary_metrics()}
        diag_ids = {m.id for m in diagnostic_metrics()}
        assert primary_ids.isdisjoint(diag_ids)


class TestDiagnosticMetrics:
    def test_contains_latency_metric(self):
        ids = {m.id for m in diagnostic_metrics()}
        assert "cold-latency" in ids

    def test_count(self):
        assert len(diagnostic_metrics()) == 1

    def test_default_is_primary_plus_diagnostic(self):
        default_ids = [m.id for m in default_metrics()]
        primary_ids = [m.id for m in primary_metrics()]
        diag_ids = [m.id for m in diagnostic_metrics()]
        assert default_ids == primary_ids + diag_ids
