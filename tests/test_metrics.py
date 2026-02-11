"""Tests for brv_bench.metrics."""

import pytest

from brv_bench.metrics import (
    ExactMatch,
    F1Score,
    LatencyMetric,
    MeanReciprocalRank,
    PrecisionAtK,
    RecallAtK,
    ResultDiversity,
    default_metrics,
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
# ResultDiversity
# ----------------------------------------------------------------


class TestResultDiversity:
    def test_identical_excerpts(self):
        results = tuple(
            SearchResult(
                path=f"{i}.md",
                title=f"{i}",
                score=1.0,
                excerpt="same words here",
            )
            for i in range(3)
        )
        qe = QueryExecution(
            query="q",
            results=results,
            total_found=3,
            duration_ms=1.0,
        )
        pairs = [(qe, _gt("q", []))]
        [result] = ResultDiversity(3).compute(pairs)
        assert result.value == 0.0

    def test_completely_different_excerpts(self):
        results = (
            SearchResult(
                path="1.md",
                title="1",
                score=1.0,
                excerpt="alpha beta gamma",
            ),
            SearchResult(
                path="2.md",
                title="2",
                score=0.9,
                excerpt="delta epsilon zeta",
            ),
        )
        qe = QueryExecution(
            query="q",
            results=results,
            total_found=2,
            duration_ms=1.0,
        )
        pairs = [(qe, _gt("q", []))]
        [result] = ResultDiversity(5).compute(pairs)
        assert result.value == 1.0

    def test_single_result(self):
        qe = QueryExecution(
            query="q",
            results=(
                SearchResult(
                    path="1.md",
                    title="1",
                    score=1.0,
                    excerpt="text",
                ),
            ),
            total_found=1,
            duration_ms=1.0,
        )
        pairs = [(qe, _gt("q", []))]
        [result] = ResultDiversity(5).compute(pairs)
        assert result.value == 1.0

    def test_empty_pairs(self):
        [result] = ResultDiversity(5).compute([])
        assert result.value == 0.0


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
# F1Score
# ----------------------------------------------------------------


class TestF1Score:
    def test_perfect_match(self):
        pairs = [
            (
                _qe("q", [], answer="the cat sat on the mat"),
                _gt("q", [], expected_answer="the cat sat on the mat"),
            )
        ]
        [result] = F1Score().compute(pairs)
        assert result.value == 1.0

    def test_zero_match(self):
        pairs = [
            (
                _qe("q", [], answer="alpha beta"),
                _gt("q", [], expected_answer="gamma delta"),
            )
        ]
        [result] = F1Score().compute(pairs)
        assert result.value == 0.0

    def test_partial_overlap(self):
        pairs = [
            (
                _qe("q", [], answer="cat dog"),
                _gt("q", [], expected_answer="cat bird"),
            )
        ]
        [result] = F1Score().compute(pairs)
        # common=1 (cat), pred=2, gold=2 → P=0.5 R=0.5 → F1=0.5
        assert result.value == pytest.approx(0.5)

    def test_normalization(self):
        pairs = [
            (
                _qe("q", [], answer="The Cat!"),
                _gt("q", [], expected_answer="a cat"),
            )
        ]
        [result] = F1Score().compute(pairs)
        assert result.value == 1.0

    def test_skips_none_answer(self):
        pairs = [
            (_qe("q", []), _gt("q", [], expected_answer="answer")),
        ]
        [result] = F1Score().compute(pairs)
        assert result.value == 0.0

    def test_skips_none_expected(self):
        pairs = [
            (_qe("q", [], answer="answer"), _gt("q", [])),
        ]
        [result] = F1Score().compute(pairs)
        assert result.value == 0.0

    def test_empty_pairs(self):
        [result] = F1Score().compute([])
        assert result.value == 0.0

    def test_averaging(self):
        pairs = [
            (
                _qe("q1", [], answer="cat"),
                _gt("q1", [], expected_answer="cat"),
            ),
            (
                _qe("q2", [], answer="alpha"),
                _gt("q2", [], expected_answer="beta"),
            ),
        ]
        [result] = F1Score().compute(pairs)
        # (1.0 + 0.0) / 2
        assert result.value == pytest.approx(0.5)


# ----------------------------------------------------------------
# ExactMatch
# ----------------------------------------------------------------


class TestExactMatch:
    def test_perfect_match(self):
        pairs = [
            (
                _qe("q", [], answer="Paris"),
                _gt("q", [], expected_answer="Paris"),
            )
        ]
        [result] = ExactMatch().compute(pairs)
        assert result.value == 1.0

    def test_zero_match(self):
        pairs = [
            (
                _qe("q", [], answer="London"),
                _gt("q", [], expected_answer="Paris"),
            )
        ]
        [result] = ExactMatch().compute(pairs)
        assert result.value == 0.0

    def test_normalization(self):
        pairs = [
            (
                _qe("q", [], answer="The Paris!"),
                _gt("q", [], expected_answer="a paris"),
            )
        ]
        [result] = ExactMatch().compute(pairs)
        assert result.value == 1.0

    def test_partial_is_not_exact(self):
        pairs = [
            (
                _qe("q", [], answer="Paris France"),
                _gt("q", [], expected_answer="Paris"),
            )
        ]
        [result] = ExactMatch().compute(pairs)
        assert result.value == 0.0

    def test_skips_none_answer(self):
        pairs = [
            (_qe("q", []), _gt("q", [], expected_answer="answer")),
        ]
        [result] = ExactMatch().compute(pairs)
        assert result.value == 0.0

    def test_skips_none_expected(self):
        pairs = [
            (_qe("q", [], answer="answer"), _gt("q", [])),
        ]
        [result] = ExactMatch().compute(pairs)
        assert result.value == 0.0

    def test_empty_pairs(self):
        [result] = ExactMatch().compute([])
        assert result.value == 0.0

    def test_averaging(self):
        pairs = [
            (
                _qe("q1", [], answer="Paris"),
                _gt("q1", [], expected_answer="Paris"),
            ),
            (
                _qe("q2", [], answer="London"),
                _gt("q2", [], expected_answer="Paris"),
            ),
        ]
        [result] = ExactMatch().compute(pairs)
        assert result.value == pytest.approx(0.5)


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
        assert "mrr" in ids
        assert "diversity@5" in ids
        assert "cold-latency" in ids
        assert "f1" in ids
        assert "exact-match" in ids

    def test_count(self):
        assert len(default_metrics()) == 9
