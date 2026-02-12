"""Tests for brv_bench.reporting.terminal and JSON report output."""

import json
from pathlib import Path

import pytest

from brv_bench.commands.evaluate import _pair_to_dict, _save_report
from brv_bench.metrics._judge.client import JudgeVerdict
from brv_bench.metrics.precision import PrecisionAtK
from brv_bench.reporting.terminal import format_report
from brv_bench.types import (
    BenchmarkReport,
    CategoryResult,
    GroundTruthEntry,
    MetricResult,
    Percentiles,
    QueryExecution,
    SearchResult,
)

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _make_report(
    metrics: tuple[MetricResult, ...],
    category_breakdown: tuple[CategoryResult, ...] = (),
) -> BenchmarkReport:
    return BenchmarkReport(
        name="test",
        memory_system="brv",
        context_tree_docs=10,
        query_count=5,
        duration_ms=1000,
        metrics=metrics,
        category_breakdown=category_breakdown,
    )


# ----------------------------------------------------------------
# Terminal reporter — primary / diagnostic split
# ----------------------------------------------------------------


class TestTerminalReporterSections:
    def test_primary_section_header_present(self):
        report = _make_report(
            metrics=(
                MetricResult(
                    name="precision@5",
                    label="Precision@5",
                    value=0.8,
                    unit="ratio",
                ),
            ),
        )
        text = format_report(report)
        assert "Primary Metrics:" in text

    def test_diagnostic_section_header_present(self):
        report = _make_report(
            metrics=(
                MetricResult(
                    name="f1",
                    label="F1 Score",
                    value=0.6,
                    unit="ratio",
                ),
            ),
        )
        text = format_report(report)
        assert "Diagnostic Metrics:" in text

    def test_primary_and_diagnostic_separated(self):
        report = _make_report(
            metrics=(
                MetricResult(
                    name="precision@5",
                    label="Precision@5",
                    value=0.8,
                    unit="ratio",
                ),
                MetricResult(
                    name="f1",
                    label="F1 Score",
                    value=0.6,
                    unit="ratio",
                ),
            ),
        )
        text = format_report(report)
        primary_pos = text.index("Primary Metrics:")
        diag_pos = text.index("Diagnostic Metrics:")
        assert primary_pos < diag_pos

    def test_no_primary_section_when_only_diagnostic(self):
        report = _make_report(
            metrics=(
                MetricResult(
                    name="exact-match",
                    label="Exact Match",
                    value=0.5,
                    unit="ratio",
                ),
            ),
        )
        text = format_report(report)
        assert "Primary Metrics:" not in text
        assert "Diagnostic Metrics:" in text

    def test_no_diagnostic_section_when_only_primary(self):
        report = _make_report(
            metrics=(
                MetricResult(
                    name="mrr",
                    label="MRR",
                    value=0.9,
                    unit="ratio",
                ),
            ),
        )
        text = format_report(report)
        assert "Primary Metrics:" in text
        assert "Diagnostic Metrics:" not in text

    def test_llm_judge_displayed_as_decimal(self):
        report = _make_report(
            metrics=(
                MetricResult(
                    name="llm-judge",
                    label="LLM Judge",
                    value=0.78,
                    unit="ratio",
                ),
            ),
        )
        text = format_report(report)
        assert "0.78" in text
        assert "78.0%" not in text

    def test_latency_section_separate(self):
        report = _make_report(
            metrics=(
                MetricResult(
                    name="cold-latency",
                    label="Cold Latency",
                    value=1.2,
                    unit="seconds",
                    percentiles=Percentiles(p50=1.0, p95=2.0, p99=3.0),
                ),
            ),
        )
        text = format_report(report)
        assert "Latency Metrics:" in text
        assert "Primary Metrics:" not in text
        assert "Diagnostic Metrics:" not in text


# ----------------------------------------------------------------
# Terminal reporter — per-category breakdown
# ----------------------------------------------------------------


class TestTerminalReporterCategoryBreakdown:
    def test_category_shows_primary_and_diagnostic(self):
        report = _make_report(
            metrics=(
                MetricResult(
                    name="precision@5",
                    label="Precision@5",
                    value=0.8,
                    unit="ratio",
                ),
            ),
            category_breakdown=(
                CategoryResult(
                    category="single-hop",
                    query_count=3,
                    metrics=(
                        MetricResult(
                            name="precision@5",
                            label="Precision@5",
                            value=0.9,
                            unit="ratio",
                        ),
                        MetricResult(
                            name="f1",
                            label="F1 Score",
                            value=0.7,
                            unit="ratio",
                        ),
                    ),
                ),
            ),
        )
        text = format_report(report)
        assert "single-hop (3 queries):" in text
        assert "Precision@5" in text
        assert "F1 Score" in text


# ----------------------------------------------------------------
# _pair_to_dict — judge verdict in JSON
# ----------------------------------------------------------------


class TestPairToDict:
    def test_without_verdict(self):
        qe = QueryExecution(
            query="q1",
            results=(
                SearchResult(
                    path="a.md", title="a", score=1.0, excerpt="text"
                ),
            ),
            total_found=1,
            duration_ms=5.0,
            answer="Paris",
        )
        gt = GroundTruthEntry(
            query="q1",
            expected_doc_ids=("a.md",),
            expected_answer="Paris",
        )
        d = _pair_to_dict(qe, gt)
        assert "judge_verdict" not in d
        assert d["query"] == "q1"
        assert d["answer"] == "Paris"

    def test_with_verdict(self):
        qe = QueryExecution(
            query="q1",
            results=(),
            total_found=0,
            duration_ms=5.0,
            answer="Paris",
        )
        gt = GroundTruthEntry(
            query="q1",
            expected_doc_ids=("a.md",),
            expected_answer="Paris",
        )
        verdict = JudgeVerdict(
            query="q1", is_correct=True, reasoning="Matches expected."
        )
        d = _pair_to_dict(qe, gt, verdict)
        assert "judge_verdict" in d
        assert d["judge_verdict"]["is_correct"] is True
        assert d["judge_verdict"]["reasoning"] == "Matches expected."


# ----------------------------------------------------------------
# _save_report — JSON report with judge verdicts
# ----------------------------------------------------------------


class TestSaveReportWithVerdicts:
    def test_report_includes_judge_verdicts(self, tmp_path: Path):
        from unittest.mock import MagicMock

        from brv_bench.metrics.llm_judge import LLMJudge

        # Build a mock LLMJudge with pre-set verdicts
        judge = MagicMock(spec=LLMJudge)
        judge.get_verdict.side_effect = lambda q: {
            "q1": JudgeVerdict(
                query="q1", is_correct=True, reasoning="correct"
            ),
            "q2": JudgeVerdict(
                query="q2", is_correct=False, reasoning="wrong"
            ),
        }.get(q)

        qe1 = QueryExecution(
            query="q1",
            results=(
                SearchResult(
                    path="a.md", title="a", score=1.0, excerpt="text"
                ),
            ),
            total_found=1,
            duration_ms=5.0,
            answer="Paris",
        )
        qe2 = QueryExecution(
            query="q2",
            results=(),
            total_found=0,
            duration_ms=3.0,
            answer="London",
        )
        gt1 = GroundTruthEntry(
            query="q1",
            expected_doc_ids=("a.md",),
            expected_answer="Paris",
        )
        gt2 = GroundTruthEntry(
            query="q2",
            expected_doc_ids=("b.md",),
            expected_answer="Berlin",
        )

        report = BenchmarkReport(
            name="test",
            memory_system="brv",
            context_tree_docs=2,
            query_count=2,
            duration_ms=1000,
            metrics=(
                MetricResult(
                    name="llm-judge",
                    label="LLM Judge",
                    value=0.5,
                    unit="ratio",
                ),
            ),
        )

        output_path = tmp_path / "report.json"
        _save_report(output_path, report, [(qe1, gt1), (qe2, gt2)], [judge])

        data = json.loads(output_path.read_text())
        pairs = data["pairs"]

        assert pairs[0]["judge_verdict"]["is_correct"] is True
        assert pairs[0]["judge_verdict"]["reasoning"] == "correct"
        assert pairs[1]["judge_verdict"]["is_correct"] is False
        assert pairs[1]["judge_verdict"]["reasoning"] == "wrong"

    def test_report_without_judge(self, tmp_path: Path):
        qe = QueryExecution(
            query="q1",
            results=(),
            total_found=0,
            duration_ms=5.0,
        )
        gt = GroundTruthEntry(query="q1", expected_doc_ids=("a.md",))
        report = BenchmarkReport(
            name="test",
            memory_system="brv",
            context_tree_docs=1,
            query_count=1,
            duration_ms=500,
            metrics=(
                MetricResult(
                    name="precision@5",
                    label="Precision@5",
                    value=0.0,
                    unit="ratio",
                ),
            ),
        )

        output_path = tmp_path / "report.json"
        metrics = [PrecisionAtK(5)]
        _save_report(output_path, report, [(qe, gt)], metrics)

        data = json.loads(output_path.read_text())
        assert "judge_verdict" not in data["pairs"][0]
