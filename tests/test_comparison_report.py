"""Tests for the side-by-side comparison report generator.

Per T6: the report has shape ``metric | A | B | delta | verdict`` with
a final line ``Decision: greenlight | red-light | discussion`` computed
deterministically from the per-metric verdicts. The decision logic is
the load-bearing detail — reviewers should not hand-tune it post-bench.
"""

from __future__ import annotations

import pytest

from brv_bench.adapters.telemetry import (
    LatencyBreakdown,
    LatencyTier,
    TelemetrySummary,
    TokenTotals,
)
from brv_bench.reporting.comparison import (
    DECISION_GREENLIGHT,
    DECISION_REDLIGHT,
    DECISION_DISCUSSION,
    Decision,
    ComparisonRow,
    build_comparison_report,
    decide_overall,
)
from brv_bench.types import BenchmarkReport, MetricResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _metric(name: str, value: float, label: str | None = None) -> MetricResult:
    return MetricResult(name=name, label=label or name, value=value, unit="ratio")


def _report(metrics_by_name: dict[str, float]) -> BenchmarkReport:
    """Build a BenchmarkReport with the given primary metrics."""
    return BenchmarkReport(
        name="locomo",
        memory_system="brv-cli",
        context_tree_docs=10,
        query_count=5,
        duration_ms=120_000.0,
        metrics=tuple(_metric(k, v) for k, v in metrics_by_name.items()),
        category_breakdown=(),
    )


def _telemetry(
    *,
    queries: int = 5,
    curates: int = 10,
    input_t: int = 50_000,
    output_t: int = 5000,
    cached_t: int = 0,
    p50_total: float = 1000.0,
    p95_total: float = 3000.0,
    p99_total: float = 4000.0,
    format_html: int = 5,
    format_md: int = 0,
    token_coverage: int = 5,
) -> TelemetrySummary:
    """Build a TelemetrySummary with deterministic numbers."""
    return TelemetrySummary(
        query_count=queries,
        curate_count=curates,
        tier_counts={2: queries // 2, 3: queries - (queries // 2)},
        format_counts={"html": format_html, "markdown": format_md, "unknown": 0},
        tokens=TokenTotals(
            input_tokens=input_t,
            output_tokens=output_t,
            cached_input_tokens=cached_t,
            cache_creation_tokens=0,
            token_field_coverage=token_coverage,
        ),
        curate_tokens=TokenTotals(
            input_tokens=200_000,
            output_tokens=10_000,
            cached_input_tokens=0,
            cache_creation_tokens=0,
            token_field_coverage=curates,
        ),
        latency=LatencyBreakdown(
            total_ms=LatencyTier(count=queries, p50=p50_total, p95=p95_total, p99=p99_total, mean=1500),
            search_ms=LatencyTier(count=queries, p50=10, p95=50, p99=80, mean=20),
            llm_ms=LatencyTier(count=queries, p50=900, p95=2900, p99=3900, mean=1400),
        ),
    )


# ---------------------------------------------------------------------------
# Per-metric verdict logic
# ---------------------------------------------------------------------------


class TestRecallVerdict:
    """Recall@5 improvement >= 5 percentage points → greenlight.
    Regression < -5 pp → red-light. Otherwise neutral / discussion."""

    def test_recall_improvement_5pp_is_greenlight(self):
        rows = build_comparison_report(
            report_a=_report({"recall@5": 0.40}),
            report_b=_report({"recall@5": 0.45}),
            telemetry_a=_telemetry(),
            telemetry_b=_telemetry(),
        ).rows
        recall_row = next(r for r in rows if "recall@5" in r.name.lower())
        assert recall_row.verdict == DECISION_GREENLIGHT

    def test_recall_regression_5pp_is_redlight(self):
        rows = build_comparison_report(
            report_a=_report({"recall@5": 0.85}),
            report_b=_report({"recall@5": 0.80}),
            telemetry_a=_telemetry(),
            telemetry_b=_telemetry(),
        ).rows
        recall_row = next(r for r in rows if "recall@5" in r.name.lower())
        assert recall_row.verdict == DECISION_REDLIGHT

    def test_small_recall_change_under_threshold_is_neutral(self):
        rows = build_comparison_report(
            report_a=_report({"recall@5": 0.80}),
            report_b=_report({"recall@5": 0.82}),  # +2 pp
            telemetry_a=_telemetry(),
            telemetry_b=_telemetry(),
        ).rows
        recall_row = next(r for r in rows if "recall@5" in r.name.lower())
        assert recall_row.verdict == DECISION_DISCUSSION


class TestLatencyVerdict:
    """p99 totalMs regression > 20% → red-light. Improvement >= 20% → greenlight.
    Within 20% either way → discussion."""

    def test_p99_regression_above_20pct_is_redlight(self):
        rows = build_comparison_report(
            report_a=_report({"recall@5": 0.5}),
            report_b=_report({"recall@5": 0.5}),
            telemetry_a=_telemetry(p99_total=1000),
            telemetry_b=_telemetry(p99_total=1300),  # +30%
        ).rows
        lat = next(r for r in rows if "p99" in r.name.lower() and "total" in r.name.lower())
        assert lat.verdict == DECISION_REDLIGHT

    def test_p99_improvement_above_20pct_is_greenlight(self):
        rows = build_comparison_report(
            report_a=_report({"recall@5": 0.5}),
            report_b=_report({"recall@5": 0.5}),
            telemetry_a=_telemetry(p99_total=1000),
            telemetry_b=_telemetry(p99_total=700),  # -30%
        ).rows
        lat = next(r for r in rows if "p99" in r.name.lower() and "total" in r.name.lower())
        assert lat.verdict == DECISION_GREENLIGHT

    def test_p99_within_band_is_discussion(self):
        rows = build_comparison_report(
            report_a=_report({"recall@5": 0.5}),
            report_b=_report({"recall@5": 0.5}),
            telemetry_a=_telemetry(p99_total=1000),
            telemetry_b=_telemetry(p99_total=1100),  # +10%
        ).rows
        lat = next(r for r in rows if "p99" in r.name.lower() and "total" in r.name.lower())
        assert lat.verdict == DECISION_DISCUSSION


# ---------------------------------------------------------------------------
# Decision aggregation (the final `Decision: ...` line)
# ---------------------------------------------------------------------------


class TestOverallDecision:
    """The final decision is a deterministic function of the per-metric
    verdicts. The reviewer reads it alongside the numbers — see T6 spec
    ('reduces room for post-hoc rationalization')."""

    def test_all_greenlight_yields_greenlight(self):
        decision = decide_overall(
            [
                ComparisonRow(name="Recall@5", value_a="x", value_b="x", delta="", verdict=DECISION_GREENLIGHT),
                ComparisonRow(name="p99 total", value_a="x", value_b="x", delta="", verdict=DECISION_GREENLIGHT),
            ]
        )
        assert decision == DECISION_GREENLIGHT

    def test_any_redlight_yields_redlight(self):
        # A single red-light dominates — same model as a CI gate.
        decision = decide_overall(
            [
                ComparisonRow(name="Recall@5", value_a="x", value_b="x", delta="", verdict=DECISION_GREENLIGHT),
                ComparisonRow(name="p99 total", value_a="x", value_b="x", delta="", verdict=DECISION_REDLIGHT),
                ComparisonRow(name="MRR", value_a="x", value_b="x", delta="", verdict=DECISION_GREENLIGHT),
            ]
        )
        assert decision == DECISION_REDLIGHT

    def test_mixed_green_and_discussion_yields_discussion(self):
        # No red, but at least one not-greenlit → human-call territory.
        decision = decide_overall(
            [
                ComparisonRow(name="Recall@5", value_a="x", value_b="x", delta="", verdict=DECISION_GREENLIGHT),
                ComparisonRow(name="p99 total", value_a="x", value_b="x", delta="", verdict=DECISION_DISCUSSION),
            ]
        )
        assert decision == DECISION_DISCUSSION

    def test_all_discussion_yields_discussion(self):
        decision = decide_overall(
            [
                ComparisonRow(name="A", value_a="x", value_b="x", delta="", verdict=DECISION_DISCUSSION),
                ComparisonRow(name="B", value_a="x", value_b="x", delta="", verdict=DECISION_DISCUSSION),
            ]
        )
        assert decision == DECISION_DISCUSSION

    def test_empty_rows_yields_discussion(self):
        # Edge case — no comparable metrics. Don't claim a verdict.
        decision = decide_overall([])
        assert decision == DECISION_DISCUSSION


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


class TestMarkdownOutput:
    def test_report_includes_summary_header(self):
        out = build_comparison_report(
            report_a=_report({"recall@5": 0.4, "mrr": 0.3}),
            report_b=_report({"recall@5": 0.9, "mrr": 0.8}),
            telemetry_a=_telemetry(),
            telemetry_b=_telemetry(),
        ).markdown
        assert "# Side-by-side benchmark report" in out
        assert "Config A" in out
        assert "Config B" in out

    def test_report_includes_markdown_table_with_columns(self):
        out = build_comparison_report(
            report_a=_report({"recall@5": 0.4}),
            report_b=_report({"recall@5": 0.9}),
            telemetry_a=_telemetry(),
            telemetry_b=_telemetry(),
        ).markdown
        # Header row + alignment row
        assert "| Metric |" in out
        assert "| Config A |" in out
        assert "| Config B |" in out
        assert "| Delta |" in out
        assert "| Verdict |" in out
        # Alignment row
        assert "|---|" in out or "| --- |" in out

    def test_report_includes_final_decision_line(self):
        report = build_comparison_report(
            report_a=_report({"recall@5": 0.4}),
            report_b=_report({"recall@5": 0.9}),
            telemetry_a=_telemetry(),
            telemetry_b=_telemetry(),
        )
        assert report.markdown.rstrip().splitlines()[-1].startswith("Decision:")

    def test_decision_value_matches_aggregated_verdict(self):
        report = build_comparison_report(
            report_a=_report({"recall@5": 0.4}),
            report_b=_report({"recall@5": 0.9}),  # +50 pp → greenlight
            telemetry_a=_telemetry(p99_total=1000),
            telemetry_b=_telemetry(p99_total=900),  # within 20% band
        )
        final = report.markdown.rstrip().splitlines()[-1]
        assert final == f"Decision: {report.decision}"

    def test_token_totals_appear_in_report(self):
        out = build_comparison_report(
            report_a=_report({"recall@5": 0.4}),
            report_b=_report({"recall@5": 0.5}),
            telemetry_a=_telemetry(input_t=50_000, output_t=5000),
            telemetry_b=_telemetry(input_t=40_000, output_t=4000),
        ).markdown
        # Token totals show somewhere
        assert "50000" in out or "50,000" in out
        assert "40000" in out or "40,000" in out


# ---------------------------------------------------------------------------
# Telemetry-absent tolerance (T6 spec: "unknown" not crash)
# ---------------------------------------------------------------------------


class TestTelemetryAbsentTolerance:
    def test_reports_unknown_for_zero_token_coverage(self):
        # No entries reported tokens — bench shows "unknown", doesn't claim 0.
        report = build_comparison_report(
            report_a=_report({"recall@5": 0.5}),
            report_b=_report({"recall@5": 0.5}),
            telemetry_a=_telemetry(token_coverage=0, input_t=0, output_t=0),
            telemetry_b=_telemetry(token_coverage=0, input_t=0, output_t=0),
        )
        # The token row(s) should label as "unknown"
        token_rows = [r for r in report.rows if "token" in r.name.lower()]
        assert any(r.value_a == "unknown" or r.value_b == "unknown" for r in token_rows)

    def test_partial_telemetry_one_side_known_other_unknown(self):
        report = build_comparison_report(
            report_a=_report({"recall@5": 0.5}),
            report_b=_report({"recall@5": 0.5}),
            telemetry_a=_telemetry(token_coverage=5, input_t=10_000),
            telemetry_b=_telemetry(token_coverage=0, input_t=0),
        )
        # When only one side has telemetry, both verdict and delta should
        # surface "unknown" rather than claim a meaningful comparison.
        token_input = next(r for r in report.rows if r.name == "Input tokens (query)")
        assert token_input.value_a == "10,000" or token_input.value_a == "10000"
        assert token_input.value_b == "unknown"
        assert token_input.delta == "unknown"
        # Verdict must be neutral / discussion when comparison impossible
        assert token_input.verdict == DECISION_DISCUSSION
