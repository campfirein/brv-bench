"""Side-by-side comparison report generator.

Given two `BenchmarkReport` + `TelemetrySummary` pairs (one per config),
produces a markdown report shaped per the T6 spec:

  | Metric | Config A | Config B | Delta | Verdict |
  ...
  Decision: greenlight | red-light | discussion

The final ``Decision:`` line is computed deterministically from the
per-metric verdicts (no hand-tuning post-bench).

Verdict thresholds are hardcoded here for transparency at review time;
move to a co-located ``decision_criteria.yaml`` once we have a second
metric set that needs different gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from brv_bench.adapters.telemetry import TelemetrySummary
from brv_bench.types import BenchmarkReport, MetricResult

#: Verdict sentinels. Strings so they read naturally in the markdown
#: table and the final Decision line without further mapping.
DECISION_GREENLIGHT = "greenlight"
DECISION_REDLIGHT = "red-light"
DECISION_DISCUSSION = "discussion"

Decision = Literal["greenlight", "red-light", "discussion"]

#: Recall@5 verdict gates. Delta in percentage points (ratio difference).
#: A 5 pp improvement is meaningful retrieval-quality progress; a 5 pp
#: regression is meaningful loss. Tune via PR review once we have more
#: data points.
_RECALL_GREENLIGHT_PP = 0.05
_RECALL_REDLIGHT_PP = -0.05

#: Tolerance for floating-point boundary checks. Without this, subtraction
#: artefacts like `0.80 - 0.85 = -0.049999...` slip below the redlight gate
#: by one ULP and silently report `discussion`. Five basis points of
#: tolerance is well below the gate's measurement noise.
_FP_TOLERANCE = 1e-9

#: p99 latency verdict gates (relative change). +20% regression is
#: noticeable to users; -20% improvement is real headroom for richer
#: queries. Same tuning caveat as above.
_LATENCY_REDLIGHT_FRAC = 0.20
_LATENCY_GREENLIGHT_FRAC = -0.20

#: Token-usage verdict gates (relative change in total input tokens).
#: Higher tokens at equivalent recall = same answer at higher cost.
#: Treat as latency-shaped for the gate.
_TOKEN_REDLIGHT_FRAC = 0.20
_TOKEN_GREENLIGHT_FRAC = -0.20


@dataclass(frozen=True)
class ComparisonRow:
    """One row in the side-by-side table.

    `value_a` / `value_b` are pre-formatted strings (e.g. ``"40.0%"``,
    ``"4000.0 ms"``, ``"50,000"``, or the sentinel ``"unknown"``).
    `delta` is also pre-formatted (e.g. ``"+5.0 pp"``, ``"-30.0%"``, or
    ``"unknown"`` when one side is missing).
    """

    name: str
    value_a: str
    value_b: str
    delta: str
    verdict: Decision


@dataclass(frozen=True)
class ComparisonReport:
    rows: tuple[ComparisonRow, ...]
    decision: Decision
    markdown: str


def build_comparison_report(
    *,
    report_a: BenchmarkReport,
    report_b: BenchmarkReport,
    telemetry_a: TelemetrySummary,
    telemetry_b: TelemetrySummary,
    label_a: str = "Config A",
    label_b: str = "Config B",
) -> ComparisonReport:
    """Assemble the side-by-side report.

    Args:
        report_a / report_b: BenchmarkReport from each config's `evaluate` run.
        telemetry_a / telemetry_b: Aggregated `TelemetryReader.read()` output
            for each config's project data dir.
        label_a / label_b: Column labels — defaults are "Config A" / "Config B".

    Returns:
        A `ComparisonReport` with structured rows, the overall decision,
        and the rendered markdown ready to write to disk.
    """
    rows: list[ComparisonRow] = []

    # 1. Primary retrieval metrics (one row per metric in report_a; report_b
    #    is matched by name). Verdict only applied to known-gate metrics.
    rows.extend(_metric_rows(report_a, report_b))

    # 2. Latency tiers (p50/p95/p99) for totalMs, searchMs, llmMs.
    rows.extend(_latency_rows(telemetry_a, telemetry_b))

    # 3. Token totals — query side, then curate side.
    rows.extend(_token_rows(telemetry_a, telemetry_b))

    # 4. Format mode and tier distribution.
    rows.extend(_format_and_tier_rows(telemetry_a, telemetry_b))

    decision = decide_overall(rows)
    markdown = _render_markdown(
        rows=rows,
        decision=decision,
        report_a=report_a,
        report_b=report_b,
        telemetry_a=telemetry_a,
        telemetry_b=telemetry_b,
        label_a=label_a,
        label_b=label_b,
    )
    return ComparisonReport(rows=tuple(rows), decision=decision, markdown=markdown)


def decide_overall(rows: Iterable[ComparisonRow]) -> Decision:
    """Aggregate per-row verdicts into the overall decision.

    Logic (matches the CI-gate model):
      - any red-light → red-light
      - else any non-greenlight (discussion) → discussion
      - else greenlight
    """
    rows_list = list(rows)
    if not rows_list:
        return DECISION_DISCUSSION
    if any(r.verdict == DECISION_REDLIGHT for r in rows_list):
        return DECISION_REDLIGHT
    if all(r.verdict == DECISION_GREENLIGHT for r in rows_list):
        return DECISION_GREENLIGHT
    return DECISION_DISCUSSION


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _metric_rows(a: BenchmarkReport, b: BenchmarkReport) -> list[ComparisonRow]:
    metrics_a = {m.name: m for m in a.metrics}
    metrics_b = {m.name: m for m in b.metrics}
    rows: list[ComparisonRow] = []
    for name in sorted(set(metrics_a) | set(metrics_b)):
        ma = metrics_a.get(name)
        mb = metrics_b.get(name)
        rows.append(_metric_row(name, ma, mb))
    return rows


def _metric_row(name: str, ma: MetricResult | None, mb: MetricResult | None) -> ComparisonRow:
    label = (ma or mb).label if (ma or mb) else name
    va = _fmt_metric(ma)
    vb = _fmt_metric(mb)
    if ma is None or mb is None or not isinstance(ma.value, (int, float)) or not isinstance(mb.value, (int, float)):
        return ComparisonRow(name=label, value_a=va, value_b=vb, delta="unknown", verdict=DECISION_DISCUSSION)
    delta_val = mb.value - ma.value
    delta_str = _fmt_pp(delta_val)
    verdict = _verdict_for_metric(name, delta_val)
    return ComparisonRow(name=label, value_a=va, value_b=vb, delta=delta_str, verdict=verdict)


def _verdict_for_metric(name: str, delta: float) -> Decision:
    """Verdict only applied to recall@K + nDCG-shaped retrieval metrics where
    "higher is better" is the universal direction. Other metrics (LLM Judge,
    MRR) get DISCUSSION until we have concrete gate criteria for them."""
    key = name.lower()
    if "recall@" in key:
        if delta >= _RECALL_GREENLIGHT_PP - _FP_TOLERANCE:
            return DECISION_GREENLIGHT
        if delta <= _RECALL_REDLIGHT_PP + _FP_TOLERANCE:
            return DECISION_REDLIGHT
        return DECISION_DISCUSSION
    # MRR / NDCG / precision / judge → defer to discussion until we have
    # an agreed-upon gate. The bench surfaces the numbers in any case.
    return DECISION_DISCUSSION


def _latency_rows(a: TelemetrySummary, b: TelemetrySummary) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    for label, attr in (("totalMs", "total_ms"), ("searchMs", "search_ms"), ("llmMs", "llm_ms")):
        ta = getattr(a.latency, attr)
        tb = getattr(b.latency, attr)
        for p_name, p_attr in (("p50", "p50"), ("p95", "p95"), ("p99", "p99")):
            va = _fmt_ms(getattr(ta, p_attr)) if ta.count > 0 else "unknown"
            vb = _fmt_ms(getattr(tb, p_attr)) if tb.count > 0 else "unknown"
            if ta.count == 0 or tb.count == 0:
                rows.append(
                    ComparisonRow(
                        name=f"{p_name} {label}",
                        value_a=va,
                        value_b=vb,
                        delta="unknown",
                        verdict=DECISION_DISCUSSION,
                    )
                )
                continue
            a_val = getattr(ta, p_attr)
            b_val = getattr(tb, p_attr)
            frac = (b_val - a_val) / a_val if a_val > 0 else 0.0
            delta = _fmt_pct(frac)
            verdict = DECISION_DISCUSSION
            # Only p99 totalMs is a verdict-gated metric per the spec.
            # p50/p95/p99 of other tiers surface for context but don't
            # contribute to the decision.
            if p_name == "p99" and label == "totalMs":
                if frac >= _LATENCY_REDLIGHT_FRAC - _FP_TOLERANCE:
                    verdict = DECISION_REDLIGHT
                elif frac <= _LATENCY_GREENLIGHT_FRAC + _FP_TOLERANCE:
                    verdict = DECISION_GREENLIGHT
            rows.append(
                ComparisonRow(name=f"{p_name} {label}", value_a=va, value_b=vb, delta=delta, verdict=verdict)
            )
    return rows


def _token_rows(a: TelemetrySummary, b: TelemetrySummary) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    # Query side first.
    rows.append(_token_row(a.tokens, b.tokens, "Input tokens (query)", "input_tokens", a.tokens.token_field_coverage, b.tokens.token_field_coverage))
    rows.append(_token_row(a.tokens, b.tokens, "Output tokens (query)", "output_tokens", a.tokens.token_field_coverage, b.tokens.token_field_coverage))
    rows.append(_token_row(a.tokens, b.tokens, "Cached input tokens (query)", "cached_input_tokens", a.tokens.token_field_coverage, b.tokens.token_field_coverage))
    # Curate side.
    rows.append(_token_row(a.curate_tokens, b.curate_tokens, "Input tokens (curate)", "input_tokens", a.curate_tokens.token_field_coverage, b.curate_tokens.token_field_coverage))
    rows.append(_token_row(a.curate_tokens, b.curate_tokens, "Output tokens (curate)", "output_tokens", a.curate_tokens.token_field_coverage, b.curate_tokens.token_field_coverage))
    return rows


def _token_row(
    a, b, label: str, attr: str, coverage_a: int, coverage_b: int  # type: ignore[no-untyped-def]
) -> ComparisonRow:
    val_a = getattr(a, attr)
    val_b = getattr(b, attr)
    str_a = "unknown" if coverage_a == 0 else f"{val_a:,}"
    str_b = "unknown" if coverage_b == 0 else f"{val_b:,}"
    if coverage_a == 0 or coverage_b == 0 or val_a == 0:
        return ComparisonRow(name=label, value_a=str_a, value_b=str_b, delta="unknown", verdict=DECISION_DISCUSSION)
    frac = (val_b - val_a) / val_a
    verdict = DECISION_DISCUSSION
    if "input tokens (query)" in label.lower():
        # Higher tokens at equivalent recall = same answer for more $.
        if frac >= _TOKEN_REDLIGHT_FRAC - _FP_TOLERANCE:
            verdict = DECISION_REDLIGHT
        elif frac <= _TOKEN_GREENLIGHT_FRAC + _FP_TOLERANCE:
            verdict = DECISION_GREENLIGHT
    return ComparisonRow(name=label, value_a=str_a, value_b=str_b, delta=_fmt_pct(frac), verdict=verdict)


def _format_and_tier_rows(a: TelemetrySummary, b: TelemetrySummary) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    # Format distribution — informational only.
    rows.append(
        ComparisonRow(
            name="Format counts (html / markdown / unknown)",
            value_a=f"{a.format_counts['html']} / {a.format_counts['markdown']} / {a.format_counts['unknown']}",
            value_b=f"{b.format_counts['html']} / {b.format_counts['markdown']} / {b.format_counts['unknown']}",
            delta="—",
            verdict=DECISION_DISCUSSION,
        )
    )
    # Tier distribution.
    rows.append(
        ComparisonRow(
            name="Tier distribution",
            value_a=_fmt_tier_counts(a.tier_counts),
            value_b=_fmt_tier_counts(b.tier_counts),
            delta="—",
            verdict=DECISION_DISCUSSION,
        )
    )
    return rows


def _fmt_tier_counts(counts: dict) -> str:
    # Sort: ints ascending first, then sentinel strings.
    int_keys = sorted(k for k in counts if isinstance(k, int))
    str_keys = sorted(k for k in counts if not isinstance(k, int))
    parts = [f"T{k}={counts[k]}" for k in int_keys] + [f"{k}={counts[k]}" for k in str_keys]
    return ", ".join(parts) if parts else "—"


def _fmt_metric(m: MetricResult | None) -> str:
    if m is None:
        return "unknown"
    if not isinstance(m.value, (int, float)):
        return str(m.value)
    if m.unit == "ratio":
        return f"{m.value * 100:.1f}%"
    if m.unit == "s":
        return f"{m.value:.2f} s"
    return f"{m.value:.2f}"


def _fmt_ms(v: float) -> str:
    return f"{v:.0f} ms"


def _fmt_pp(delta: float) -> str:
    """Format a percentage-point delta for ratio metrics (Recall, NDCG)."""
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.1f} pp"


def _fmt_pct(frac: float) -> str:
    """Format a relative-change percentage (for latency, tokens)."""
    sign = "+" if frac >= 0 else ""
    return f"{sign}{frac * 100:.1f}%"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_markdown(
    *,
    rows: list[ComparisonRow],
    decision: Decision,
    report_a: BenchmarkReport,
    report_b: BenchmarkReport,
    telemetry_a: TelemetrySummary,
    telemetry_b: TelemetrySummary,
    label_a: str,
    label_b: str,
) -> str:
    lines: list[str] = []
    lines.append("# Side-by-side benchmark report")
    lines.append("")
    lines.append(f"- Dataset: `{report_a.name}`")
    lines.append(f"- Memory system: `{report_a.memory_system}`")
    lines.append(
        f"- {label_a}: {report_a.query_count} queries, "
        f"{telemetry_a.curate_count} curate ops, {report_a.duration_ms / 1000:.1f}s wall"
    )
    lines.append(
        f"- {label_b}: {report_b.query_count} queries, "
        f"{telemetry_b.curate_count} curate ops, {report_b.duration_ms / 1000:.1f}s wall"
    )
    lines.append("")
    lines.append(f"| Metric | {label_a} | {label_b} | Delta | Verdict |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row.name} | {row.value_a} | {row.value_b} | {row.delta} | {row.verdict} |"
        )
    lines.append("")
    lines.append(f"Decision: {decision}")
    return "\n".join(lines) + "\n"
