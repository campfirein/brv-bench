"""Terminal report formatter."""

import os
from pathlib import Path

from brv_bench.types import BenchmarkReport, MetricResult

# Metrics displayed as decimal (0.xx) rather than percentage
_DECIMAL_METRICS = {"mrr", "diversity"}


def _format_value(m: MetricResult) -> str:
    """Format a single metric value for display."""
    if any(k in m.name.lower() for k in _DECIMAL_METRICS):
        return f"{m.value:.2f}"
    return f"{m.value:.1%}"


def _format_quality_section(
    metrics: tuple[MetricResult, ...] | list[MetricResult],
) -> list[str]:
    """Render quality metrics as label/value rows."""
    quality = [m for m in metrics if m.percentiles is None]
    if not quality:
        return []
    lines: list[str] = []
    max_lbl = max(len(m.label) for m in quality)
    for m in quality:
        lines.append(f"  {m.label:<{max_lbl}}  {_format_value(m):>10}")
    return lines


def _format_latency_section(
    metrics: tuple[MetricResult, ...] | list[MetricResult],
) -> list[str]:
    """Render latency metrics with percentile columns."""
    latency = [m for m in metrics if m.percentiles is not None]
    if not latency:
        return []
    lines: list[str] = []
    lines.append("")
    lines.append("  Latency Metrics:")
    lines.append("  " + "-" * 56)
    lines.append(f"  {'Metric':<21}{'Mean':>8}{'p50':>9}{'p95':>9}{'p99':>9}")
    lines.append("  " + "-" * 56)
    for m in latency:
        p = m.percentiles
        mean_s = f"{m.value:.1f}s"
        p50_s = f"{p.p50:.1f}s"
        p95_s = f"{p.p95:.1f}s"
        p99_s = f"{p.p99:.1f}s"
        lines.append(
            f"  {m.label:<21}{mean_s:>8}{p50_s:>9}{p95_s:>9}{p99_s:>9}"
        )
    return lines


def format_report(report: BenchmarkReport) -> str:
    """Format benchmark report as a boxed terminal table."""
    W = 64
    SEP = "=" * W
    THIN = "-" * W

    lines: list[str] = [SEP]
    lines.append(f"  {'Dataset:':<15}{report.name}")
    lines.append(f"  {'Memory System:':<15}{report.memory_system}")
    lines.append(
        f"  {'Context tree:':<15}{report.context_tree_docs} documents"
    )
    lines.append(f"  {'Queries:':<15}{report.query_count}")
    lines.append(THIN)

    # --- Overall quality metrics ---
    quality_rows = _format_quality_section(report.metrics)
    if quality_rows:
        lines.append("")
        lines.append("  Quality Metrics (Overall):")
        lines.append("  " + "-" * 40)
        lines.extend(quality_rows)

    # Latency metrics
    lines.extend(_format_latency_section(report.metrics))

    # Per-category breakdown
    if report.category_breakdown:
        lines.append("")
        lines.append(THIN)
        lines.append("  Per-Category Breakdown:")
        lines.append(THIN)
        for cr in report.category_breakdown:
            lines.append("")
            lines.append(f"  {cr.category} ({cr.query_count} queries):")
            lines.append("  " + "-" * 40)
            lines.extend(_format_quality_section(cr.metrics))

    lines.append(SEP)
    return "\n".join(lines)


def save_summary(report: BenchmarkReport, txt_path: Path) -> None:
    """Save the formatted report summary to a .txt file."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(format_report(report) + os.linesep)
