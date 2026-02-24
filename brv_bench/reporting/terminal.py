"""Terminal report formatter."""

import os
from pathlib import Path

from brv_bench.metrics import primary_metrics
from brv_bench.types import BenchmarkReport, MetricResult

# Metrics displayed as decimal (0.xx) rather than percentage
_DECIMAL_METRICS = {"mrr", "ndcg", "llm-judge"}

# Primary metric IDs derived from the registry (not hardcoded)
_PRIMARY_IDS: set[str] = {m.id for m in primary_metrics()}


def _format_value(m: MetricResult, query_count: int | None = None) -> str:
    """Format a single metric value for display."""
    if any(k in m.name.lower() for k in _DECIMAL_METRICS):
        base = f"{m.value:.2f}"
        if "llm-judge" in m.name.lower() and query_count is not None:
            correct = round(m.value * query_count)
            return f"{base} ({correct}/{query_count})"
        return base
    return f"{m.value:.1%}"


def _format_quality_section(
    metrics: tuple[MetricResult, ...] | list[MetricResult],
    query_count: int | None = None,
) -> list[str]:
    """Render quality metrics as label/value rows."""
    quality = [m for m in metrics if m.percentiles is None]
    if not quality:
        return []
    lines: list[str] = []
    max_lbl = max(len(m.label) for m in quality)
    for m in quality:
        lines.append(f"  {m.label:<{max_lbl}}  {_format_value(m, query_count):>10}")
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

    elapsed_s = report.duration_ms / 1000
    duration_str = f"{elapsed_s / 60:.1f}min" if elapsed_s >= 60 else f"{elapsed_s:.1f}s"

    lines: list[str] = [SEP]
    lines.append(f"  {'Dataset:':<15}{report.name}")
    lines.append(f"  {'Memory System:':<15}{report.memory_system}")
    lines.append(
        f"  {'Context tree:':<15}{report.context_tree_docs} documents"
    )
    lines.append(f"  {'Queries:':<15}{report.query_count}")
    lines.append(f"  {'Duration:':<15}{duration_str}")
    lines.append(THIN)

    # --- Split quality metrics into primary / diagnostic ---
    quality = [m for m in report.metrics if m.percentiles is None]
    primary = [m for m in quality if m.name in _PRIMARY_IDS]
    diagnostic = [m for m in quality if m.name not in _PRIMARY_IDS]

    if primary:
        lines.append("")
        lines.append("  Primary Metrics:")
        lines.append("  " + "-" * 40)
        lines.extend(_format_quality_section(primary, report.query_count))

    if diagnostic:
        lines.append("")
        lines.append("  Diagnostic Metrics:")
        lines.append("  " + "-" * 40)
        lines.extend(_format_quality_section(diagnostic, report.query_count))

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
            cat_quality = [m for m in cr.metrics if m.percentiles is None]
            cat_primary = [m for m in cat_quality if m.name in _PRIMARY_IDS]
            cat_diag = [m for m in cat_quality if m.name not in _PRIMARY_IDS]
            if cat_primary:
                lines.append("  " + "-" * 40)
                lines.extend(_format_quality_section(cat_primary, cr.query_count))
            if cat_diag:
                lines.append("  " + "-" * 40)
                lines.extend(_format_quality_section(cat_diag, cr.query_count))

    lines.append(SEP)
    return "\n".join(lines)


def save_summary(report: BenchmarkReport, txt_path: Path) -> None:
    """Save the formatted report summary to a .txt file."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(format_report(report) + os.linesep)
