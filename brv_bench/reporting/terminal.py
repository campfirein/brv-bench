"""Terminal report formatter."""

import os
from pathlib import Path

from brv_bench.types import BenchmarkReport

# Metrics displayed as decimal (0.xx) rather than percentage
_DECIMAL_METRICS = {"mrr", "diversity"}


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

    quality = [m for m in report.metrics if m.percentiles is None]
    latency = [m for m in report.metrics if m.percentiles is not None]

    if quality:
        lines.append("")
        lines.append("  Quality Metrics:")
        lines.append("  " + "-" * 40)
        max_lbl = max(len(m.label) for m in quality)
        for m in quality:
            if any(k in m.name.lower() for k in _DECIMAL_METRICS):
                val = f"{m.value:.2f}"
            else:
                val = f"{m.value:.1%}"
            lines.append(f"  {m.label:<{max_lbl}}  {val:>10}")

    if latency:
        lines.append("")
        lines.append("  Latency Metrics:")
        lines.append("  " + "-" * 56)
        lines.append(
            f"  {'Metric':<21}{'Mean':>8}{'p50':>9}{'p95':>9}{'p99':>9}"
        )
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

    lines.append(SEP)
    return "\n".join(lines)


def save_summary(report: BenchmarkReport, txt_path: Path) -> None:
    """Save the formatted report summary to a .txt file."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(format_report(report) + os.linesep)
