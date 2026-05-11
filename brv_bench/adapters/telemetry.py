"""Telemetry consumer — reads brv's `query-log/*.json` and
`curate-log/*.json` files from a project's data dir and aggregates
token totals, latency percentiles (p50/p95/p99), tier distribution,
and format-mode counts.

Resilient by design: malformed JSON, non-dict roots, and missing fields
are tolerated. Missing fields are reported as `"unknown"` (per T6
spec — bench must not crash on older daemons that predate the
T5 telemetry producer).

Read paths:
    <project-data-dir>/query-log/qry-<ts>.json
    <project-data-dir>/curate-log/cur-<ts>.json

Path resolution lives in `brv_bench.brv_io.resolve_project_data_dir`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatencyTier:
    """Latency percentiles for a single timing tier (totalMs, searchMs, llmMs)."""

    count: int  #: How many entries contributed (omitted-field entries excluded)
    p50: float
    p95: float
    p99: float
    mean: float


@dataclass(frozen=True)
class LatencyBreakdown:
    """All three timing tiers brv emits on QueryLogEntry / CurateLogEntry."""

    total_ms: LatencyTier
    search_ms: LatencyTier
    llm_ms: LatencyTier


@dataclass(frozen=True)
class TokenTotals:
    """Sum of token fields across all entries that reported them.

    `token_field_coverage` is the number of entries with at least one
    token field present — when zero, the totals are uninformative
    (no entries had the new T5 fields) and the reporter labels them
    as `"unknown"` rather than `0`.
    """

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cache_creation_tokens: int
    token_field_coverage: int  #: count of entries that contributed at least one token field


@dataclass(frozen=True)
class TelemetrySummary:
    """Aggregated telemetry for one project's worth of brv activity.

    Produced by `TelemetryReader.read()`. Consumed by the comparison
    report generator which formats it into a side-by-side markdown
    table per the T6 spec.
    """

    query_count: int
    curate_count: int
    tier_counts: dict[Any, int]  #: tier int OR the sentinel string `"unknown"`
    format_counts: dict[str, int]  #: keys: `"html"`, `"markdown"`, `"unknown"`
    tokens: TokenTotals  #: aggregated across queries
    curate_tokens: TokenTotals  #: aggregated across curates
    latency: LatencyBreakdown


class TelemetryReader:
    """Reads brv telemetry logs from a project data directory.

    Construct with the absolute path returned by
    `resolve_project_data_dir(cwd)`. Call `read()` to get a single
    aggregated `TelemetrySummary`.

    Selective reads use the optional `since` parameter on `read()`
    (mtime cutoff in seconds-since-epoch) so a two-config bench can
    distinguish entries written during this run from leftovers of
    prior runs against the same project.
    """

    def __init__(self, project_data_dir: str | Path) -> None:
        self._dir = Path(project_data_dir)

    def read(self, *, since: float | None = None) -> TelemetrySummary:
        """Read and aggregate all entries (or only those newer than ``since``).

        Args:
            since: If set, only entries whose file mtime is >= this value
                are included. Used to isolate one config's run from
                prior runs against the same data directory.

        Returns:
            `TelemetrySummary` with token totals, percentiles, and counts.
            Returns an empty summary if neither log directory exists.
        """
        query_entries = self._read_dir(self._dir / "query-log", "qry", since=since)
        curate_entries = self._read_dir(self._dir / "curate-log", "cur", since=since)

        tier_counts = _count_tiers(query_entries)
        format_counts = _count_formats(query_entries)
        tokens = _sum_tokens(query_entries)
        curate_tokens = _sum_tokens(curate_entries)
        latency = _latency_breakdown(query_entries)

        return TelemetrySummary(
            query_count=len(query_entries),
            curate_count=len(curate_entries),
            tier_counts=tier_counts,
            format_counts=format_counts,
            tokens=tokens,
            curate_tokens=curate_tokens,
            latency=latency,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_dir(
        self,
        dir_path: Path,
        prefix: str,
        *,
        since: float | None,
    ) -> list[dict]:
        """Read every `<prefix>-*.json` in `dir_path`. Skip corrupt entries."""
        if not dir_path.is_dir():
            return []

        entries: list[dict] = []
        for f in sorted(dir_path.glob(f"{prefix}-*.json")):
            if since is not None and f.stat().st_mtime < since:
                continue
            try:
                with f.open() as fp:
                    payload = json.load(fp)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("skipping malformed telemetry file %s: %s", f, exc)
                continue
            if not isinstance(payload, dict):
                logger.warning("skipping non-dict telemetry root %s", f)
                continue
            entries.append(payload)
        return entries


# ---------------------------------------------------------------------------
# Aggregators (module-level so tests can call them directly if needed)
# ---------------------------------------------------------------------------


def _count_tiers(entries: list[dict]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for entry in entries:
        tier = entry.get("tier", "unknown")
        if not isinstance(tier, int):
            tier = "unknown"
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def _count_formats(entries: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {"html": 0, "markdown": 0, "unknown": 0}
    for entry in entries:
        fmt = entry.get("format")
        if fmt in ("html", "markdown"):
            out[fmt] += 1
        else:
            out["unknown"] += 1
    return out


def _sum_tokens(entries: list[dict]) -> TokenTotals:
    input_t = output_t = cached_t = cache_create_t = 0
    coverage = 0
    for entry in entries:
        had_any = False
        for src_key, target in (
            ("inputTokens", "input"),
            ("outputTokens", "output"),
            ("cachedInputTokens", "cached"),
            ("cacheCreationTokens", "cache_create"),
        ):
            val = entry.get(src_key)
            if not isinstance(val, (int, float)):
                continue
            had_any = True
            v = int(val)
            if target == "input":
                input_t += v
            elif target == "output":
                output_t += v
            elif target == "cached":
                cached_t += v
            elif target == "cache_create":
                cache_create_t += v
        if had_any:
            coverage += 1
    return TokenTotals(
        input_tokens=input_t,
        output_tokens=output_t,
        cached_input_tokens=cached_t,
        cache_creation_tokens=cache_create_t,
        token_field_coverage=coverage,
    )


def _percentiles(values: list[float]) -> LatencyTier:
    """Compute p50 / p95 / p99 + mean from a list of latency values.

    Uses linear-interpolation percentiles to match numpy's default,
    so reports are reproducible across the bench and any downstream
    analytics that re-derive percentiles from the same raw data.
    """
    if not values:
        return LatencyTier(count=0, p50=0.0, p95=0.0, p99=0.0, mean=0.0)
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    return LatencyTier(
        count=n,
        p50=_quantile(s, 0.50),
        p95=_quantile(s, 0.95),
        p99=_quantile(s, 0.99),
        mean=mean,
    )


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile (numpy default 'linear' / type 7)."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    idx = q * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def _latency_breakdown(entries: list[dict]) -> LatencyBreakdown:
    totals: list[float] = []
    searches: list[float] = []
    llms: list[float] = []
    for entry in entries:
        timing = entry.get("timing")
        if not isinstance(timing, dict):
            continue
        for key, bucket in (("totalMs", totals), ("searchMs", searches), ("llmMs", llms)):
            v = timing.get(key)
            if isinstance(v, (int, float)):
                bucket.append(float(v))
    return LatencyBreakdown(
        total_ms=_percentiles(totals),
        search_ms=_percentiles(searches),
        llm_ms=_percentiles(llms),
    )
