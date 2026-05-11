"""Tests for `TelemetryReader` — reads `qry-*.json` and `cur-*.json` from a
brv project's data directory and aggregates token totals, latency
percentiles, and format-mode counts.

Per the T6 spec the reader must tolerate absent fields by reporting
`"unknown"` instead of crashing — older daemons or non-T5 branches will
emit entries without the new telemetry fields. Operational safety
matters during the development cycle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brv_bench.adapters.telemetry import TelemetryReader, TelemetrySummary


def _write_log(dir_: Path, name: str, payload: dict) -> Path:
    """Write one JSON log entry under `dir_/<subdir>/<name>.json`."""
    dir_.mkdir(parents=True, exist_ok=True)
    p = dir_ / f"{name}.json"
    p.write_text(json.dumps(payload))
    return p


def _query_entry(
    *,
    tier: int = 3,
    fmt: str = "html",
    input_tokens: int | None = 1000,
    output_tokens: int | None = 100,
    cached_input_tokens: int | None = 0,
    cache_creation_tokens: int | None = None,
    total_ms: float | None = 5000.0,
    search_ms: float | None = 50.0,
    llm_ms: float | None = 4900.0,
    status: str = "completed",
) -> dict:
    """Synthesize a query-log entry matching the real on-disk schema."""
    entry: dict = {
        "id": "qry-1778431925823",
        "query": "test query",
        "tier": tier,
        "format": fmt,
        "status": status,
        "matchedDocs": [],
        "timing": {},
    }
    if total_ms is not None:
        entry["timing"]["totalMs"] = total_ms
    if search_ms is not None:
        entry["timing"]["searchMs"] = search_ms
    if llm_ms is not None:
        entry["timing"]["llmMs"] = llm_ms
    if input_tokens is not None:
        entry["inputTokens"] = input_tokens
    if output_tokens is not None:
        entry["outputTokens"] = output_tokens
    if cached_input_tokens is not None:
        entry["cachedInputTokens"] = cached_input_tokens
    if cache_creation_tokens is not None:
        entry["cacheCreationTokens"] = cache_creation_tokens
    return entry


def _curate_entry(
    *,
    fmt: str = "html",
    input_tokens: int | None = 20000,
    output_tokens: int | None = 800,
    cached_input_tokens: int | None = 0,
    total_ms: float | None = 12000.0,
    llm_ms: float | None = 10000.0,
    status: str = "completed",
) -> dict:
    entry: dict = {
        "id": "cur-1778432414899",
        "format": fmt,
        "status": status,
        "timing": {},
    }
    if total_ms is not None:
        entry["timing"]["totalMs"] = total_ms
    if llm_ms is not None:
        entry["timing"]["llmMs"] = llm_ms
    if input_tokens is not None:
        entry["inputTokens"] = input_tokens
    if output_tokens is not None:
        entry["outputTokens"] = output_tokens
    if cached_input_tokens is not None:
        entry["cachedInputTokens"] = cached_input_tokens
    return entry


class TestEmptyProject:
    def test_returns_empty_summary_when_no_log_dirs_exist(self, tmp_path: Path):
        reader = TelemetryReader(tmp_path)
        summary = reader.read()
        assert summary.query_count == 0
        assert summary.curate_count == 0
        assert summary.tokens.input_tokens == 0
        assert summary.tokens.output_tokens == 0

    def test_empty_log_dirs_present_but_no_entries(self, tmp_path: Path):
        (tmp_path / "query-log").mkdir()
        (tmp_path / "curate-log").mkdir()
        summary = TelemetryReader(tmp_path).read()
        assert summary.query_count == 0
        assert summary.curate_count == 0


class TestQueryAggregation:
    def test_aggregates_token_totals_across_queries(self, tmp_path: Path):
        qdir = tmp_path / "query-log"
        _write_log(qdir, "qry-1", _query_entry(input_tokens=1000, output_tokens=100, cached_input_tokens=50))
        _write_log(qdir, "qry-2", _query_entry(input_tokens=2000, output_tokens=200, cached_input_tokens=150))
        _write_log(qdir, "qry-3", _query_entry(input_tokens=3000, output_tokens=300, cached_input_tokens=0))
        summary = TelemetryReader(tmp_path).read()
        assert summary.query_count == 3
        assert summary.tokens.input_tokens == 6000
        assert summary.tokens.output_tokens == 600
        assert summary.tokens.cached_input_tokens == 200

    def test_aggregates_cache_creation_tokens_when_present(self, tmp_path: Path):
        qdir = tmp_path / "query-log"
        _write_log(qdir, "qry-1", _query_entry(cache_creation_tokens=500))
        _write_log(qdir, "qry-2", _query_entry(cache_creation_tokens=300))
        summary = TelemetryReader(tmp_path).read()
        assert summary.tokens.cache_creation_tokens == 800

    def test_tier_distribution_counts(self, tmp_path: Path):
        qdir = tmp_path / "query-log"
        _write_log(qdir, "qry-1", _query_entry(tier=2))
        _write_log(qdir, "qry-2", _query_entry(tier=2))
        _write_log(qdir, "qry-3", _query_entry(tier=3))
        _write_log(qdir, "qry-4", _query_entry(tier=4))
        summary = TelemetryReader(tmp_path).read()
        assert summary.tier_counts == {2: 2, 3: 1, 4: 1}

    def test_format_counts(self, tmp_path: Path):
        qdir = tmp_path / "query-log"
        _write_log(qdir, "qry-1", _query_entry(fmt="html"))
        _write_log(qdir, "qry-2", _query_entry(fmt="html"))
        _write_log(qdir, "qry-3", _query_entry(fmt="markdown"))
        summary = TelemetryReader(tmp_path).read()
        assert summary.format_counts == {"html": 2, "markdown": 1, "unknown": 0}

    def test_latency_percentiles_for_total_ms(self, tmp_path: Path):
        # 5 entries with deterministic latencies; verify p50/p95/p99 use
        # nearest-rank percentiles (matching numpy's default percentile method).
        qdir = tmp_path / "query-log"
        latencies = [100, 200, 300, 400, 500]  # ms
        for i, ms in enumerate(latencies):
            _write_log(qdir, f"qry-{i}", _query_entry(total_ms=ms))
        summary = TelemetryReader(tmp_path).read()
        # p50 of 5 values via linear interpolation is index 2.0 = 300
        assert summary.latency.total_ms.p50 == 300.0
        # p95 of 5 values: 0.95 * 4 = 3.8 → between 400 and 500 → 480
        assert summary.latency.total_ms.p95 == pytest.approx(480.0)
        # p99 of 5 values: 0.99 * 4 = 3.96 → 496
        assert summary.latency.total_ms.p99 == pytest.approx(496.0)

    def test_latency_p50_p95_p99_for_each_timing_tier(self, tmp_path: Path):
        qdir = tmp_path / "query-log"
        for i, (t, s, l) in enumerate(
            [(100, 10, 80), (200, 20, 160), (300, 30, 250), (400, 40, 340), (500, 50, 420)]
        ):
            _write_log(qdir, f"qry-{i}", _query_entry(total_ms=t, search_ms=s, llm_ms=l))
        summary = TelemetryReader(tmp_path).read()
        assert summary.latency.search_ms.p50 == 30.0
        assert summary.latency.llm_ms.p50 == 250.0


class TestCurateAggregation:
    def test_aggregates_curate_tokens(self, tmp_path: Path):
        cdir = tmp_path / "curate-log"
        _write_log(cdir, "cur-1", _curate_entry(input_tokens=10000, output_tokens=500))
        _write_log(cdir, "cur-2", _curate_entry(input_tokens=20000, output_tokens=600))
        summary = TelemetryReader(tmp_path).read()
        assert summary.curate_count == 2
        assert summary.curate_tokens.input_tokens == 30000
        assert summary.curate_tokens.output_tokens == 1100


class TestAbsentFieldTolerance:
    """T6 spec: 'if older daemon emits QueryLogEntry without new fields,
    bench reports them as "unknown" and proceeds; doesn't crash.'"""

    def test_missing_token_fields_treated_as_zero_not_crash(self, tmp_path: Path):
        qdir = tmp_path / "query-log"
        # Entry with NO token fields at all (legacy daemon shape)
        _write_log(
            qdir,
            "qry-1",
            {"id": "qry-1", "tier": 3, "status": "completed", "timing": {"totalMs": 5000}},
        )
        summary = TelemetryReader(tmp_path).read()
        assert summary.query_count == 1
        assert summary.tokens.input_tokens == 0
        assert summary.tokens.output_tokens == 0
        # An entry with no input_tokens means we observed 0 token-bearing
        # entries — the field is "unknown" for reporting purposes, not 0.
        assert summary.tokens.token_field_coverage == 0  # 0 of 1 entries had tokens

    def test_missing_format_field_counts_as_unknown(self, tmp_path: Path):
        qdir = tmp_path / "query-log"
        _write_log(qdir, "qry-1", {"id": "qry-1", "tier": 2, "status": "completed"})
        summary = TelemetryReader(tmp_path).read()
        assert summary.format_counts["unknown"] == 1
        assert summary.format_counts["html"] == 0
        assert summary.format_counts["markdown"] == 0

    def test_missing_tier_field_counts_as_unknown_tier(self, tmp_path: Path):
        qdir = tmp_path / "query-log"
        _write_log(qdir, "qry-1", {"id": "qry-1", "status": "completed"})
        summary = TelemetryReader(tmp_path).read()
        # Use sentinel "unknown" string key (not 0/None) so downstream
        # reporting can distinguish "missing" from "tier 0".
        assert summary.tier_counts.get("unknown") == 1

    def test_corrupt_json_file_skipped_not_fatal(self, tmp_path: Path):
        qdir = tmp_path / "query-log"
        qdir.mkdir()
        (qdir / "qry-corrupt.json").write_text("{not valid json")
        _write_log(qdir, "qry-good", _query_entry(input_tokens=500, output_tokens=50))
        summary = TelemetryReader(tmp_path).read()
        assert summary.query_count == 1  # only the good entry counted
        assert summary.tokens.input_tokens == 500

    def test_non_dict_root_skipped(self, tmp_path: Path):
        qdir = tmp_path / "query-log"
        qdir.mkdir()
        (qdir / "qry-array.json").write_text('["not", "a", "dict"]')
        _write_log(qdir, "qry-good", _query_entry())
        summary = TelemetryReader(tmp_path).read()
        assert summary.query_count == 1


class TestSinceFilter:
    """For two-config orchestration the bench needs to read only entries
    written during ONE config's run, not entries left over from previous
    runs against the same project. A timestamp filter keyed off the
    file's mtime is the cleanest selector (no clock-coupling with brv)."""

    def test_since_filter_excludes_older_entries(self, tmp_path: Path):
        import time as _time

        qdir = tmp_path / "query-log"
        qdir.mkdir()
        old = _write_log(qdir, "qry-old", _query_entry(input_tokens=100))
        # Backdate the old file to before the cutoff.
        old_mtime = _time.time() - 3600
        os_utime_args = (old_mtime, old_mtime)
        import os
        os.utime(old, os_utime_args)

        cutoff = _time.time() - 60  # 1 min ago
        _write_log(qdir, "qry-new", _query_entry(input_tokens=200))

        summary = TelemetryReader(tmp_path).read(since=cutoff)
        assert summary.query_count == 1
        assert summary.tokens.input_tokens == 200

    def test_no_since_filter_reads_everything(self, tmp_path: Path):
        qdir = tmp_path / "query-log"
        _write_log(qdir, "qry-1", _query_entry(input_tokens=100))
        _write_log(qdir, "qry-2", _query_entry(input_tokens=200))
        summary = TelemetryReader(tmp_path).read()
        assert summary.query_count == 2
        assert summary.tokens.input_tokens == 300
