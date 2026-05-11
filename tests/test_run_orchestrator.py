"""Unit tests for the `brv-bench run` orchestrator (T6 two-config A/B).

The orchestrator wires together: per-config adapters (each pinned to its own
brv checkout), the curate + evaluate phases (reusing existing commands),
the telemetry reader (per-config, since-filtered), and the comparison
reporter. Failure here is the most likely place for a real bench run to
fall over, so coverage focuses on contract + ordering rather than the
already-covered building blocks.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from brv_bench.adapters.telemetry import (
    LatencyBreakdown,
    LatencyTier,
    TelemetrySummary,
    TokenTotals,
)
from brv_bench.commands.run import RunConfig, run_two_config_bench
from brv_bench.reporting.comparison import (
    DECISION_DISCUSSION,
    DECISION_GREENLIGHT,
)
from brv_bench.types import (
    BenchmarkDataset,
    BenchmarkReport,
    CorpusDocument,
    GroundTruthEntry,
    MetricResult,
    PromptConfig,
)


PROMPT_CONFIG = PromptConfig(
    curate_template="curate: {doc_id} {source}\n{content}",
    query_template="{question}",
    judge_template="judge:{question}|{generated_answer}|{gold_answer}",
    justifier_template="justify:{question}|{context}",
)


def _dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        name="locomo",
        corpus=(CorpusDocument(doc_id="session_1", content="dialog 1", source="conv_26"),),
        entries=(
            GroundTruthEntry(
                query="Q?",
                expected_doc_ids=("session_1",),
                category="temporal",
                expected_answer="A",
            ),
        ),
    )


def _telemetry(input_tokens: int = 1000, p99_total: float = 1000.0) -> TelemetrySummary:
    return TelemetrySummary(
        query_count=1,
        curate_count=1,
        tier_counts={3: 1},
        format_counts={"html": 1, "markdown": 0, "unknown": 0},
        tokens=TokenTotals(
            input_tokens=input_tokens, output_tokens=100, cached_input_tokens=0,
            cache_creation_tokens=0, token_field_coverage=1,
        ),
        curate_tokens=TokenTotals(
            input_tokens=10_000, output_tokens=500, cached_input_tokens=0,
            cache_creation_tokens=0, token_field_coverage=1,
        ),
        latency=LatencyBreakdown(
            total_ms=LatencyTier(count=1, p50=p99_total, p95=p99_total, p99=p99_total, mean=p99_total),
            search_ms=LatencyTier(count=1, p50=50, p95=50, p99=50, mean=50),
            llm_ms=LatencyTier(count=1, p50=950, p95=950, p99=950, mean=950),
        ),
    )


def _benchmark_report(name: str, recall: float) -> BenchmarkReport:
    return BenchmarkReport(
        name="locomo",
        memory_system=name,
        context_tree_docs=1,
        query_count=1,
        duration_ms=10_000,
        metrics=(MetricResult(name="recall@5", label="Recall@5", value=recall, unit="ratio"),),
        category_breakdown=(),
    )


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


def _make_brv_checkout(parent: Path, name: str) -> Path:
    """Create a minimal stub byterover-cli checkout: just bin/run.js so
    the orchestrator's pre-flight validation passes. The orchestrator's
    _restart_daemon is patched in tests so the script content is unused."""
    checkout = parent / name
    bin_dir = checkout / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    run_js = bin_dir / "run.js"
    run_js.write_text("#!/usr/bin/env node\n")
    run_js.chmod(0o755)
    return checkout


def _patch_run_dependencies(report_a: BenchmarkReport, report_b: BenchmarkReport,
                            telemetry_a: TelemetrySummary, telemetry_b: TelemetrySummary):
    """Patch the orchestrator's collaborators: curate, evaluate, telemetry reader."""
    curate_summary = MagicMock(total=1, succeeded=1, failed=0, results=())
    curate = AsyncMock(return_value=curate_summary)
    # Two evaluate calls in sequence — return A then B
    evaluate = AsyncMock(side_effect=[report_a, report_b])

    telemetry_reader = MagicMock()
    telemetry_reader.read = MagicMock(side_effect=[telemetry_a, telemetry_b])

    return {
        "curate": curate,
        "evaluate": evaluate,
        "telemetry_reader_class": MagicMock(return_value=telemetry_reader),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunOrchestrator:
    def test_runs_curate_then_evaluate_per_config(self, tmp_path: Path):
        report_a = _benchmark_report("brv-cli-A", recall=0.40)
        report_b = _benchmark_report("brv-cli-B", recall=0.90)
        mocks = _patch_run_dependencies(report_a, report_b, _telemetry(), _telemetry())

        config = RunConfig(
            dataset=_dataset(),
            prompt_config=PROMPT_CONFIG,
            brv_a_dir=_make_brv_checkout(tmp_path, "brv_a"),
            brv_b_dir=_make_brv_checkout(tmp_path, "brv_b"),
            output_path=tmp_path / "report.md",
            limit=5,
        )

        with patch("brv_bench.commands.run.curate", mocks["curate"]), \
             patch("brv_bench.commands.run.evaluate", mocks["evaluate"]), \
             patch("brv_bench.commands.run.TelemetryReader", mocks["telemetry_reader_class"]):
            result = asyncio.run(run_two_config_bench(config))

        # Both configs ran (one curate per config, one evaluate per config)
        assert mocks["curate"].await_count == 2
        assert mocks["evaluate"].await_count == 2
        # Both telemetries read
        assert mocks["telemetry_reader_class"].call_count == 2
        # Result has both reports + a final comparison
        assert result.config_a_report.memory_system == "brv-cli-A"
        assert result.config_b_report.memory_system == "brv-cli-B"
        assert result.comparison.markdown.startswith("# Side-by-side")
        # Decision computed
        assert result.comparison.decision in {DECISION_GREENLIGHT, DECISION_DISCUSSION, "red-light"}

    def test_writes_output_markdown_to_disk(self, tmp_path: Path):
        report_a = _benchmark_report("A", recall=0.40)
        report_b = _benchmark_report("B", recall=0.90)
        mocks = _patch_run_dependencies(report_a, report_b, _telemetry(), _telemetry())

        output = tmp_path / "report.md"
        config = RunConfig(
            dataset=_dataset(),
            prompt_config=PROMPT_CONFIG,
            brv_a_dir=_make_brv_checkout(tmp_path, "brv_a"),
            brv_b_dir=_make_brv_checkout(tmp_path, "brv_b"),
            output_path=output,
            limit=5,
        )

        with patch("brv_bench.commands.run.curate", mocks["curate"]), \
             patch("brv_bench.commands.run.evaluate", mocks["evaluate"]), \
             patch("brv_bench.commands.run.TelemetryReader", mocks["telemetry_reader_class"]):
            asyncio.run(run_two_config_bench(config))

        assert output.exists()
        content = output.read_text()
        assert "Side-by-side" in content
        assert content.rstrip().splitlines()[-1].startswith("Decision:")

    def test_passes_working_dir_to_adapters(self, tmp_path: Path):
        # The orchestrator must build adapters pinned to each working dir so
        # subprocess calls land in the right brv project. Without this, both
        # configs would run against the same `.brv/` project — defeating the
        # entire two-config setup.
        report_a = _benchmark_report("A", recall=0.40)
        report_b = _benchmark_report("B", recall=0.90)
        mocks = _patch_run_dependencies(report_a, report_b, _telemetry(), _telemetry())

        adapter_class = MagicMock()
        # The evaluate mock receives the adapter as first positional argument.
        # We assert the adapter was constructed with the right working_dir.

        brv_a = _make_brv_checkout(tmp_path, "brv_a")
        brv_b = _make_brv_checkout(tmp_path, "brv_b")
        config = RunConfig(
            dataset=_dataset(),
            prompt_config=PROMPT_CONFIG,
            brv_a_dir=brv_a,
            brv_b_dir=brv_b,
            output_path=tmp_path / "out.md",
            limit=5,
        )

        with patch("brv_bench.commands.run.curate", mocks["curate"]), \
             patch("brv_bench.commands.run.evaluate", mocks["evaluate"]), \
             patch("brv_bench.commands.run.TelemetryReader", mocks["telemetry_reader_class"]), \
             patch("brv_bench.commands.run.BrvCliAdapter", adapter_class):
            asyncio.run(run_two_config_bench(config))

        # Adapter built twice, once per config, with matching working_dirs.
        assert adapter_class.call_count == 2
        ctor_kwargs = [call.kwargs for call in adapter_class.call_args_list]
        working_dirs = [kw.get("working_dir") for kw in ctor_kwargs]
        assert set(working_dirs) == {brv_a, brv_b}

    def test_telemetry_reader_constructed_with_per_config_data_dir(self, tmp_path: Path):
        # Each config's telemetry must be read from its own project data dir
        # (resolved from its working dir). Otherwise both configs would read
        # the same telemetry and the comparison would be meaningless.
        report_a = _benchmark_report("A", recall=0.40)
        report_b = _benchmark_report("B", recall=0.90)
        mocks = _patch_run_dependencies(report_a, report_b, _telemetry(), _telemetry())

        brv_a = _make_brv_checkout(tmp_path, "brv_a")
        brv_b = _make_brv_checkout(tmp_path, "brv_b")
        config = RunConfig(
            dataset=_dataset(),
            prompt_config=PROMPT_CONFIG,
            brv_a_dir=brv_a,
            brv_b_dir=brv_b,
            output_path=tmp_path / "out.md",
            limit=5,
        )

        with patch("brv_bench.commands.run.curate", mocks["curate"]), \
             patch("brv_bench.commands.run.evaluate", mocks["evaluate"]), \
             patch("brv_bench.commands.run.TelemetryReader", mocks["telemetry_reader_class"]), \
             patch("brv_bench.commands.run.resolve_project_data_dir") as resolve:
            resolve.side_effect = lambda d: f"/fake/data/{Path(d).name}"
            asyncio.run(run_two_config_bench(config))

        # TelemetryReader called once per config with the resolved data dir
        ctor_args = [call.args[0] for call in mocks["telemetry_reader_class"].call_args_list]
        assert ctor_args == ["/fake/data/brv_a", "/fake/data/brv_b"]

    def test_telemetry_since_filter_uses_per_config_start_time(self, tmp_path: Path):
        # The since-filter is what isolates one config's logs from prior runs
        # or from the other config's logs in the same data-dir. Each
        # telemetry read must use the start_time captured BEFORE that
        # config's curate began.
        report_a = _benchmark_report("A", recall=0.40)
        report_b = _benchmark_report("B", recall=0.90)
        telemetry_reader_a = MagicMock()
        telemetry_reader_a.read = MagicMock(return_value=_telemetry())
        telemetry_reader_b = MagicMock()
        telemetry_reader_b.read = MagicMock(return_value=_telemetry())
        telemetry_class = MagicMock(side_effect=[telemetry_reader_a, telemetry_reader_b])

        config = RunConfig(
            dataset=_dataset(),
            prompt_config=PROMPT_CONFIG,
            brv_a_dir=_make_brv_checkout(tmp_path, "brv_a"),
            brv_b_dir=_make_brv_checkout(tmp_path, "brv_b"),
            output_path=tmp_path / "out.md",
            limit=5,
        )

        curate = AsyncMock(return_value=MagicMock(total=1, succeeded=1, failed=0, results=()))
        evaluate = AsyncMock(side_effect=[report_a, report_b])

        with patch("brv_bench.commands.run.curate", curate), \
             patch("brv_bench.commands.run.evaluate", evaluate), \
             patch("brv_bench.commands.run.TelemetryReader", telemetry_class):
            asyncio.run(run_two_config_bench(config))

        # Both readers were called with a `since=` kwarg set to a positive
        # float (the timestamp captured before each curate). The exact value
        # is not pinned — we just verify the contract.
        for reader in (telemetry_reader_a, telemetry_reader_b):
            call = reader.read.call_args
            since = call.kwargs.get("since")
            assert isinstance(since, float)
            assert since > 0

    def test_missing_bin_run_js_raises_before_any_curate(self, tmp_path: Path):
        """If a working dir doesn't have bin/run.js the orchestrator must
        fail BEFORE any subprocess fires — otherwise we burn LLM tokens
        on a misconfigured setup."""
        report_a = _benchmark_report("A", recall=0.40)
        report_b = _benchmark_report("B", recall=0.90)
        mocks = _patch_run_dependencies(report_a, report_b, _telemetry(), _telemetry())

        # brv_a is deliberately empty — no bin/run.js so pre-flight fails
        brv_a = tmp_path / "brv_a"
        brv_a.mkdir(exist_ok=True)
        # brv_b has the right shape (orchestrator should never get this far)
        brv_b = _make_brv_checkout(tmp_path, "brv_b")

        config = RunConfig(
            dataset=_dataset(),
            prompt_config=PROMPT_CONFIG,
            brv_a_dir=brv_a,
            brv_b_dir=brv_b,
            output_path=tmp_path / "out.md",
            limit=5,
        )

        with patch("brv_bench.commands.run.curate", mocks["curate"]), \
             patch("brv_bench.commands.run.evaluate", mocks["evaluate"]), \
             patch("brv_bench.commands.run.TelemetryReader", mocks["telemetry_reader_class"]):
            with pytest.raises(FileNotFoundError, match="brv entrypoint not found"):
                asyncio.run(run_two_config_bench(config))

        # Crucially: zero curates fired before the failure
        assert mocks["curate"].await_count == 0
        assert mocks["evaluate"].await_count == 0

    def test_daemon_restart_called_around_each_config(self, tmp_path: Path):
        """The daemon must be restarted before AND after each config so
        the two configs don't accidentally exercise each other's daemon
        code path. Single global daemon socket = one version at a time."""
        report_a = _benchmark_report("A", recall=0.40)
        report_b = _benchmark_report("B", recall=0.90)
        mocks = _patch_run_dependencies(report_a, report_b, _telemetry(), _telemetry())

        # Both checkouts have bin/run.js
        for name in ("brv_a", "brv_b"):
            d = tmp_path / name / "bin"
            d.mkdir(parents=True)
            (d / "run.js").write_text("#!/usr/bin/env node\n")

        config = RunConfig(
            dataset=_dataset(),
            prompt_config=PROMPT_CONFIG,
            brv_a_dir=_make_brv_checkout(tmp_path, "brv_a"),
            brv_b_dir=_make_brv_checkout(tmp_path, "brv_b"),
            output_path=tmp_path / "out.md",
            limit=5,
        )

        restart = AsyncMock()
        with patch("brv_bench.commands.run.curate", mocks["curate"]), \
             patch("brv_bench.commands.run.evaluate", mocks["evaluate"]), \
             patch("brv_bench.commands.run.TelemetryReader", mocks["telemetry_reader_class"]), \
             patch("brv_bench.commands.run._restart_daemon", restart):
            asyncio.run(run_two_config_bench(config))

        # Restart fires: before A, after A, before B, after B = 4 times
        assert restart.await_count == 4
        # First two restarts use A's bin path
        assert "brv_a" in restart.await_args_list[0].args[0]
        assert "brv_a" in restart.await_args_list[1].args[0]
        # Last two restarts use B's bin path
        assert "brv_b" in restart.await_args_list[2].args[0]
        assert "brv_b" in restart.await_args_list[3].args[0]

    def test_config_b_runs_even_if_config_a_evaluate_returns_empty(self, tmp_path: Path):
        # Soft-fail isolation — a degenerate Config A result (e.g. zero
        # retrieved doc_ids) should not prevent Config B from running.
        # The comparison still produces a report; it just shows A's metrics
        # as zero / discussion verdict.
        empty_report_a = BenchmarkReport(
            name="locomo", memory_system="A", context_tree_docs=1, query_count=1,
            duration_ms=1.0, metrics=(), category_breakdown=(),
        )
        report_b = _benchmark_report("B", recall=0.90)
        mocks = _patch_run_dependencies(empty_report_a, report_b, _telemetry(), _telemetry())

        config = RunConfig(
            dataset=_dataset(),
            prompt_config=PROMPT_CONFIG,
            brv_a_dir=_make_brv_checkout(tmp_path, "brv_a"),
            brv_b_dir=_make_brv_checkout(tmp_path, "brv_b"),
            output_path=tmp_path / "out.md",
            limit=5,
        )

        with patch("brv_bench.commands.run.curate", mocks["curate"]), \
             patch("brv_bench.commands.run.evaluate", mocks["evaluate"]), \
             patch("brv_bench.commands.run.TelemetryReader", mocks["telemetry_reader_class"]):
            result = asyncio.run(run_two_config_bench(config))

        # B's evaluate ran
        assert mocks["evaluate"].await_count == 2
        assert result.config_b_report.metrics  # non-empty
