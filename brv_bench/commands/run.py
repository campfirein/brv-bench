"""Two-config A/B orchestrator (`brv-bench run`).

Runs `curate` + `evaluate` against two pre-prepared byterover-cli
checkouts (one per config), reads per-config telemetry from each
checkout's project data dir, and produces a side-by-side comparison
markdown report ending with `Decision: greenlight | red-light | discussion`.

Per the T6 decision-criteria doc the verdict is gated on three axes:
accuracy, latency, and tokens. Tokens require T5 telemetry on BOTH
configs — see the `Decision Criteria` document in Linear for the exact
thresholds. The reporter's gate logic mirrors that doc.

## Daemon lifecycle

The brv daemon is global per-user (single socket under the platform's
data dir). If a daemon is already running when we invoke Config B's
`bin/run.js`, the CLI will talk to that daemon — which is whichever
version forked it first. To prevent Config B from accidentally
exercising Config A's daemon code path, the orchestrator restarts the
daemon between configs via `<checkout>/bin/run.js restart`. The same
restart is also issued before Config A so any pre-existing daemon from
a prior session is replaced by a fresh fork from Config A's checkout.

## Setup contract (must be done by the caller before `run_two_config_bench`)

Each working dir must be a byterover-cli checkout where:
- `npm install` has completed
- `npm run build` has produced `dist/`
- `bin/run.js` is executable and resolves Node correctly

If `bin/run.js` doesn't exist, the orchestrator raises before spending
any LLM tokens.

The bench does NOT install dependencies or build either checkout — the
team's environment is left untouched; only the subprocess `cwd` and
`brv_bin` are derived from the working-dir path.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brv_bench.adapters.brv_cli import BrvCliAdapter
from brv_bench.adapters.justifier import AnswerJustifier
from brv_bench.adapters.telemetry import TelemetryReader, TelemetrySummary
from brv_bench.brv_io import resolve_project_data_dir
from brv_bench.commands.curate import curate
from brv_bench.commands.evaluate import evaluate
from brv_bench.metrics.base import Metric
from brv_bench.reporting.comparison import ComparisonReport, build_comparison_report
from brv_bench.types import BenchmarkDataset, BenchmarkReport, PromptConfig

logger = logging.getLogger(__name__)


def _resolve_brv_bin(brv_dir: Path) -> str:
    """Resolve the per-checkout brv entrypoint.

    The byterover-cli convention is `<checkout>/bin/run.js` — a
    `#!/usr/bin/env node` script with executable permission. We
    verify both before returning; any deviation surfaces immediately
    instead of mid-curate.
    """
    bin_path = brv_dir / "bin" / "run.js"
    if not bin_path.exists():
        raise FileNotFoundError(
            f"brv entrypoint not found: {bin_path}. Each working dir passed to "
            f"`brv-bench run` must be a byterover-cli checkout with `npm install` "
            f"and `npm run build` completed. The orchestrator does NOT build the "
            f"checkouts itself — that's the caller's responsibility."
        )
    if not bin_path.is_file():
        raise FileNotFoundError(f"brv entrypoint is not a regular file: {bin_path}")
    return str(bin_path)


async def _restart_daemon(brv_bin: str, cwd: Path) -> None:
    """Stop any running brv daemon so the next invocation forks fresh.

    The daemon socket is global per-user (single instance regardless of
    which checkout forked it). Restarting before each config ensures the
    daemon code path matches the config's checkout, not whichever brv
    binary happened to start the daemon first.

    `brv restart` is the canonical CLI for this — it sends a graceful
    shutdown and waits for the daemon to exit. Non-zero exit codes are
    logged but not fatal: a missing-daemon failure is the desired state.
    """
    proc = await asyncio.create_subprocess_exec(
        brv_bin,
        "restart",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.debug(
            "brv restart returned %d (likely no daemon running): %s",
            proc.returncode, (stderr or stdout).decode(errors="replace").strip(),
        )


@dataclass(frozen=True)
class RunConfig:
    """Input to `run_two_config_bench`.

    `brv_a_dir` / `brv_b_dir` are pre-prepared byterover-cli checkouts —
    the orchestrator spawns brv via `<dir>/bin/run.js` per config so two
    branches can run side-by-side without `npm link` conflicts. Each
    checkout must have `npm install && npm run build` already done.

    `output_path` is the markdown report destination.

    `judge` and `justifier` instances, when set, get reused across both
    configs (the LLM-side judging is config-independent so we pay the
    setup cost once).
    """

    dataset: BenchmarkDataset
    prompt_config: PromptConfig
    brv_a_dir: Path
    brv_b_dir: Path
    output_path: Path
    limit: int = 10
    metrics: tuple[Metric, ...] = field(default_factory=tuple)
    justifier: AnswerJustifier | None = None
    label_a: str = "Config A (main + T5 backport)"
    label_b: str = "Config B (HTML)"


@dataclass(frozen=True)
class RunResult:
    """What `run_two_config_bench` returns to the CLI."""

    config_a_report: BenchmarkReport
    config_b_report: BenchmarkReport
    telemetry_a: TelemetrySummary
    telemetry_b: TelemetrySummary
    comparison: ComparisonReport


async def run_two_config_bench(config: RunConfig) -> RunResult:
    """Run the full A/B pipeline and write the comparison report to disk.

    Per-config sequence:
      1. Capture start_time (used as telemetry since-filter cutoff)
      2. Build BrvCliAdapter pinned to the config's working dir + bin
      3. Run curate phase against dataset.corpus
      4. Run evaluate phase against dataset.entries
      5. Resolve project data dir from working dir
      6. Read telemetry filtered by start_time

    Then: assemble ComparisonReport, write its markdown to output_path,
    return RunResult.

    Args:
        config: The orchestrator inputs.

    Returns:
        RunResult containing both per-config BenchmarkReports, both
        TelemetrySummaries, and the assembled ComparisonReport.
    """
    config_a_report, telemetry_a = await _run_one_config(
        label=config.label_a,
        dataset=config.dataset,
        prompt_config=config.prompt_config,
        brv_dir=config.brv_a_dir,
        limit=config.limit,
        metrics=list(config.metrics),
        justifier=config.justifier,
    )

    config_b_report, telemetry_b = await _run_one_config(
        label=config.label_b,
        dataset=config.dataset,
        prompt_config=config.prompt_config,
        brv_dir=config.brv_b_dir,
        limit=config.limit,
        metrics=list(config.metrics),
        justifier=config.justifier,
    )

    comparison = build_comparison_report(
        report_a=config_a_report,
        report_b=config_b_report,
        telemetry_a=telemetry_a,
        telemetry_b=telemetry_b,
        label_a=config.label_a,
        label_b=config.label_b,
    )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(comparison.markdown)
    logger.info("wrote comparison report to %s (decision: %s)",
                config.output_path, comparison.decision)

    return RunResult(
        config_a_report=config_a_report,
        config_b_report=config_b_report,
        telemetry_a=telemetry_a,
        telemetry_b=telemetry_b,
        comparison=comparison,
    )


# ---------------------------------------------------------------------------
# Per-config plumbing
# ---------------------------------------------------------------------------


async def _run_one_config(
    *,
    label: str,
    dataset: BenchmarkDataset,
    prompt_config: PromptConfig,
    brv_dir: Path,
    limit: int,
    metrics: list[Metric],
    justifier: AnswerJustifier | None,
) -> tuple[BenchmarkReport, TelemetrySummary]:
    """Run one config end-to-end: curate, evaluate, read telemetry.

    The `since` cutoff for telemetry reading is captured BEFORE the
    curate phase so prior runs against the same project data dir don't
    leak into this config's totals.
    """
    logger.info("[%s] starting run against %s", label, brv_dir)

    # Each checkout has its own bin/run.js; invoking it directly avoids
    # `npm link` conflicts when two brv versions need to run side-by-side
    # on the same machine. The resolver validates existence before any
    # LLM call so misconfiguration surfaces immediately.
    brv_bin = _resolve_brv_bin(brv_dir)

    # Daemon hygiene: kill any lingering daemon before this config starts so
    # the brv invocations below fork a fresh daemon from THIS checkout's
    # dist/, not whichever version got there first. The same restart fires
    # again at the END of this config so the next config starts equally clean.
    await _restart_daemon(brv_bin, brv_dir)

    start_time = time.time()

    adapter = BrvCliAdapter(
        prompt_config=prompt_config,
        justifier=justifier,
        working_dir=brv_dir,
        brv_bin=brv_bin,
    )

    # 1. Curate the corpus. brv_bin + cwd pin this config's curates to its
    #    own checkout's bin/run.js + .brv/ project — otherwise both configs
    #    would invoke the globally-linked `brv` from PATH and write to the
    #    same data dir, collapsing the two-config setup.
    summary = await curate(dataset.corpus, prompt_config, brv_bin=brv_bin, cwd=brv_dir)
    logger.info(
        "[%s] curate done: %d/%d succeeded", label, summary.succeeded, summary.total,
    )

    # 2. Evaluate the entries.
    report = await evaluate(
        adapter=adapter,
        dataset=dataset,
        metrics=metrics,
        limit=limit,
        output_path=None,  # we don't write per-config JSON — only the comparison
    )
    logger.info("[%s] evaluate done: %d queries", label, report.query_count)

    # 3. Read telemetry from the per-config data dir, filtered by start_time.
    data_dir = resolve_project_data_dir(brv_dir)
    telemetry = TelemetryReader(data_dir).read(since=start_time)
    logger.info(
        "[%s] telemetry: %d queries, %d curates, token coverage %d/%d",
        label, telemetry.query_count, telemetry.curate_count,
        telemetry.tokens.token_field_coverage, telemetry.query_count,
    )

    # 4. Stop the daemon so the next config starts from a clean slate.
    await _restart_daemon(brv_bin, brv_dir)

    return report, telemetry
