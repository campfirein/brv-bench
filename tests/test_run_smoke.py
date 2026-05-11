"""End-to-end smoke test for `brv-bench run`.

Wires up the real orchestrator + adapter + telemetry reader + comparison
reporter, but replaces the brv binary with a tiny stub script that:
  - On `curate`: writes a synthetic `cur-<ts>.json` to the project data dir
  - On `query`: emits a synthetic `completed` event JSON line on stdout AND
    writes a synthetic `qry-<ts>.json` to the project data dir
  - On `restart`: no-op

Total cost: zero LLM tokens. Catches subprocess wiring bugs (cwd, brv_bin,
project-data-dir resolution, NDJSON parsing, telemetry since-filter,
report assembly + write) the pure-mock unit tests don't exercise.

Run before any real bench invocation — failures here mean the bench
would burn LLM tokens before discovering the bug.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from brv_bench.brv_io import resolve_project_data_dir
from brv_bench.commands.run import RunConfig, run_two_config_bench
from brv_bench.types import (
    BenchmarkDataset,
    CorpusDocument,
    GroundTruthEntry,
    PromptConfig,
)


_PROMPT = PromptConfig(
    curate_template="curate: {doc_id} {source}\n{content}",
    query_template="{question}",
)


def _write_stub_brv(checkout_dir: Path, tag: str) -> Path:
    """Write a `<checkout>/bin/run.js` shell-style stub that fakes a brv
    binary.

    The stub:
      - On `curate`: writes a cur-<ts>.json telemetry entry to the project
        data dir resolved from cwd
      - On `query`: prints a `completed` JSON event to stdout AND writes
        a qry-<ts>.json telemetry entry
      - On `restart`: exits 0 silently

    `tag` lets each config's stub produce distinguishable telemetry
    (different token counts) so the test can verify per-config isolation.
    """
    bin_dir = checkout_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    run_js = bin_dir / "run.js"

    # We're not actually running node; the file just needs to be executable
    # and have a working shebang. A Python shebang works since macOS / Linux
    # both have python3 on PATH for the test environment.
    script = f"""#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

# Mirror brv's project-data-dir encoding for this stub.
import platform

def _resolve_data_dir(cwd):
    cwd = str(Path(cwd).resolve())
    cwd = cwd.lstrip('/')
    parts = [p for p in cwd.replace('\\\\', '/').split('/') if p]
    sanitized = '--'.join(p.replace('%', '%25').replace('--', '%2D%2D') for p in parts)
    override = os.environ.get('BRV_DATA_DIR')
    if override:
        return Path(override) / 'projects' / sanitized
    system = platform.system()
    if system == 'Darwin':
        return Path.home() / 'Library' / 'Application Support' / 'brv' / 'projects' / sanitized
    if system == 'Linux':
        xdg = os.environ.get('XDG_DATA_HOME')
        base = Path(xdg) if xdg else Path.home() / '.local' / 'share'
        return base / 'brv' / 'projects' / sanitized
    return Path.home() / '.brv' / 'projects' / sanitized

cmd = sys.argv[1] if len(sys.argv) > 1 else ''
cwd = os.getcwd()
data_dir = _resolve_data_dir(cwd)
ts = int(time.time() * 1000)
tag = {tag!r}

if cmd == 'curate':
    log_dir = data_dir / 'curate-log'
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {{
        'id': f'cur-{{ts}}',
        'status': 'completed',
        'format': 'html',
        'inputTokens': 1000 if tag == 'A' else 1500,
        'outputTokens': 100 if tag == 'A' else 150,
        'cachedInputTokens': 0,
        'timing': {{'totalMs': 5000 if tag == 'A' else 6000, 'llmMs': 4900 if tag == 'A' else 5900}},
        'tag': tag,
    }}
    (log_dir / f'cur-{{ts}}.json').write_text(json.dumps(entry))
    sys.exit(0)

if cmd == 'query':
    # Emit a completed event so the bench's NDJSON parser sees structured
    # matchedDocs + result text.
    completed = {{
        'command': 'query',
        'data': {{
            'event': 'completed',
            'durationMs': 100,
            'tier': 2,
            'topScore': 0.95,
            'matchedDocs': [
                {{'path': 'conv_26/session_1.html', 'score': 0.95, 'title': 'Session 1'}}
            ],
            'result': '**Details**: stub answer for ' + tag + '\\n**Sources**:\\n- .brv/context-tree/conv_26/session_1.html',
            'status': 'completed',
            'taskId': f't-{{ts}}',
        }},
        'success': True,
    }}
    print(json.dumps(completed))
    log_dir = data_dir / 'query-log'
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {{
        'id': f'qry-{{ts}}',
        'status': 'completed',
        'tier': 3 if tag == 'A' else 2,
        'format': 'html',
        'inputTokens': 5000 if tag == 'A' else 0,
        'outputTokens': 500 if tag == 'A' else 0,
        'cachedInputTokens': 0,
        'timing': {{'totalMs': 200 if tag == 'B' else 8000, 'llmMs': 0 if tag == 'B' else 7800, 'searchMs': 20}},
        'tag': tag,
        'matchedDocs': [{{'path': 'conv_26/session_1.html', 'score': 0.95, 'title': 'S1'}}],
    }}
    (log_dir / f'qry-{{ts}}.json').write_text(json.dumps(entry))
    sys.exit(0)

if cmd == 'restart':
    sys.exit(0)

# Unknown command: succeed quietly (don't break daemon-shaped operations)
sys.exit(0)
"""
    run_js.write_text(script)
    run_js.chmod(run_js.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return run_js


def _tiny_dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        name="locomo",
        corpus=(
            CorpusDocument(doc_id="session_1", content="dialog 1", source="conv_26"),
        ),
        entries=(
            GroundTruthEntry(
                query="Test query?",
                expected_doc_ids=("session_1",),
                category="single-hop",
                expected_answer="Test answer",
            ),
        ),
    )


class TestRunSmoke:
    """End-to-end with stub brv binaries — no LLM tokens consumed."""

    def test_full_pipeline_produces_comparison_report(self, tmp_path: Path):
        """Wire the real orchestrator end-to-end against stubbed brv
        binaries. Expect: comparison markdown written, decision line
        present, telemetry from BOTH stubs visible (token counts
        differ A vs B so per-config isolation is provable)."""
        brv_a = tmp_path / "brv_a"
        brv_b = tmp_path / "brv_b"
        _write_stub_brv(brv_a, tag="A")
        _write_stub_brv(brv_b, tag="B")

        # Override the brv data dir so the stubs write to a tmp location
        # we control + can clean up.
        data_root = tmp_path / "brv_data"
        data_root.mkdir()

        config = RunConfig(
            dataset=_tiny_dataset(),
            prompt_config=_PROMPT,
            brv_a_dir=brv_a,
            brv_b_dir=brv_b,
            output_path=tmp_path / "out.md",
            limit=5,
        )

        with patch.dict(os.environ, {"BRV_DATA_DIR": str(data_root)}, clear=False):
            result = asyncio.run(run_two_config_bench(config))

        # 1. Output markdown exists and has the expected shape
        assert config.output_path.exists()
        content = config.output_path.read_text()
        assert content.startswith("# Side-by-side")
        assert "| Metric |" in content
        assert content.rstrip().splitlines()[-1].startswith("Decision:")

        # 2. Both configs produced telemetry visible in the result
        assert result.telemetry_a.curate_count == 1
        assert result.telemetry_a.query_count == 1
        assert result.telemetry_b.curate_count == 1
        assert result.telemetry_b.query_count == 1

        # 3. Per-config isolation: A stubs report 1000 input tokens on curate,
        #    B stubs report 1500. If the bench accidentally read the wrong
        #    project-data-dir, the values would collide.
        assert result.telemetry_a.curate_tokens.input_tokens == 1000
        assert result.telemetry_b.curate_tokens.input_tokens == 1500

        # 4. Format counts (T5 telemetry produces 'html' on both stubs)
        assert result.telemetry_a.format_counts["html"] >= 1
        assert result.telemetry_b.format_counts["html"] >= 1

        # 5. Decision is computable (greenlight / red-light / discussion)
        assert result.comparison.decision in ("greenlight", "red-light", "discussion")

    def test_telemetry_since_filter_excludes_pre_run_entries(self, tmp_path: Path):
        """A stale entry in the project data dir (older than the run's
        start_time) must NOT contaminate the per-config telemetry totals.
        This is the load-bearing isolation the since-filter provides."""
        import time

        brv_a = tmp_path / "brv_a"
        brv_b = tmp_path / "brv_b"
        _write_stub_brv(brv_a, tag="A")
        _write_stub_brv(brv_b, tag="B")

        data_root = tmp_path / "brv_data"
        data_root.mkdir()

        # Pre-seed a stale curate entry under brv_a's resolved data dir.
        with patch.dict(os.environ, {"BRV_DATA_DIR": str(data_root)}, clear=False):
            data_a = Path(resolve_project_data_dir(brv_a))
        stale_curate_dir = data_a / "curate-log"
        stale_curate_dir.mkdir(parents=True, exist_ok=True)
        stale_file = stale_curate_dir / "cur-stale.json"
        stale_file.write_text(
            json.dumps({
                "id": "cur-stale",
                "inputTokens": 999_999,  # absurd marker
                "outputTokens": 999_999,
                "timing": {"totalMs": 5000},
                "format": "html",
                "status": "completed",
            })
        )
        # Backdate the stale file to before the run.
        old_mtime = time.time() - 3600
        os.utime(stale_file, (old_mtime, old_mtime))

        config = RunConfig(
            dataset=_tiny_dataset(),
            prompt_config=_PROMPT,
            brv_a_dir=brv_a,
            brv_b_dir=brv_b,
            output_path=tmp_path / "out.md",
            limit=5,
        )

        with patch.dict(os.environ, {"BRV_DATA_DIR": str(data_root)}, clear=False):
            result = asyncio.run(run_two_config_bench(config))

        # The 999_999 stale entry must NOT contaminate A's totals — only
        # the 1000-token entry produced by the stub during this run counts.
        assert result.telemetry_a.curate_tokens.input_tokens == 1000
        assert result.telemetry_a.curate_count == 1
