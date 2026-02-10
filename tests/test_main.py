"""Tests for __main__.py entry point."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from brv_bench.__main__ import load_ground_truth, main, parse_args
from brv_bench.types import GroundTruthDataset

# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_curate_command(self):
        args = parse_args(["curate", "--source", "/some/path"])
        assert args.command == "curate"
        assert args.source == Path("/some/path")

    def test_evaluate_command(self):
        args = parse_args(["evaluate", "--ground-truth", "gt.json"])
        assert args.command == "evaluate"
        assert args.ground_truth == Path("gt.json")
        assert args.limit == 10  # default

    def test_evaluate_custom_limit(self):
        args = parse_args(["evaluate", "--ground-truth", "gt.json", "--limit", "5"])
        assert args.limit == 5

    def test_no_command_fails(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_curate_missing_source_fails(self):
        with pytest.raises(SystemExit):
            parse_args(["curate"])

    def test_evaluate_missing_ground_truth_fails(self):
        with pytest.raises(SystemExit):
            parse_args(["evaluate"])


# ---------------------------------------------------------------------------
# load_ground_truth
# ---------------------------------------------------------------------------


class TestLoadGroundTruth:
    def test_loads_valid_json(self, tmp_path: Path):
        gt_file = tmp_path / "gt.json"
        gt_data = {
            "name": "test-project",
            "entries": [
                {
                    "query": "How does auth work?",
                    "expected_docs": ["auth/oauth.md"],
                    "category": "natural-language",
                },
                {
                    "query": "Database schema",
                    "expected_docs": ["db/schema.md", "db/migrations.md"],
                },
            ],
        }
        gt_file.write_text(json.dumps(gt_data))

        dataset = load_ground_truth(gt_file)

        assert isinstance(dataset, GroundTruthDataset)
        assert dataset.name == "test-project"
        assert len(dataset.entries) == 2
        assert dataset.entries[0].query == "How does auth work?"
        assert dataset.entries[0].expected_docs == ("auth/oauth.md",)
        assert dataset.entries[0].category == "natural-language"
        # default category
        assert dataset.entries[1].category == "unspecified"

    def test_missing_file_exits(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            load_ground_truth(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def test_curate_success(self, tmp_path: Path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "a.md").write_text("content")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"OK", b"")

        with patch(
            "brv_bench.commands.curate.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            import asyncio

            code = asyncio.run(main(["curate", "--source", str(source)]))

        assert code == 0

    def test_curate_with_failures_returns_1(self, tmp_path: Path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "a.md").write_text("content")

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"error")

        with patch(
            "brv_bench.commands.curate.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            import asyncio

            code = asyncio.run(main(["curate", "--source", str(source)]))

        assert code == 1

    def test_evaluate_no_adapter_returns_1(self, tmp_path: Path):
        gt_file = tmp_path / "gt.json"
        gt_data = {
            "name": "test",
            "entries": [
                {"query": "q", "expected_docs": ["a.md"]},
            ],
        }
        gt_file.write_text(json.dumps(gt_data))

        import asyncio

        code = asyncio.run(main(["evaluate", "--ground-truth", str(gt_file)]))

        # Returns 1 because no adapter is configured yet
        assert code == 1
