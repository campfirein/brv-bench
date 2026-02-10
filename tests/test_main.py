"""Tests for __main__.py entry point."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from brv_bench.__main__ import load_dataset, main, parse_args
from brv_bench.types import BenchmarkDataset


# ----------------------------------------------------------------
# parse_args
# ----------------------------------------------------------------


class TestParseArgs:
    def test_curate_command(self):
        args = parse_args(["curate", "--source", "/some/path"])
        assert args.command == "curate"
        assert args.source == Path("/some/path")

    def test_evaluate_command(self):
        args = parse_args(
            ["evaluate", "--ground-truth", "gt.json"]
        )
        assert args.command == "evaluate"
        assert args.ground_truth == Path("gt.json")
        assert args.limit == 10

    def test_evaluate_custom_limit(self):
        args = parse_args(
            [
                "evaluate",
                "--ground-truth",
                "gt.json",
                "--limit",
                "5",
            ]
        )
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


# ----------------------------------------------------------------
# load_dataset
# ----------------------------------------------------------------


class TestLoadDataset:
    def test_loads_valid_json(self, tmp_path: Path):
        gt_file = tmp_path / "dataset.json"
        gt_data = {
            "name": "test-project",
            "corpus": [
                {
                    "doc_id": "auth/oauth.md",
                    "content": "OAuth details",
                    "source": "session_1",
                },
                {
                    "doc_id": "db/schema.md",
                    "content": "DB schema",
                },
            ],
            "entries": [
                {
                    "query": "How does auth work?",
                    "expected_doc_ids": ["auth/oauth.md"],
                    "category": "single-hop",
                    "expected_answer": "Uses OAuth",
                },
                {
                    "query": "Database schema",
                    "expected_doc_ids": [
                        "db/schema.md",
                        "db/migrations.md",
                    ],
                },
            ],
        }
        gt_file.write_text(json.dumps(gt_data))

        dataset = load_dataset(gt_file)

        assert isinstance(dataset, BenchmarkDataset)
        assert dataset.name == "test-project"
        assert len(dataset.corpus) == 2
        assert dataset.corpus[0].doc_id == "auth/oauth.md"
        assert dataset.corpus[0].source == "session_1"
        assert dataset.corpus[1].source == ""
        assert len(dataset.entries) == 2
        assert dataset.entries[0].expected_doc_ids == (
            "auth/oauth.md",
        )
        assert dataset.entries[0].category == "single-hop"
        assert dataset.entries[0].expected_answer == "Uses OAuth"
        assert dataset.entries[1].category == "unspecified"
        assert dataset.entries[1].expected_answer is None

    def test_loads_without_corpus(self, tmp_path: Path):
        gt_file = tmp_path / "dataset.json"
        gt_data = {
            "name": "legacy",
            "entries": [
                {
                    "query": "q",
                    "expected_doc_ids": ["a.md"],
                },
            ],
        }
        gt_file.write_text(json.dumps(gt_data))

        dataset = load_dataset(gt_file)
        assert len(dataset.corpus) == 0
        assert len(dataset.entries) == 1

    def test_missing_file_exits(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            load_dataset(tmp_path / "missing.json")


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------


class TestMain:
    def test_curate_success(self, tmp_path: Path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "a.md").write_text("content")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"OK", b"")

        with patch(
            "brv_bench.commands.curate"
            ".asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            code = asyncio.run(
                main(["curate", "--source", str(source)])
            )

        assert code == 0

    def test_curate_with_failures_returns_1(
        self, tmp_path: Path
    ):
        source = tmp_path / "source"
        source.mkdir()
        (source / "a.md").write_text("content")

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"error")

        with patch(
            "brv_bench.commands.curate"
            ".asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            code = asyncio.run(
                main(["curate", "--source", str(source)])
            )

        assert code == 1

    def test_evaluate_no_adapter_returns_1(
        self, tmp_path: Path
    ):
        gt_file = tmp_path / "dataset.json"
        gt_data = {
            "name": "test",
            "corpus": [],
            "entries": [
                {
                    "query": "q",
                    "expected_doc_ids": ["a.md"],
                },
            ],
        }
        gt_file.write_text(json.dumps(gt_data))

        code = asyncio.run(
            main(
                [
                    "evaluate",
                    "--ground-truth",
                    str(gt_file),
                ]
            )
        )
        assert code == 1
