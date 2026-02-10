"""Tests for the curate command."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from brv_bench.commands.curate import (
    CurateSummary,
    collect_source_files,
    curate,
    curate_file,
)

# ---------------------------------------------------------------------------
# collect_source_files
# ---------------------------------------------------------------------------


class TestCollectSourceFiles:
    def test_collects_files(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("content a")
        (tmp_path / "b.txt").write_text("content b")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.md").write_text("content c")

        files = collect_source_files(tmp_path)

        assert len(files) == 3
        names = [f.name for f in files]
        assert "a.md" in names
        assert "b.txt" in names
        assert "c.md" in names

    def test_sorted_determinism(self, tmp_path: Path):
        (tmp_path / "z.md").write_text("z")
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "m.md").write_text("m")

        files = collect_source_files(tmp_path)
        names = [f.name for f in files]
        assert names == sorted(names)

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            collect_source_files(Path("/nonexistent/path"))

    def test_empty_dir(self, tmp_path: Path):
        files = collect_source_files(tmp_path)
        assert files == []


# ---------------------------------------------------------------------------
# curate_file (mocked subprocess)
# ---------------------------------------------------------------------------


class TestCurateFile:
    def test_success(self, tmp_path: Path):
        file = tmp_path / "test.md"
        file.write_text("content")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"OK curated", b"")

        with patch(
            "brv_bench.commands.curate.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            result = asyncio.run(curate_file(file))

        assert result.success is True
        assert result.message == "OK curated"
        assert result.file == file
        mock_exec.assert_called_once()

    def test_failure(self, tmp_path: Path):
        file = tmp_path / "bad.md"
        file.write_text("content")

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"Error: auth required")

        with patch(
            "brv_bench.commands.curate.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = asyncio.run(curate_file(file))

        assert result.success is False
        assert "auth required" in result.message


# ---------------------------------------------------------------------------
# curate (full pipeline, mocked)
# ---------------------------------------------------------------------------


class TestCurate:
    def test_all_succeed(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "b.md").write_text("b")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"OK", b"")

        with patch(
            "brv_bench.commands.curate.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            summary = asyncio.run(curate(tmp_path))

        assert isinstance(summary, CurateSummary)
        assert summary.total == 2
        assert summary.succeeded == 2
        assert summary.failed == 0

    def test_partial_failure(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "b.md").write_text("b")

        call_count = 0

        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            proc = AsyncMock()
            if call_count == 1:
                proc.returncode = 0
                proc.communicate.return_value = (b"OK", b"")
            else:
                proc.returncode = 1
                proc.communicate.return_value = (b"", b"failed")
            return proc

        with patch(
            "brv_bench.commands.curate.asyncio.create_subprocess_exec",
            side_effect=mock_exec,
        ):
            summary = asyncio.run(curate(tmp_path))

        assert summary.total == 2
        assert summary.succeeded == 1
        assert summary.failed == 1

    def test_empty_dir(self, tmp_path: Path):
        summary = asyncio.run(curate(tmp_path))

        assert summary.total == 0
        assert summary.succeeded == 0
        assert summary.failed == 0
        assert summary.results == ()
