"""Tests for the curate command."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from brv_bench.commands.curate import (
    CurateSummary,
    curate,
    curate_doc,
)
from brv_bench.types import CorpusDocument, PromptConfig

PROMPT_CONFIG = PromptConfig(
    curate_template="CURATE {doc_id} {source}\n{content}",
    query_template="QUERY {question}",
)


def _doc(doc_id: str = "d1", content: str = "hello") -> CorpusDocument:
    return CorpusDocument(doc_id=doc_id, content=content, source="s1")


# ---------------------------------------------------------------------------
# curate_doc (mocked subprocess)
# ---------------------------------------------------------------------------


class TestCurateDoc:
    def test_success(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"OK curated", b"")

        with patch(
            "brv_bench.commands.curate.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            result = asyncio.run(curate_doc(_doc(), PROMPT_CONFIG))

        assert result.success is True
        assert result.message == "OK curated"
        assert result.doc_id == "d1"
        mock_exec.assert_called_once()

    def test_failure(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"Error: auth required")

        with patch(
            "brv_bench.commands.curate.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            result = asyncio.run(curate_doc(_doc(), PROMPT_CONFIG))

        assert result.success is False
        assert "auth required" in result.message

    def test_formats_with_template(self):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"OK", b"")

        with patch(
            "brv_bench.commands.curate.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            doc = CorpusDocument(doc_id="x1", content="body", source="src")
            asyncio.run(curate_doc(doc, PROMPT_CONFIG))

        call_args = mock_exec.call_args[0]
        # The formatted content is the 3rd arg (after "brv", "curate")
        assert "CURATE x1 src" in call_args[2]
        assert "body" in call_args[2]


# ---------------------------------------------------------------------------
# curate (full pipeline, mocked)
# ---------------------------------------------------------------------------


class TestCurate:
    def test_all_succeed(self):
        corpus = (_doc("a"), _doc("b"))
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"OK", b"")

        with patch(
            "brv_bench.commands.curate.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            summary = asyncio.run(curate(corpus, PROMPT_CONFIG))

        assert isinstance(summary, CurateSummary)
        assert summary.total == 2
        assert summary.succeeded == 2
        assert summary.failed == 0

    def test_partial_failure(self):
        corpus = (_doc("a"), _doc("b"))
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
            summary = asyncio.run(curate(corpus, PROMPT_CONFIG))

        assert summary.total == 2
        assert summary.succeeded == 1
        assert summary.failed == 1

    def test_empty_corpus(self):
        summary = asyncio.run(curate((), PROMPT_CONFIG))

        assert summary.total == 0
        assert summary.succeeded == 0
        assert summary.failed == 0
        assert summary.results == ()
