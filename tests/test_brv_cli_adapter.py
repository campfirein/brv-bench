"""Tests for BrvCliAdapter."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from brv_bench.adapters.brv_cli import BrvCliAdapter
from brv_bench.types import PromptConfig


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------

PROMPT_CONFIG = PromptConfig(
    curate_template="CURATE: {doc_id} {source}\n{content}",
    query_template="QUERY: {question}",
)


def _mock_proc(returncode: int = 0, stdout: str = "") -> AsyncMock:
    """Create a mock subprocess."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate.return_value = (
        stdout.encode(),
        b"",
    )
    return proc


def _query_json(answer: str, sources: str) -> str:
    """Build a brv query JSON response."""
    result = f"ANSWER: {answer}\nSOURCES: {sources}"
    return json.dumps({
        "command": "query",
        "data": {"result": result, "status": "completed"},
        "success": True,
        "timestamp": "2024-01-01T00:00:00Z",
    })


# ----------------------------------------------------------------
# setup
# ----------------------------------------------------------------


class TestSetup:
    def test_verifies_brv_exists(self):
        with patch("brv_bench.adapters.brv_cli.asyncio") as mock_aio:
            mock_aio.create_subprocess_exec = AsyncMock(
                return_value=_mock_proc(0, '{"success":true}'),
            )
            adapter = BrvCliAdapter(PROMPT_CONFIG)
            asyncio.run(adapter.setup())
            call_args = mock_aio.create_subprocess_exec.call_args
            assert call_args[0][:2] == ("brv", "status")

    def test_fails_without_brv(self):
        with patch("brv_bench.adapters.brv_cli.asyncio") as mock_aio:
            mock_aio.create_subprocess_exec = AsyncMock(
                return_value=_mock_proc(1, "not found"),
            )
            adapter = BrvCliAdapter(PROMPT_CONFIG)
            with pytest.raises(RuntimeError, match="brv CLI not available"):
                asyncio.run(adapter.setup())


# ----------------------------------------------------------------
# query
# ----------------------------------------------------------------


class TestQuery:
    def test_parses_structured_response(self):
        resp = _query_json("Max", "conv-26_s1")
        with patch("brv_bench.adapters.brv_cli.asyncio") as mock_aio:
            mock_aio.create_subprocess_exec = AsyncMock(
                return_value=_mock_proc(0, resp),
            )
            mock_aio.subprocess = asyncio.subprocess
            adapter = BrvCliAdapter(PROMPT_CONFIG)
            result = asyncio.run(
                adapter.query("What is the puppy's name?", 10)
            )
            assert result.answer == "Max"
            assert len(result.results) == 1
            assert result.results[0].path == "conv-26_s1"

    def test_parses_multiple_sources(self):
        resp = _query_json("counseling", "conv-26_s1, conv-26_s4")
        with patch("brv_bench.adapters.brv_cli.asyncio") as mock_aio:
            mock_aio.create_subprocess_exec = AsyncMock(
                return_value=_mock_proc(0, resp),
            )
            mock_aio.subprocess = asyncio.subprocess
            adapter = BrvCliAdapter(PROMPT_CONFIG)
            result = asyncio.run(adapter.query("Career?", 10))
            assert result.answer == "counseling"
            assert len(result.results) == 2
            assert result.results[0].path == "conv-26_s1"
            assert result.results[1].path == "conv-26_s4"

    def test_fallback_on_unstructured_response(self):
        raw = json.dumps({
            "command": "query",
            "data": {"result": "Some free-form answer", "status": "completed"},
            "success": True,
        })
        with patch("brv_bench.adapters.brv_cli.asyncio") as mock_aio:
            mock_aio.create_subprocess_exec = AsyncMock(
                return_value=_mock_proc(0, raw),
            )
            mock_aio.subprocess = asyncio.subprocess
            adapter = BrvCliAdapter(PROMPT_CONFIG)
            result = asyncio.run(adapter.query("test?", 10))
            assert result.answer == "Some free-form answer"
            assert result.results == ()

    def test_timing_is_positive(self):
        resp = _query_json("yes", "conv-26_s1")
        with patch("brv_bench.adapters.brv_cli.asyncio") as mock_aio:
            mock_aio.create_subprocess_exec = AsyncMock(
                return_value=_mock_proc(0, resp),
            )
            mock_aio.subprocess = asyncio.subprocess
            adapter = BrvCliAdapter(PROMPT_CONFIG)
            result = asyncio.run(adapter.query("test?", 10))
            assert result.duration_ms > 0

    def test_limit_truncates_results(self):
        sources = ", ".join(f"doc_{i}" for i in range(5))
        resp = _query_json("answer", sources)
        with patch("brv_bench.adapters.brv_cli.asyncio") as mock_aio:
            mock_aio.create_subprocess_exec = AsyncMock(
                return_value=_mock_proc(0, resp),
            )
            mock_aio.subprocess = asyncio.subprocess
            adapter = BrvCliAdapter(PROMPT_CONFIG)
            result = asyncio.run(adapter.query("test?", 3))
            assert len(result.results) == 3
            assert result.total_found == 5


# ----------------------------------------------------------------
# reset / teardown
# ----------------------------------------------------------------


class TestResetTeardown:
    def test_reset_noop(self):
        adapter = BrvCliAdapter(PROMPT_CONFIG)
        asyncio.run(adapter.reset())

    def test_teardown_noop(self):
        adapter = BrvCliAdapter(PROMPT_CONFIG)
        asyncio.run(adapter.teardown())


# ----------------------------------------------------------------
# _parse_query_response
# ----------------------------------------------------------------


class TestParseQueryResponse:
    def test_valid_json(self):
        raw = _query_json("hello", "doc_1, doc_2")
        answer, ids = BrvCliAdapter._parse_query_response(raw)
        assert answer == "hello"
        assert ids == ["doc_1", "doc_2"]

    def test_invalid_json_returns_raw(self):
        answer, ids = BrvCliAdapter._parse_query_response("not json")
        assert answer == "not json"
        assert ids == []

    def test_missing_answer_line_uses_full_result(self):
        raw = json.dumps({
            "command": "query",
            "data": {"result": "just text", "status": "completed"},
            "success": True,
        })
        answer, ids = BrvCliAdapter._parse_query_response(raw)
        assert answer == "just text"
        assert ids == []
