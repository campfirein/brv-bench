"""Tests for BrvCliAdapter."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from brv_bench.adapters.brv_cli import BrvCliAdapter, _extract_details, _extract_doc_ids
from brv_bench.types import PromptConfig


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------

PROMPT_CONFIG = PromptConfig(
    curate_template="CURATE: {doc_id} {source}\n{content}",
    query_template="{question}",
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


def _query_json(result_text: str) -> str:
    """Build a brv query JSON response with markdown content."""
    return json.dumps({
        "command": "query",
        "data": {"result": result_text, "status": "completed"},
        "success": True,
        "timestamp": "2024-01-01T00:00:00Z",
    })


def _markdown_response(
    details: str,
    sources: str,
    gaps: str = "None",
) -> str:
    """Build a markdown result with Details, Sources, Gaps sections."""
    return (
        f"**Summary**: Some summary\n"
        f"**Details**: {details}\n"
        f"**Sources**: {sources}\n"
        f"**Gaps**: {gaps}"
    )


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
# _extract_details
# ----------------------------------------------------------------


class TestExtractDetails:
    def test_extracts_between_details_and_sources(self):
        text = _markdown_response(
            "Caroline went to LGBTQ group",
            ".brv/context-tree/conv_26/session_1/key_facts.md",
        )
        assert _extract_details(text) == "Caroline went to LGBTQ group"

    def test_extracts_multiline_details(self):
        text = (
            "**Details**: Line one\nLine two\nLine three\n"
            "**Sources**: .brv/context-tree/x/y/z.md"
        )
        result = _extract_details(text)
        assert "Line one" in result
        assert "Line three" in result

    def test_fallback_to_full_text_when_no_details(self):
        text = "Just some plain text without markers"
        assert _extract_details(text) == text


# ----------------------------------------------------------------
# _extract_doc_ids
# ----------------------------------------------------------------


class TestExtractDocIds:
    def test_single_path(self):
        text = "**Sources**: .brv/context-tree/conv_26/session_1/key_facts.md"
        assert _extract_doc_ids(text) == ["session_1"]

    def test_multiple_paths(self):
        text = (
            "**Sources**: .brv/context-tree/conv_26/session_1/key_facts.md, "
            ".brv/context-tree/conv_26/session_3/key_facts.md"
        )
        ids = _extract_doc_ids(text)
        assert ids == ["session_1", "session_3"]

    def test_deduplicates_paths(self):
        text = (
            "**Sources**: .brv/context-tree/conv_26/session_1/key_facts.md\n"
            ".brv/context-tree/conv_26/session_1/other.md"
        )
        ids = _extract_doc_ids(text)
        assert ids == ["session_1"]

    def test_sources_none(self):
        text = "**Sources**: None"
        assert _extract_doc_ids(text) == []

    def test_sources_none_case_insensitive(self):
        text = "**Sources**: none"
        assert _extract_doc_ids(text) == []

    def test_no_sources_section(self):
        text = "Just some text without sources"
        assert _extract_doc_ids(text) == []

    def test_multiline_sources(self):
        text = (
            "**Sources**:\n"
            "- .brv/context-tree/q1/session_1/facts.md\n"
            "- .brv/context-tree/q1/session_2/facts.md\n"
            "**Gaps**: None"
        )
        ids = _extract_doc_ids(text)
        assert ids == ["session_1", "session_2"]


# ----------------------------------------------------------------
# _parse_query_response
# ----------------------------------------------------------------


class TestParseQueryResponse:
    def test_parses_markdown_response(self):
        md = _markdown_response(
            "Key facts about session",
            ".brv/context-tree/conv_26/session_1/key_facts.md",
        )
        raw = _query_json(md)
        context, ids = BrvCliAdapter._parse_query_response(raw)
        assert context == "Key facts about session"
        assert ids == ["session_1"]

    def test_invalid_json_returns_raw(self):
        context, ids = BrvCliAdapter._parse_query_response("not json")
        assert context == "not json"
        assert ids == []

    def test_sources_none_returns_empty_list(self):
        md = _markdown_response("No info available", "None")
        raw = _query_json(md)
        context, ids = BrvCliAdapter._parse_query_response(raw)
        assert "No info available" in context
        assert ids == []

    def test_no_details_section_uses_full_result(self):
        raw = _query_json("Just plain text")
        context, ids = BrvCliAdapter._parse_query_response(raw)
        assert context == "Just plain text"
        assert ids == []


# ----------------------------------------------------------------
# query
# ----------------------------------------------------------------


class TestQuery:
    def test_parses_sources_from_paths(self):
        md = _markdown_response(
            "Some key facts",
            ".brv/context-tree/conv_26/session_1/key_facts.md",
        )
        resp = _query_json(md)
        with patch("brv_bench.adapters.brv_cli.asyncio") as mock_aio:
            mock_aio.create_subprocess_exec = AsyncMock(
                return_value=_mock_proc(0, resp),
            )
            mock_aio.subprocess = asyncio.subprocess
            adapter = BrvCliAdapter(PROMPT_CONFIG)
            result = asyncio.run(adapter.query("test?", 10))
            assert len(result.results) == 1
            assert result.results[0].path == "session_1"

    def test_multiple_source_paths(self):
        md = _markdown_response(
            "facts",
            ".brv/context-tree/c/session_1/f.md, "
            ".brv/context-tree/c/session_4/f.md",
        )
        resp = _query_json(md)
        with patch("brv_bench.adapters.brv_cli.asyncio") as mock_aio:
            mock_aio.create_subprocess_exec = AsyncMock(
                return_value=_mock_proc(0, resp),
            )
            mock_aio.subprocess = asyncio.subprocess
            adapter = BrvCliAdapter(PROMPT_CONFIG)
            result = asyncio.run(adapter.query("test?", 10))
            assert len(result.results) == 2
            assert result.results[0].path == "session_1"
            assert result.results[1].path == "session_4"

    def test_without_justifier_answer_is_context(self):
        md = _markdown_response(
            "Raw key facts content",
            ".brv/context-tree/c/session_1/f.md",
        )
        resp = _query_json(md)
        with patch("brv_bench.adapters.brv_cli.asyncio") as mock_aio:
            mock_aio.create_subprocess_exec = AsyncMock(
                return_value=_mock_proc(0, resp),
            )
            mock_aio.subprocess = asyncio.subprocess
            adapter = BrvCliAdapter(PROMPT_CONFIG)
            result = asyncio.run(adapter.query("test?", 10))
            assert result.answer == "Raw key facts content"

    def test_with_justifier(self):
        md = _markdown_response(
            "Key facts here",
            ".brv/context-tree/c/session_1/f.md",
        )
        resp = _query_json(md)

        mock_justifier = AsyncMock()
        mock_justifier.justify = AsyncMock(return_value="Concise answer")

        with patch("brv_bench.adapters.brv_cli.asyncio") as mock_aio:
            mock_aio.create_subprocess_exec = AsyncMock(
                return_value=_mock_proc(0, resp),
            )
            mock_aio.subprocess = asyncio.subprocess
            adapter = BrvCliAdapter(PROMPT_CONFIG, justifier=mock_justifier)
            result = asyncio.run(adapter.query("What happened?", 10))
            assert result.answer == "Concise answer"
            mock_justifier.justify.assert_called_once_with(
                "What happened?", "Key facts here",
            )

    def test_timing_is_positive(self):
        md = _markdown_response("x", ".brv/context-tree/c/s/f.md")
        resp = _query_json(md)
        with patch("brv_bench.adapters.brv_cli.asyncio") as mock_aio:
            mock_aio.create_subprocess_exec = AsyncMock(
                return_value=_mock_proc(0, resp),
            )
            mock_aio.subprocess = asyncio.subprocess
            adapter = BrvCliAdapter(PROMPT_CONFIG)
            result = asyncio.run(adapter.query("test?", 10))
            assert result.duration_ms > 0

    def test_limit_truncates_results(self):
        paths = ", ".join(
            f".brv/context-tree/c/topic_{i}/f.md" for i in range(5)
        )
        md = _markdown_response("answer", paths)
        resp = _query_json(md)
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
