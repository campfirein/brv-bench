"""Tests for BrvCliAdapter."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

from brv_bench.adapters.brv_cli import (
    BrvCliAdapter,
    _extract_details,
    _extract_doc_ids,
    _extract_source_from_query,
)
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
    return json.dumps(
        {
            "command": "query",
            "data": {"result": result_text, "status": "completed"},
            "success": True,
            "timestamp": "2024-01-01T00:00:00Z",
        }
    )


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
    def test_setup_is_noop(self):
        """setup() completes without error and makes no subprocess calls."""
        adapter = BrvCliAdapter(PROMPT_CONFIG)
        asyncio.run(adapter.setup())  # should not raise


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

    def test_filters_blocks_by_valid_topics(self):
        text = (
            "**Details**: \n"
            "### Session 1\nFacts about session 1\n"
            "### Session 2\nFacts about session 2\n"
            "### Session 3\nFacts about session 3\n"
            "**Sources**: x"
        )
        result = _extract_details(
            text, valid_topics={"session_1", "session_3"}
        )
        assert "Facts about session 1" in result
        assert "Facts about session 3" in result
        assert "Facts about session 2" not in result

    def test_valid_topics_none_keeps_all(self):
        text = (
            "**Details**: \n"
            "### Session 1\nFacts 1\n"
            "### Session 2\nFacts 2\n"
            "**Sources**: x"
        )
        result = _extract_details(text, valid_topics=None)
        assert "Facts 1" in result
        assert "Facts 2" in result

    def test_topic_header_normalisation(self):
        """'### Session 2' normalises to 'session_2' for matching."""
        text = "**Details**: \n### Session 2\nContent here\n**Sources**: x"
        result = _extract_details(text, valid_topics={"session_2"})
        assert "Content here" in result

    def test_topic_header_with_domain_suffix(self):
        """'### Session 30 - bf659f65' normalises to 'session_30'."""
        text = (
            "**Details**: \n"
            "### Session 30 - bf659f65\nCorrect facts\n"
            "### Session 31 - bf659f65\nOther facts\n"
            "**Sources**: x"
        )
        result = _extract_details(text, valid_topics={"session_30"})
        assert "Correct facts" in result
        assert "Other facts" not in result

    def test_yaml_frontmatter_not_split_as_block_boundary(self):
        """YAML --- inside a session block must not fragment the block."""
        text = (
            "**Details**: \n"
            "### Session 1\n"
            "---\ntitle: Key Facts\n---\n"
            "## Key Facts\n- Some fact\n"
            "### Session 2\n"
            "---\ntitle: Key Facts\n---\n"
            "## Key Facts\n- Other fact\n"
            "**Sources**: x"
        )
        result = _extract_details(text, valid_topics={"session_1"})
        assert "Some fact" in result
        assert "Other fact" not in result


# ----------------------------------------------------------------
# _extract_doc_ids
# ----------------------------------------------------------------


class TestExtractSourceFromQuery:
    def test_conversation_prefix(self):
        q = "Conversation: conv-26\nWhat happened?"
        assert _extract_source_from_query(q) == "conv-26"

    def test_question_id_prefix(self):
        q = "Question ID: gpt4_2655b836\nSome question"
        assert _extract_source_from_query(q) == "gpt4_2655b836"

    def test_no_prefix(self):
        assert _extract_source_from_query("plain question") is None


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

    def test_filters_wrong_domain_when_source_provided(self):
        text = (
            "**Sources**: .brv/context-tree/conv_26/session_1/f.md, "
            ".brv/context-tree/conv_99/session_2/f.md"
        )
        ids = _extract_doc_ids(text, source="conv_26")
        assert ids == ["session_1"]

    def test_exact_domain_match_required(self):
        text = "**Sources**: .brv/context-tree/conv_26/session_3/f.md"
        ids = _extract_doc_ids(text, source="conv_26")
        assert ids == ["session_3"]

    def test_source_none_preserves_all(self):
        text = (
            "**Sources**: .brv/context-tree/conv_26/session_1/f.md, "
            ".brv/context-tree/conv_99/session_2/f.md"
        )
        ids = _extract_doc_ids(text, source=None)
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
        context, ids = BrvCliAdapter._parse_query_response(
            raw,
            "Conversation: conv_26\nWhat happened?",
        )
        assert context == "Key facts about session"
        assert ids == ["session_1"]

    def test_filters_wrong_domain(self):
        details = (
            "\n### Session 1\nCorrect domain facts\n"
            "### Session 2\nWrong domain facts"
        )
        md = _markdown_response(
            details,
            ".brv/context-tree/conv_26/session_1/f.md, "
            ".brv/context-tree/conv_99/session_2/f.md",
        )
        raw = _query_json(md)
        context, ids = BrvCliAdapter._parse_query_response(
            raw,
            "Conversation: conv_26\nq",
        )
        assert ids == ["session_1"]
        assert "Correct domain facts" in context
        assert "Wrong domain facts" not in context

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
            result = asyncio.run(
                adapter.query("Conversation: conv_26\ntest?", 10),
            )
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
            result = asyncio.run(
                adapter.query("Conversation: c\ntest?", 10),
            )
            assert len(result.results) == 2
            assert result.results[0].path == "session_1"
            assert result.results[1].path == "session_4"

    def test_filters_wrong_domain_in_query(self):
        md = _markdown_response(
            "facts",
            ".brv/context-tree/conv_26/session_1/f.md, "
            ".brv/context-tree/conv_99/session_2/f.md",
        )
        resp = _query_json(md)
        with patch("brv_bench.adapters.brv_cli.asyncio") as mock_aio:
            mock_aio.create_subprocess_exec = AsyncMock(
                return_value=_mock_proc(0, resp),
            )
            mock_aio.subprocess = asyncio.subprocess
            adapter = BrvCliAdapter(PROMPT_CONFIG)
            result = asyncio.run(
                adapter.query("Conversation: conv_26\ntest?", 10),
            )
            assert len(result.results) == 1
            assert result.results[0].path == "session_1"

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
            result = asyncio.run(
                adapter.query("Conversation: c\ntest?", 10),
            )
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
            query = "Conversation: c\nWhat happened?"
            result = asyncio.run(adapter.query(query, 10))
            assert result.answer == "Concise answer"
            mock_justifier.justify.assert_called_once_with(
                query,
                "Key facts here",
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
            result = asyncio.run(
                adapter.query("Conversation: c\ntest?", 10),
            )
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
            result = asyncio.run(
                adapter.query("Conversation: c\ntest?", 3),
            )
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
