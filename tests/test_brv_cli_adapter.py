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


def _query_json(
    result_text: str,
    matched_docs: list[dict] | None = None,
) -> str:
    """Build a brv query JSON response matching the real ``--format json`` shape.

    Real brv emits one JSON object per line, with `data.event == "completed"`
    on the terminal event of every successful query. This helper produces
    a single-line completed event (the Tier 2 / cache-hit shape); use
    `_query_ndjson_stream()` for multi-event Tier 3 / 4 streams.

    When ``matched_docs`` is provided, the structured ``matchedDocs`` array is
    attached to the completed event — the post-HTML-migration response shape
    and the bench's preferred doc-id source. Without it, callers still get
    the legacy regex-on-prose path.
    """
    data: dict = {"event": "completed", "result": result_text, "status": "completed"}
    if matched_docs is not None:
        data["matchedDocs"] = matched_docs
    return json.dumps(
        {
            "command": "query",
            "data": data,
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


# ----------------------------------------------------------------
# HTML-migration: matchedDocs primary source + .html / 2-segment paths
# ----------------------------------------------------------------


class TestMatchedDocsPrimary:
    """After the HTML migration `brv query --format json` emits a structured
    ``data.matchedDocs[]`` array on the completed event. The adapter prefers
    this over regex-on-prose because it is format-agnostic (no .md vs .html
    coupling), tier-agnostic (Tier 2 dot-rendered prose AND free-form Tier 3
    LLM responses both surface the same matchedDocs), and survives the
    response-shape changes that broke the legacy parser."""

    def test_extracts_doc_ids_from_matched_docs_html(self):
        md = _markdown_response("free-form llm answer", "anything")
        matched = [
            {"path": "conv_26/session_1.html", "score": 0.85, "title": "Session 1"},
            {"path": "conv_26/session_3.html", "score": 0.82, "title": "Session 3"},
        ]
        raw = _query_json(md, matched_docs=matched)
        context, ids = BrvCliAdapter._parse_query_response(
            raw, "Conversation: conv_26\nq?"
        )
        assert ids == ["session_1", "session_3"]

    def test_matched_docs_filters_by_source(self):
        matched = [
            {"path": "conv_26/session_1.html", "score": 0.9, "title": "S1"},
            {"path": "conv_99/session_2.html", "score": 0.85, "title": "S2"},
        ]
        raw = _query_json(_markdown_response("x", "y"), matched_docs=matched)
        _, ids = BrvCliAdapter._parse_query_response(
            raw, "Conversation: conv_26\nq?"
        )
        assert ids == ["session_1"]

    def test_matched_docs_dedups(self):
        matched = [
            {"path": "conv_26/session_1.html", "score": 0.9, "title": "S1"},
            {"path": "conv_26/session_1.html", "score": 0.85, "title": "S1"},
        ]
        raw = _query_json(_markdown_response("x", "y"), matched_docs=matched)
        _, ids = BrvCliAdapter._parse_query_response(raw, "")
        assert ids == ["session_1"]

    def test_matched_docs_handles_legacy_md_paths(self):
        # Mixed-format result: HTML (new) + .md (legacy still on disk).
        # Both must be recognised so partially migrated trees don't lose
        # legacy doc_ids from the recall.
        matched = [
            {"path": "conv_26/session_1.html", "score": 0.9, "title": "S1"},
            {"path": "conv_26/session_2/key_facts.md", "score": 0.85, "title": "S2"},
        ]
        raw = _query_json(_markdown_response("x", "y"), matched_docs=matched)
        _, ids = BrvCliAdapter._parse_query_response(raw, "Conversation: conv_26\nq?")
        assert ids == ["session_1", "session_2"]

    def test_matched_docs_trumps_sources_block_when_both_present(self):
        # Bench should consume matchedDocs as primary. Sources-block parsing
        # is a defensive fallback for unusual / legacy responses; it must NOT
        # override matchedDocs when both are present, otherwise the regex
        # path's failure modes leak back into the new-format flow.
        md = _markdown_response(
            "details", ".brv/context-tree/conv_26/old_topic/key_facts.md"
        )
        matched = [
            {"path": "conv_26/session_1.html", "score": 0.9, "title": "S1"}
        ]
        raw = _query_json(md, matched_docs=matched)
        _, ids = BrvCliAdapter._parse_query_response(raw, "Conversation: conv_26\nq?")
        assert ids == ["session_1"]

    def test_falls_back_to_sources_regex_when_matched_docs_absent(self):
        # Cache-hit / tier 0/1 responses may omit matchedDocs. The Sources-
        # block regex remains the defensive fallback.
        md = _markdown_response(
            "answer",
            ".brv/context-tree/conv_26/session_5/key_facts.md",
        )
        raw = _query_json(md)  # no matched_docs
        _, ids = BrvCliAdapter._parse_query_response(raw, "Conversation: conv_26\nq?")
        assert ids == ["session_5"]


class TestExtractDocIdsHtmlAware:
    """`_extract_doc_ids` is the regex fallback. Post-HTML-migration the
    Sources block in `data.result` carries ``.html`` 2-segment paths — the
    regex must accept both new and legacy shapes."""

    def test_html_2segment_path(self):
        text = "**Sources**: .brv/context-tree/conv_26/session_1.html"
        assert _extract_doc_ids(text) == ["session_1"]

    def test_legacy_md_3segment_path(self):
        text = "**Sources**: .brv/context-tree/conv_26/session_1/key_facts.md"
        assert _extract_doc_ids(text) == ["session_1"]

    def test_mixed_html_and_md_in_sources(self):
        text = (
            "**Sources**:\n"
            "- .brv/context-tree/conv_26/session_1.html\n"
            "- .brv/context-tree/conv_26/session_2/key_facts.md\n"
        )
        assert _extract_doc_ids(text) == ["session_1", "session_2"]

    def test_html_path_filters_wrong_domain(self):
        text = (
            "**Sources**: .brv/context-tree/conv_26/session_1.html, "
            ".brv/context-tree/conv_99/session_2.html"
        )
        assert _extract_doc_ids(text, source="conv_26") == ["session_1"]

    def test_htm_extension_legacy_html(self):
        text = "**Sources**: .brv/context-tree/conv_26/session_1.htm"
        assert _extract_doc_ids(text) == ["session_1"]


# ----------------------------------------------------------------
# NDJSON: brv query --format json emits one event per line
# ----------------------------------------------------------------


class TestParseNdjsonStream:
    """`brv query --format json` emits one JSON object per line — one
    per lifecycle event (`thinking`, `toolCall`, `response`,
    `completed`). Tier 2 (direct response, no LLM) emits a single
    `completed` line; Tier 3/4 emit several event lines and the
    `completed` line is the LAST one. Earlier `_parse_query_response`
    called `json.loads(whole_stdout)` which raised on multi-line
    streams, falling through to the empty-result branch and silently
    losing the matchedDocs payload of every Tier 3/4 query."""

    def _ndjson_tier3(self) -> str:
        """Realistic multi-line stdout the way Tier 3 emits it."""
        events = [
            {"command": "query", "data": {"event": "thinking", "taskId": "t1"}},
            {
                "command": "query",
                "data": {
                    "args": {"code": "// search and read", "silent": True},
                    "event": "toolCall",
                    "taskId": "t1",
                    "toolName": "code_exec",
                },
            },
            {
                "command": "query",
                "data": {
                    "event": "toolResult",
                    "success": True,
                    "taskId": "t1",
                    "toolName": "code_exec",
                },
            },
            {
                "command": "query",
                "data": {
                    "content": "**Summary**: ...\n**Details**: ...\n**Sources**:\n- `.brv/context-tree/conv_26/session_2.html`",
                    "event": "response",
                    "taskId": "t1",
                },
            },
            {
                "command": "query",
                "data": {
                    "durationMs": 7093,
                    "event": "completed",
                    "matchedDocs": [
                        {"path": "conv_26/session_2.html", "score": 0.85, "title": "Session 2"},
                        {"path": "conv_26/session_8.html", "score": 0.66, "title": "Session 8"},
                    ],
                    "result": "**Details**: charity race for mental health.\n**Sources**:\n- .brv/context-tree/conv_26/session_2.html",
                    "status": "completed",
                    "taskId": "t1",
                    "tier": 3,
                    "topScore": 0.85,
                },
            },
        ]
        return "\n".join(json.dumps(e) for e in events)

    def test_tier3_multiline_extracts_doc_ids_from_completed_event(self):
        raw = self._ndjson_tier3()
        context, ids = BrvCliAdapter._parse_query_response(
            raw, "Conversation: conv_26\nq?"
        )
        # Doc-ids come from matchedDocs on the `completed` line, not the prose
        # of any earlier event, and not by trying to json.loads the whole
        # stream (which would raise).
        assert ids == ["session_2", "session_8"]
        assert "charity race for mental health" in context

    def test_tier3_multiline_filter_by_source(self):
        # Same NDJSON, but with a foreign-domain doc mixed in
        events = json.loads(self._ndjson_tier3().splitlines()[-1])
        events["data"]["matchedDocs"].insert(
            1, {"path": "conv_99/session_2.html", "score": 0.7, "title": "wrong"}
        )
        raw = (
            "\n".join(self._ndjson_tier3().splitlines()[:-1])
            + "\n"
            + json.dumps(events)
        )
        _, ids = BrvCliAdapter._parse_query_response(
            raw, "Conversation: conv_26\nq?"
        )
        assert ids == ["session_2", "session_8"]

    def test_tier3_picks_last_completed_event_when_multiple(self):
        # Defensive: if the stream emits multiple completed events (only ever
        # happens on retry / restart paths), the LAST one wins because that's
        # the final state.
        first_completed = {
            "command": "query",
            "data": {
                "event": "completed",
                "matchedDocs": [
                    {"path": "conv_26/session_x.html", "score": 0.5, "title": "stale"}
                ],
                "result": "stale",
                "status": "completed",
                "taskId": "t1",
                "tier": 3,
            },
        }
        last_completed = {
            "command": "query",
            "data": {
                "event": "completed",
                "matchedDocs": [
                    {"path": "conv_26/session_y.html", "score": 0.9, "title": "fresh"}
                ],
                "result": "fresh",
                "status": "completed",
                "taskId": "t1",
                "tier": 3,
            },
        }
        raw = json.dumps(first_completed) + "\n" + json.dumps(last_completed)
        _, ids = BrvCliAdapter._parse_query_response(raw, "")
        assert ids == ["session_y"]

    def test_tier2_single_line_still_works(self):
        # Regression guard — the previously-supported Tier 2 single-event
        # stream must keep parsing.
        completed = {
            "command": "query",
            "data": {
                "durationMs": 6,
                "event": "completed",
                "matchedDocs": [
                    {"path": "conv_26/session_1.html", "score": 0.95, "title": "S1"}
                ],
                "result": "**Details**: x\n**Sources**:\n- .brv/context-tree/conv_26/session_1.html",
                "status": "completed",
                "taskId": "t1",
                "tier": 2,
            },
        }
        raw = json.dumps(completed)
        _, ids = BrvCliAdapter._parse_query_response(raw, "")
        assert ids == ["session_1"]

    def test_blank_lines_in_stream_are_tolerated(self):
        completed = {
            "command": "query",
            "data": {
                "event": "completed",
                "matchedDocs": [
                    {"path": "conv_26/session_3.html", "score": 1.0, "title": "S3"}
                ],
                "result": "x",
                "status": "completed",
                "taskId": "t1",
                "tier": 2,
            },
        }
        raw = "\n\n" + json.dumps(completed) + "\n\n"
        _, ids = BrvCliAdapter._parse_query_response(raw, "")
        assert ids == ["session_3"]

    def test_corrupt_event_line_is_skipped_not_fatal(self):
        # If one event line is malformed (rare but possible — e.g., a
        # truncated chunk from a flushed pipe), the parser should keep
        # walking the rest of the stream rather than aborting.
        completed = {
            "command": "query",
            "data": {
                "event": "completed",
                "matchedDocs": [
                    {"path": "conv_26/session_4.html", "score": 1.0, "title": "S4"}
                ],
                "result": "x",
                "status": "completed",
                "taskId": "t1",
                "tier": 2,
            },
        }
        raw = "{not-valid-json\n" + json.dumps(completed)
        _, ids = BrvCliAdapter._parse_query_response(raw, "")
        assert ids == ["session_4"]

    def test_no_completed_event_falls_back_to_raw(self):
        # If the stream genuinely has no completed event (degenerate), keep
        # the previous (raw_json, []) behaviour rather than throwing.
        raw = json.dumps({"command": "query", "data": {"event": "thinking"}})
        context, ids = BrvCliAdapter._parse_query_response(raw, "")
        assert ids == []
        # context preserves the raw input in the absence of a result
        assert context == raw
