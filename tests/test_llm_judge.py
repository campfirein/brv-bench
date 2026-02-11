"""Tests for brv_bench.metrics.llm_judge and brv_bench.metrics._judge.*."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from brv_bench.metrics._judge.client import (
    AnthropicJudgeClient,
    GeminiJudgeClient,
    JudgeClient,
    JudgeVerdict,
    OpenAIJudgeClient,
    create_judge_client,
    parse_verdict,
)
from brv_bench.metrics._judge.prompts import DEFAULT_JUDGE_PROMPT
from brv_bench.metrics.llm_judge import (
    LLMJudge,
    _cache_key,
    _load_cache,
    _save_cache,
)
from brv_bench.types import GroundTruthEntry, QueryExecution, SearchResult

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _qe(
    query: str,
    answer: str | None = None,
) -> QueryExecution:
    return QueryExecution(
        query=query,
        results=(
            SearchResult(
                path="doc.md", title="doc", score=1.0, excerpt="text"
            ),
        ),
        total_found=1,
        duration_ms=5.0,
        answer=answer,
    )


def _gt(
    query: str,
    expected_answer: str | None = None,
    category: str = "unspecified",
) -> GroundTruthEntry:
    return GroundTruthEntry(
        query=query,
        expected_doc_ids=("doc.md",),
        category=category,
        expected_answer=expected_answer,
    )


class MockJudgeClient(JudgeClient):
    """In-memory mock that returns preset verdicts by query text."""

    def __init__(
        self,
        verdicts: dict[str, bool] | None = None,
        *,
        default: bool = True,
    ) -> None:
        self._verdicts = verdicts or {}
        self._default = default
        self.calls: list[str] = []

    async def judge(self, query: str, prompt: str) -> JudgeVerdict:
        self.calls.append(query)
        is_correct = self._verdicts.get(query, self._default)
        return JudgeVerdict(
            query=query,
            is_correct=is_correct,
            reasoning="mock",
        )


# ----------------------------------------------------------------
# parse_verdict
# ----------------------------------------------------------------


class TestParseVerdict:
    def test_correct_verdict(self):
        raw = '{"reasoning": "Matches.", "verdict": "correct"}'
        v = parse_verdict("q1", raw)
        assert v.is_correct is True
        assert v.query == "q1"
        assert v.reasoning == "Matches."

    def test_incorrect_verdict(self):
        raw = '{"reasoning": "Wrong.", "verdict": "incorrect"}'
        v = parse_verdict("q1", raw)
        assert v.is_correct is False

    def test_case_insensitive_verdict(self):
        raw = '{"reasoning": "ok", "verdict": "CORRECT"}'
        v = parse_verdict("q1", raw)
        assert v.is_correct is True

    def test_extra_whitespace(self):
        raw = '{"reasoning": "ok", "verdict": " correct "}'
        v = parse_verdict("q1", raw)
        assert v.is_correct is True

    def test_invalid_json_degrades_to_incorrect(self):
        v = parse_verdict("q1", "not json at all")
        assert v.is_correct is False
        assert "Failed to parse" in v.reasoning

    def test_missing_verdict_key_degrades_to_incorrect(self):
        raw = '{"reasoning": "oops"}'
        v = parse_verdict("q1", raw)
        assert v.is_correct is False

    def test_empty_string_degrades_to_incorrect(self):
        v = parse_verdict("q1", "")
        assert v.is_correct is False


# ----------------------------------------------------------------
# LLMJudge.compute — core scoring
# ----------------------------------------------------------------


class TestLLMJudgeCompute:
    def test_all_correct(self):
        client = MockJudgeClient(default=True)
        metric = LLMJudge(client=client)
        pairs = [
            (_qe("q1", answer="Paris"), _gt("q1", expected_answer="Paris")),
            (_qe("q2", answer="Tokyo"), _gt("q2", expected_answer="Tokyo")),
        ]
        [result] = metric.compute(pairs)
        assert result.value == 1.0
        assert result.name == "llm-judge"
        assert result.label == "LLM Judge"
        assert result.unit == "ratio"

    def test_all_incorrect(self):
        client = MockJudgeClient(default=False)
        metric = LLMJudge(client=client)
        pairs = [
            (_qe("q1", answer="London"), _gt("q1", expected_answer="Paris")),
            (_qe("q2", answer="Seoul"), _gt("q2", expected_answer="Tokyo")),
        ]
        [result] = metric.compute(pairs)
        assert result.value == 0.0

    def test_mixed_verdicts(self):
        client = MockJudgeClient(verdicts={"q1": True, "q2": False})
        metric = LLMJudge(client=client)
        pairs = [
            (_qe("q1", answer="Paris"), _gt("q1", expected_answer="Paris")),
            (_qe("q2", answer="Seoul"), _gt("q2", expected_answer="Tokyo")),
        ]
        [result] = metric.compute(pairs)
        assert result.value == pytest.approx(0.5)

    def test_skips_none_answer(self):
        client = MockJudgeClient(default=True)
        metric = LLMJudge(client=client)
        pairs = [
            (_qe("q1", answer=None), _gt("q1", expected_answer="Paris")),
            (_qe("q2", answer="Tokyo"), _gt("q2", expected_answer="Tokyo")),
        ]
        [result] = metric.compute(pairs)
        # Only q2 is scorable → 1 correct / 1 total = 1.0
        assert result.value == 1.0
        assert len(client.calls) == 1

    def test_skips_none_expected_answer(self):
        client = MockJudgeClient(default=True)
        metric = LLMJudge(client=client)
        pairs = [
            (_qe("q1", answer="Paris"), _gt("q1", expected_answer=None)),
            (_qe("q2", answer="Tokyo"), _gt("q2", expected_answer="Tokyo")),
        ]
        [result] = metric.compute(pairs)
        assert result.value == 1.0
        assert len(client.calls) == 1

    def test_skips_empty_answer(self):
        client = MockJudgeClient(default=True)
        metric = LLMJudge(client=client)
        pairs = [
            (_qe("q1", answer=""), _gt("q1", expected_answer="Paris")),
        ]
        [result] = metric.compute(pairs)
        # Empty string is falsy → filtered out → no scorable pairs
        assert result.value == 0.0
        assert len(client.calls) == 0

    def test_empty_pairs(self):
        client = MockJudgeClient()
        metric = LLMJudge(client=client)
        [result] = metric.compute([])
        assert result.value == 0.0
        assert len(client.calls) == 0

    def test_metric_id(self):
        metric = LLMJudge(client=MockJudgeClient())
        assert metric.id == "llm-judge"


# ----------------------------------------------------------------
# LLMJudge.compute — caching
# ----------------------------------------------------------------


class TestLLMJudgeCache:
    def test_cache_saves_and_loads(self, tmp_path: Path):
        cache_file = tmp_path / "cache.json"
        client = MockJudgeClient(verdicts={"q1": True})
        metric = LLMJudge(client=client, cache_path=cache_file)

        pairs = [
            (_qe("q1", answer="Paris"), _gt("q1", expected_answer="Paris")),
        ]
        metric.compute(pairs)

        # Cache file should exist
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert len(data) == 1
        entry = next(iter(data.values()))
        assert entry["is_correct"] is True

    def test_cached_verdicts_skip_api_calls(self, tmp_path: Path):
        cache_file = tmp_path / "cache.json"
        client = MockJudgeClient(verdicts={"q1": True})
        metric = LLMJudge(client=client, cache_path=cache_file)

        pairs = [
            (_qe("q1", answer="Paris"), _gt("q1", expected_answer="Paris")),
        ]

        # First run — calls the API
        metric.compute(pairs)
        assert len(client.calls) == 1

        # Second run — should use cache, no new API calls
        client.calls.clear()
        metric.compute(pairs)
        assert len(client.calls) == 0

    def test_partial_cache_hit(self, tmp_path: Path):
        cache_file = tmp_path / "cache.json"
        client = MockJudgeClient(verdicts={"q1": True, "q2": False})
        metric = LLMJudge(client=client, cache_path=cache_file)

        pairs_1 = [
            (_qe("q1", answer="Paris"), _gt("q1", expected_answer="Paris")),
        ]
        metric.compute(pairs_1)
        assert len(client.calls) == 1

        # Now add q2 — only q2 should trigger an API call
        client.calls.clear()
        pairs_2 = [
            (_qe("q1", answer="Paris"), _gt("q1", expected_answer="Paris")),
            (_qe("q2", answer="Seoul"), _gt("q2", expected_answer="Tokyo")),
        ]
        [result] = metric.compute(pairs_2)
        assert len(client.calls) == 1
        assert client.calls[0] == "q2"
        assert result.value == pytest.approx(0.5)

    def test_corrupt_cache_ignored(self, tmp_path: Path):
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("not valid json{{{")

        client = MockJudgeClient(verdicts={"q1": True})
        metric = LLMJudge(client=client, cache_path=cache_file)

        pairs = [
            (_qe("q1", answer="Paris"), _gt("q1", expected_answer="Paris")),
        ]
        [result] = metric.compute(pairs)
        assert result.value == 1.0
        # Should have called API since cache was corrupt
        assert len(client.calls) == 1

    def test_no_cache_by_default(self):
        client = MockJudgeClient(verdicts={"q1": True})
        metric = LLMJudge(client=client)

        pairs = [
            (_qe("q1", answer="Paris"), _gt("q1", expected_answer="Paris")),
        ]
        # Should work fine without cache
        [result] = metric.compute(pairs)
        assert result.value == 1.0


# ----------------------------------------------------------------
# Cache key determinism
# ----------------------------------------------------------------


class TestCacheKey:
    def test_same_inputs_same_key(self):
        k1 = _cache_key("q", "a", "b")
        k2 = _cache_key("q", "a", "b")
        assert k1 == k2

    def test_different_inputs_different_key(self):
        k1 = _cache_key("q", "a", "b")
        k2 = _cache_key("q", "a", "c")
        assert k1 != k2

    def test_key_is_hex_string(self):
        key = _cache_key("q", "a", "b")
        assert isinstance(key, str)
        int(key, 16)  # Should not raise


# ----------------------------------------------------------------
# Cache persistence helpers
# ----------------------------------------------------------------


class TestCachePersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        cache_file = tmp_path / "sub" / "cache.json"
        verdicts = {
            "key1": JudgeVerdict(query="q1", is_correct=True, reasoning="yes"),
            "key2": JudgeVerdict(query="q2", is_correct=False, reasoning="no"),
        }
        _save_cache(cache_file, verdicts)
        loaded = _load_cache(cache_file)
        assert len(loaded) == 2
        assert loaded["key1"].is_correct is True
        assert loaded["key2"].is_correct is False
        assert loaded["key1"].query == "q1"

    def test_load_nonexistent_returns_empty(self, tmp_path: Path):
        result = _load_cache(tmp_path / "nope.json")
        assert result == {}

    def test_load_corrupt_returns_empty(self, tmp_path: Path):
        cache_file = tmp_path / "bad.json"
        cache_file.write_text("{bad}")
        result = _load_cache(cache_file)
        assert result == {}

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        cache_file = tmp_path / "a" / "b" / "c" / "cache.json"
        _save_cache(cache_file, {})
        assert cache_file.exists()


# ----------------------------------------------------------------
# Prompt template
# ----------------------------------------------------------------


class TestPromptTemplate:
    def test_default_template_has_placeholders(self):
        assert "{question}" in DEFAULT_JUDGE_PROMPT
        assert "{expected_answer}" in DEFAULT_JUDGE_PROMPT
        assert "{predicted_answer}" in DEFAULT_JUDGE_PROMPT

    def test_default_template_formats_cleanly(self):
        result = DEFAULT_JUDGE_PROMPT.format(
            question="What city?",
            expected_answer="Paris",
            predicted_answer="Paris, France",
        )
        assert "What city?" in result
        assert "Paris" in result
        assert "Paris, France" in result

    def test_none_template_uses_default(self):
        client = MockJudgeClient(default=True)
        metric = LLMJudge(client=client, prompt_template=None)
        pairs = [
            (_qe("q1", answer="Paris"), _gt("q1", expected_answer="Paris")),
        ]
        [result] = metric.compute(pairs)
        assert result.value == 1.0
        # Verify the default template was used (client was called)
        assert len(client.calls) == 1

    def test_custom_template(self):
        template = (
            "Q: {question}\nA: {expected_answer}\n"
            "P: {predicted_answer}\nJudge:"
        )
        client = MockJudgeClient(default=True)
        metric = LLMJudge(client=client, prompt_template=template)
        pairs = [
            (_qe("q1", answer="Paris"), _gt("q1", expected_answer="Paris")),
        ]
        [result] = metric.compute(pairs)
        assert result.value == 1.0


# ----------------------------------------------------------------
# Concurrency
# ----------------------------------------------------------------


class TestConcurrency:
    def test_concurrency_limits_parallel_calls(self):
        """Verify the metric completes with limited concurrency."""
        client = MockJudgeClient(default=True)
        metric = LLMJudge(client=client, concurrency=2)
        pairs = [
            (
                _qe(f"q{i}", answer=f"a{i}"),
                _gt(f"q{i}", expected_answer=f"a{i}"),
            )
            for i in range(10)
        ]
        [result] = metric.compute(pairs)
        assert result.value == 1.0
        assert len(client.calls) == 10


# ----------------------------------------------------------------
# create_judge_client factory
# ----------------------------------------------------------------


class TestCreateJudgeClient:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown judge backend"):
            create_judge_client("unknown")

    def test_anthropic_backend_without_sdk(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                create_judge_client("anthropic", api_key="key")

    def test_gemini_backend_without_sdk(self):
        with patch.dict("sys.modules", {"google": None, "google.genai": None}):
            with pytest.raises(ImportError, match="google-genai"):
                create_judge_client("gemini", api_key="key")

    def test_openai_backend_without_sdk(self):
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                create_judge_client("openai", api_key="key")


# ----------------------------------------------------------------
# AnthropicJudgeClient
# ----------------------------------------------------------------


class TestAnthropicJudgeClient:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        mock_mod = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_mod}):
            with pytest.raises(ValueError, match="API key"):
                AnthropicJudgeClient(api_key=None)

    def test_sends_correct_params(self, monkeypatch: pytest.MonkeyPatch):
        mock_mod = MagicMock()
        mock_async_client = MagicMock()
        mock_mod.AsyncAnthropic.return_value = mock_async_client

        # Simulate response
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"reasoning": "ok", "verdict": "correct"}')
        ]
        mock_async_client.messages.create = AsyncMock(
            return_value=mock_response
        )

        with patch.dict("sys.modules", {"anthropic": mock_mod}):
            client = AnthropicJudgeClient(api_key="test-key")

        import asyncio

        verdict = asyncio.run(client.judge("q1", "judge this"))

        mock_async_client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-5-20250929",
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": "judge this"}],
        )
        assert verdict.is_correct is True
        assert verdict.query == "q1"


# ----------------------------------------------------------------
# OpenAIJudgeClient
# ----------------------------------------------------------------


class TestOpenAIJudgeClient:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_mod = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_mod}):
            with pytest.raises(ValueError, match="API key"):
                OpenAIJudgeClient(api_key=None)

    def test_sends_correct_params(self, monkeypatch: pytest.MonkeyPatch):
        mock_mod = MagicMock()
        mock_async_client = MagicMock()
        mock_mod.AsyncOpenAI.return_value = mock_async_client

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"reasoning": "wrong", "verdict": "incorrect"}'
                )
            )
        ]
        mock_async_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        with patch.dict("sys.modules", {"openai": mock_mod}):
            client = OpenAIJudgeClient(api_key="test-key")

        import asyncio

        verdict = asyncio.run(client.judge("q1", "judge this"))

        mock_async_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-2024-08-06",
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": "judge this"}],
        )
        assert verdict.is_correct is False
        assert verdict.query == "q1"


# ----------------------------------------------------------------
# GeminiJudgeClient
# ----------------------------------------------------------------


class TestGeminiJudgeClient:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        mock_google = MagicMock()
        mock_genai = MagicMock()
        mock_google.genai = mock_genai
        with patch.dict(
            "sys.modules",
            {"google": mock_google, "google.genai": mock_genai},
        ):
            with pytest.raises(ValueError, match="API key"):
                GeminiJudgeClient(api_key=None)

    def test_sends_correct_params(self, monkeypatch: pytest.MonkeyPatch):
        mock_google = MagicMock()
        mock_genai = MagicMock()
        mock_google.genai = mock_genai
        mock_client_instance = MagicMock()
        mock_genai.Client.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_response.text = '{"reasoning": "matches", "verdict": "correct"}'
        mock_client_instance.aio.models.generate_content = AsyncMock(
            return_value=mock_response,
        )

        with patch.dict(
            "sys.modules",
            {"google": mock_google, "google.genai": mock_genai},
        ):
            client = GeminiJudgeClient(api_key="test-key")

        import asyncio

        verdict = asyncio.run(client.judge("q1", "judge this"))

        mock_client_instance.aio.models.generate_content.assert_called_once_with(
            model="gemini-2.5-flash",
            contents="judge this",
            config={
                "temperature": 0.0,
                "max_output_tokens": 256,
            },
        )
        assert verdict.is_correct is True
        assert verdict.query == "q1"


# ----------------------------------------------------------------
# CLI flags (smoke test via argparse)
# ----------------------------------------------------------------


class TestCLIFlags:
    def test_judge_flags_in_help(self):
        from brv_bench.__main__ import parse_args

        with pytest.raises(SystemExit):
            parse_args(["evaluate", "--help"])

    def test_judge_defaults(self):
        from brv_bench.__main__ import parse_args

        args = parse_args(
            ["evaluate", "--ground-truth", "data.json", "--judge"]
        )
        assert args.judge is True
        assert args.judge_backend == "gemini"
        assert args.judge_model is None
        assert args.judge_concurrency == 5
        assert args.judge_cache is None

    def test_judge_custom_options(self):
        from brv_bench.__main__ import parse_args

        args = parse_args(
            [
                "evaluate",
                "--ground-truth",
                "data.json",
                "--judge",
                "--judge-backend",
                "openai",
                "--judge-model",
                "gpt-4o",
                "--judge-concurrency",
                "10",
                "--judge-cache",
                "/tmp/cache.json",
            ]
        )
        assert args.judge_backend == "openai"
        assert args.judge_model == "gpt-4o"
        assert args.judge_concurrency == 10
        assert args.judge_cache == Path("/tmp/cache.json")

    def test_no_judge_flag_defaults_false(self):
        from brv_bench.__main__ import parse_args

        args = parse_args(["evaluate", "--ground-truth", "data.json"])
        assert args.judge is False
