"""LLM judge client abstraction.

Provides a thin async interface over LLM provider APIs so the judge
metric is backend-agnostic.  The ``anthropic``, ``openai``,
``google-genai``, and ``ollama`` SDKs are optional dependencies — a
clear error is raised at construction time if the chosen backend is
not installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from brv_bench.metrics._judge.constants import (
    ANTHROPIC_ADAPTIVE_EFFORT,
    ANTHROPIC_ADAPTIVE_MODELS,
    ANTHROPIC_DEFAULT_MODEL,
    ANTHROPIC_ENABLED_PREFIXES,
    ANTHROPIC_THINKING_BUDGET,
    GEMINI_DEFAULT_MODEL,
    GEMINI_MAX_RETRIES,
    GEMINI_RETRY_INITIAL_WAIT,
    GEMINI_THINKING_BUDGET_DISABLED,
    JUDGE_MAX_TOKENS,
    OLLAMA_DEFAULT_HOST,
    OLLAMA_DEFAULT_MODEL,
    OPENAI_DEFAULT_MODEL,
    RAW_CALL_DEFAULT_MAX_TOKENS,
)

# ── Verdict ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JudgeVerdict:
    """Single judge evaluation result."""

    query: str
    is_correct: bool
    reasoning: str


def parse_verdict(query: str, raw: str) -> JudgeVerdict:
    """Parse raw LLM output into a *JudgeVerdict*.

    Falls back to ``is_correct=False`` when the response cannot be
    parsed — this is intentional graceful degradation so a single
    malformed response does not crash the entire evaluation run.
    """
    try:
        # Strip markdown code fences that some models wrap around JSON output.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        data = json.loads(cleaned)
        verdict = str(data.get("verdict", "")).lower().strip()
        reasoning = str(data.get("reasoning", ""))
        return JudgeVerdict(
            query=query,
            is_correct=verdict == "correct",
            reasoning=reasoning,
        )
    except (json.JSONDecodeError, AttributeError, TypeError):
        return JudgeVerdict(
            query=query,
            is_correct=False,
            reasoning=f"Failed to parse judge response: {raw!r}",
        )


# ── Client ABC ───────────────────────────────────────────────────────


class JudgeClient(ABC):
    """Abstract async LLM judge client."""

    @abstractmethod
    async def raw_call(
        self, prompt: str, *, max_tokens: int = RAW_CALL_DEFAULT_MAX_TOKENS
    ) -> str:
        """Send *prompt* to the LLM and return raw text.

        Args:
            prompt: Fully formatted prompt.
            max_tokens: Maximum tokens in the response.
        """

    async def judge(self, query: str, prompt: str) -> JudgeVerdict:
        """Send *prompt* to the LLM and return a verdict.

        Args:
            query: The original question (used as key in the verdict).
            prompt: Fully formatted judge prompt.
        """
        raw = await self.raw_call(prompt, max_tokens=JUDGE_MAX_TOKENS)
        return parse_verdict(query, raw)


# ── Anthropic ────────────────────────────────────────────────────────


def _anthropic_thinking_mode(model: str) -> str | None:
    """Return 'adaptive' (Opus/Sonnet 4.6), 'enabled' (other 4/3.7), or None."""
    if any(p in model for p in ANTHROPIC_ADAPTIVE_MODELS):
        return "adaptive"
    if any(p in model for p in ANTHROPIC_ENABLED_PREFIXES):
        return "enabled"
    return None


class AnthropicJudgeClient(JudgeClient):
    """Judge client backed by the Anthropic Messages API."""

    def __init__(
        self,
        model: str = ANTHROPIC_DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic "
                "judge backend.  Install it with: "
                "pip install 'brv-bench[judge]'"
            ) from exc

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "An Anthropic API key is required.  Set the "
                "ANTHROPIC_API_KEY environment variable or pass "
                "api_key= explicitly."
            )
        self._client = anthropic.AsyncAnthropic(api_key=resolved_key)
        self._model = model

    async def raw_call(
        self, prompt: str, *, max_tokens: int = RAW_CALL_DEFAULT_MAX_TOKENS
    ) -> str:
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Thinking config — temperature must be omitted when thinking is active.
        # Opus/Sonnet 4.6: adaptive thinking; effort goes in output_config
        #   (separate top-level param), NOT inside the thinking dict.
        # Other Claude 4 / 3.7: manual extended thinking with minimum budget.
        mode = _anthropic_thinking_mode(self._model)
        if mode == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": ANTHROPIC_ADAPTIVE_EFFORT}
        elif mode == "enabled":
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": ANTHROPIC_THINKING_BUDGET,
            }
        else:
            kwargs["temperature"] = 0.0

        # Use streaming to avoid the SDK's 10-minute timeout guard that
        # triggers when max_tokens is large (e.g. 32 768 for the justifier).
        async with self._client.messages.stream(**kwargs) as stream:
            message = await stream.get_final_message()
        # Response may contain thinking blocks; return the first text block.
        for block in message.content:
            if block.type == "text":
                return block.text
        return ""


# ── OpenAI ───────────────────────────────────────────────────────────

# Both o-series and GPT-5 series are reasoning models in the Chat Completions
# API and share the same parameter shape:
#   - max_completion_tokens  (not max_tokens)
#   - reasoning_effort="..."  (top-level string — NOT nested reasoning object;
#     the nested reasoning={effort:...} format is the Responses API only)
#   - no temperature
#
# Effort values differ by model:
#   o-series (o1/o3/o4-mini): low | medium | high   → use "low"
#   gpt-5 (original):         minimal | low | medium | high → use "minimal"
#   gpt-5.1 / gpt-5.2 / gpt-5.3 / gpt-5.x-codex*: none | low | medium | high | xhigh → use "none"
#
# * Codex models (gpt-5.2-codex, gpt-5.3-codex) currently require the Responses
#   API, not Chat Completions. They match the gpt-5 prefix but will fail at the
#   API level if passed here — which is the correct behaviour.
#
# Standard models (gpt-4o etc.) use max_tokens + temperature=0.


def _openai_model_class(model: str) -> str:
    """Classify OpenAI model for API parameter selection.

    Returns:
        'reasoning' — o-series or GPT-5 series (max_completion_tokens + reasoning_effort)
        'standard'  — everything else (max_tokens + temperature=0)
    """
    if model.startswith("gpt-5") or re.match(r"^o\d", model):
        return "reasoning"
    return "standard"


def _openai_min_effort(model: str) -> str:
    """Return the lowest reasoning_effort value for a given reasoning model.

    o-series:        "low"     (minimum available; none/minimal not supported)
    gpt-5 original:  "minimal" (unique to gpt-5 and gpt-5-* variants)
    gpt-5.1+:        "none"    (gpt-5.1, gpt-5.2, gpt-5.3, gpt-5.x-codex, etc.)
    """
    if re.match(r"^o\d", model):  # o-series
        return "low"
    if model == "gpt-5" or model.startswith("gpt-5-"):
        return "minimal"
    return "none"


class OpenAIJudgeClient(JudgeClient):
    """Judge client backed by the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str = OPENAI_DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for the OpenAI "
                "judge backend.  Install it with: "
                "pip install 'brv-bench[judge]'"
            ) from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "An OpenAI API key is required.  Set the "
                "OPENAI_API_KEY environment variable or pass "
                "api_key= explicitly."
            )
        self._client = openai.AsyncOpenAI(api_key=resolved_key)
        self._model = model

    async def raw_call(
        self, prompt: str, *, max_tokens: int = RAW_CALL_DEFAULT_MAX_TOKENS
    ) -> str:
        kwargs: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }

        model_class = _openai_model_class(self._model)

        if model_class == "reasoning":
            # o-series and GPT-5: top-level reasoning_effort string, no temperature.
            kwargs["max_completion_tokens"] = max_tokens
            kwargs["reasoning_effort"] = _openai_min_effort(self._model)
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = 0.0

        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


# ── Gemini ───────────────────────────────────────────────────────────


class GeminiJudgeClient(JudgeClient):
    """Judge client backed by the Google Gemini API (google-genai SDK)."""

    def __init__(
        self,
        model: str = GEMINI_DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required for the Gemini "
                "judge backend.  Install it with: "
                "pip install 'brv-bench[judge]'"
            ) from exc

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "A Gemini API key is required.  Set the "
                "GEMINI_API_KEY environment variable or pass "
                "api_key= explicitly."
            )
        self._client = genai.Client(api_key=resolved_key)
        self._model = model

    async def raw_call(
        self, prompt: str, *, max_tokens: int = RAW_CALL_DEFAULT_MAX_TOKENS
    ) -> str:
        from google.genai import errors as genai_errors

        config: dict = {
            "temperature": 0.0,
            "max_output_tokens": max_tokens,
        }

        # Reduce thinking overhead — judge task is simple classification.
        # Gemini 3 models use thinkingLevel; 2.5 models use thinkingBudget.
        if "gemini-3" in self._model:
            config["thinking_config"] = {"thinking_level": "low"}
        else:
            config["thinking_config"] = {
                "thinking_budget": GEMINI_THINKING_BUDGET_DISABLED
            }

        max_retries = GEMINI_MAX_RETRIES
        wait = GEMINI_RETRY_INITIAL_WAIT
        for attempt in range(max_retries + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                return response.text or ""
            except (genai_errors.ServerError, genai_errors.ClientError) as exc:
                retryable = (
                    isinstance(exc, genai_errors.ServerError) and exc.code == 503
                ) or (
                    isinstance(exc, genai_errors.ClientError) and exc.code == 429
                )
                if not retryable or attempt == max_retries:
                    raise
                logger.warning(
                    "Gemini %s (attempt %d/%d) — retrying in %.0fs: %s",
                    exc.code, attempt + 1, max_retries, wait, exc,
                )
                await asyncio.sleep(wait)
                wait = min(wait * 2, 300.0)


# ── Ollama ───────────────────────────────────────────────────────────


class OllamaJudgeClient(JudgeClient):
    """Judge client backed by an Ollama server via the official SDK.

    Uses the ``ollama`` Python package which natively handles thinking
    mode: the SDK separates reasoning (``message.thinking``) from the
    final answer (``message.content``), so no manual tag stripping is
    needed.

    Thinking is disabled by default (``think=False``) since the judge
    and justifier tasks are simple classification / synthesis.
    """

    def __init__(
        self,
        model: str = OLLAMA_DEFAULT_MODEL,
        host: str = OLLAMA_DEFAULT_HOST,
        think: bool = False,
    ) -> None:
        if not model:
            raise ValueError(
                "A model name is required for the ollama backend.  "
                "Pass --judge-model / --justifier-model "
                "(e.g. 'qwen3.5:9b')."
            )
        try:
            import ollama as _ollama
        except ImportError as exc:
            raise ImportError(
                "The 'ollama' package is required for the Ollama "
                "judge backend.  Install it with: pip install ollama"
            ) from exc

        self._client = _ollama.AsyncClient(host=host)
        self._model = model
        self._think = think

    async def raw_call(
        self, prompt: str, *, max_tokens: int = RAW_CALL_DEFAULT_MAX_TOKENS
    ) -> str:
        response = await self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            think=self._think,
            options={
                "num_predict": max_tokens,
                "num_ctx": max_tokens,
                "temperature": 0.0,
            },
        )
        return response.message.content or ""


# ── Factory ──────────────────────────────────────────────────────────

_BACKENDS: dict[str, type[JudgeClient]] = {
    "anthropic": AnthropicJudgeClient,
    "gemini": GeminiJudgeClient,
    "ollama": OllamaJudgeClient,
    "openai": OpenAIJudgeClient,
}


def create_judge_client(
    backend: str,
    model: str | None = None,
    api_key: str | None = None,
    host: str | None = None,
) -> JudgeClient:
    """Create a judge client by backend name.

    Args:
        backend: ``'anthropic'``, ``'gemini'``, ``'ollama'``, or
            ``'openai'``.
        model: Optional model override (uses backend default if *None*).
        api_key: Optional API key (reads env var if *None*).
        host: Ollama server host for the ``ollama`` backend
            (defaults to ``http://localhost:11434``).

    Raises:
        ValueError: If *backend* is not recognised.
    """
    cls = _BACKENDS.get(backend)
    if cls is None:
        supported = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"Unknown judge backend {backend!r}.  "
            f"Supported backends: {supported}"
        )
    kwargs: dict[str, str] = {}
    if model is not None:
        kwargs["model"] = model
    if api_key is not None and backend != "ollama":
        kwargs["api_key"] = api_key
    if host is not None and backend == "ollama":
        kwargs["host"] = host
    return cls(**kwargs)
