"""LLM judge client abstraction.

Provides a thin async interface over LLM provider APIs so the judge
metric is backend-agnostic.  The ``anthropic``, ``openai``, and
``google-genai`` SDKs are optional dependencies — a clear error is
raised at construction time if the chosen backend is not installed.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
        data = json.loads(raw)
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
    async def judge(self, query: str, prompt: str) -> JudgeVerdict:
        """Send *prompt* to the LLM and return a verdict.

        Args:
            query: The original question (used as key in the verdict).
            prompt: Fully formatted judge prompt.
        """


# ── Anthropic ────────────────────────────────────────────────────────


class AnthropicJudgeClient(JudgeClient):
    """Judge client backed by the Anthropic Messages API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        api_key: str | None = None,
    ) -> None:
        try:
            import anthropic  # noqa: F811
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

    async def judge(self, query: str, prompt: str) -> JudgeVerdict:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        return parse_verdict(query, raw)


# ── OpenAI ───────────────────────────────────────────────────────────


class OpenAIJudgeClient(JudgeClient):
    """Judge client backed by the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        api_key: str | None = None,
    ) -> None:
        try:
            import openai  # noqa: F811
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

    async def judge(self, query: str, prompt: str) -> JudgeVerdict:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or ""
        return parse_verdict(query, raw)


# ── Gemini ───────────────────────────────────────────────────────────


class GeminiJudgeClient(JudgeClient):
    """Judge client backed by the Google Gemini API (google-genai SDK)."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        try:
            from google import genai  # noqa: F811
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

    async def judge(self, query: str, prompt: str) -> JudgeVerdict:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config={
                "temperature": 0.0,
                "max_output_tokens": 256,
            },
        )
        raw = response.text or ""
        return parse_verdict(query, raw)


# ── Factory ──────────────────────────────────────────────────────────

_BACKENDS: dict[str, type[JudgeClient]] = {
    "anthropic": AnthropicJudgeClient,
    "gemini": GeminiJudgeClient,
    "openai": OpenAIJudgeClient,
}


def create_judge_client(
    backend: str,
    model: str | None = None,
    api_key: str | None = None,
) -> JudgeClient:
    """Create a judge client by backend name.

    Args:
        backend: ``'anthropic'``, ``'gemini'``, or ``'openai'``.
        model: Optional model override (uses backend default if *None*).
        api_key: Optional API key (reads env var if *None*).

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
    if api_key is not None:
        kwargs["api_key"] = api_key
    return cls(**kwargs)
