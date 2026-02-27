"""Tests for brv_bench.adapters.justifier."""

import asyncio
from unittest.mock import AsyncMock

from brv_bench.adapters.justifier import AnswerJustifier
from brv_bench.metrics._judge.client import JudgeClient, JudgeVerdict


# ----------------------------------------------------------------
# Mock client
# ----------------------------------------------------------------


class MockRawClient(JudgeClient):
    """Mock that records prompts and returns a preset response."""

    def __init__(self, response: str = "mock answer") -> None:
        self._response = response
        self.prompts: list[str] = []

    async def raw_call(self, prompt: str, *, max_tokens: int = 512) -> str:
        self.prompts.append(prompt)
        return self._response


# ----------------------------------------------------------------
# AnswerJustifier
# ----------------------------------------------------------------


class TestAnswerJustifier:
    def test_formats_template_and_returns_answer(self):
        client = MockRawClient(response="  Paris  ")
        template = "Context: {context}\nQ: {question}\nAnswer:"
        justifier = AnswerJustifier(client=client, prompt_template=template)

        result = asyncio.run(justifier.justify("Where?", "France facts"))

        assert result == "Paris"
        assert len(client.prompts) == 1
        assert "Context: France facts" in client.prompts[0]
        assert "Q: Where?" in client.prompts[0]

    def test_empty_context(self):
        client = MockRawClient(
            response="I don't have enough information to answer this question."
        )
        template = "{context}\n{question}"
        justifier = AnswerJustifier(client=client, prompt_template=template)

        result = asyncio.run(justifier.justify("What?", ""))

        assert "don't have enough information" in result
        assert client.prompts[0].startswith("\n")

    def test_strips_whitespace(self):
        client = MockRawClient(response="\n  answer \n ")
        template = "{context} {question}"
        justifier = AnswerJustifier(client=client, prompt_template=template)

        result = asyncio.run(justifier.justify("q", "c"))
        assert result == "answer"
