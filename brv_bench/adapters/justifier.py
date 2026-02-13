"""Answer justifier — LLM-based answer synthesis from retrieved context.

Takes raw context (key facts) retrieved by ``brv query`` and produces
a concise answer suitable for F1/EM/LLM-Judge evaluation metrics.
"""

from __future__ import annotations

from brv_bench.metrics._judge.client import JudgeClient


class AnswerJustifier:
    """Synthesise a concise answer from retrieved context via an LLM."""

    def __init__(
        self,
        client: JudgeClient,
        prompt_template: str,
    ) -> None:
        self._client = client
        self._template = prompt_template

    async def justify(self, question: str, context: str) -> str:
        """Produce a concise answer given *question* and *context*.

        Args:
            question: The original benchmark question.
            context: Raw key-facts text from ``brv query``.

        Returns:
            Stripped LLM response text.
        """
        prompt = self._template.format(question=question, context=context)
        raw = await self._client.raw_call(prompt, max_tokens=1024)
        return raw.strip()
