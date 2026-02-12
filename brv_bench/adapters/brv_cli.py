"""BRV CLI adapter.

Bridges brv-bench to the `brv` CLI using headless JSON mode.
Queries the context tree and returns deterministic doc_ids from file paths.
An optional AnswerJustifier synthesises a concise answer via an external LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from brv_bench.adapters.base import RetrievalAdapter
from brv_bench.adapters.justifier import AnswerJustifier
from brv_bench.types import (
    PromptConfig,
    QueryExecution,
    SearchResult,
)

logger = logging.getLogger(__name__)

# Regex for a context-tree file path:
# .brv/context-tree/{domain}/{topic}/{file}.md  → doc_id = {topic}
_PATH_RE = re.compile(
    r"\.brv/context-tree/[^/]+/([^/]+)/[^/]+\.md",
)


class BrvCliAdapter(RetrievalAdapter):
    """Adapter that shells out to the brv CLI in headless mode."""

    def __init__(
        self,
        prompt_config: PromptConfig,
        justifier: AnswerJustifier | None = None,
    ) -> None:
        self._prompt_config = prompt_config
        self._justifier = justifier

    @property
    def name(self) -> str:
        return "brv-cli"

    @property
    def supports_warm_latency(self) -> bool:
        return False

    async def setup(self) -> None:
        """Verify brv CLI is available."""
        await self._verify_brv()

    async def query(self, query: str, limit: int) -> QueryExecution:
        """Run a query against the brv context tree."""
        formatted = self._prompt_config.query_template.format(
            question=query,
        )

        start = time.perf_counter()
        _, stdout = await self._run_brv(
            "query", formatted, "--headless", "--format", "json",
        )
        duration_ms = (time.perf_counter() - start) * 1000

        context_text, doc_ids = self._parse_query_response(stdout)

        if self._justifier:
            answer = await self._justifier.justify(query, context_text)
        else:
            answer = context_text

        results = tuple(
            SearchResult(
                path=doc_id,
                title=doc_id,
                score=1.0,
                excerpt="",
            )
            for doc_id in doc_ids
        )

        return QueryExecution(
            query=query,
            results=results[:limit],
            total_found=len(results),
            duration_ms=duration_ms,
            answer=answer,
        )

    async def reset(self) -> None:
        """No-op — brv CLI has no cache control."""

    async def teardown(self) -> None:
        """No-op — no persistent resources to clean up."""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _verify_brv(self) -> None:
        """Check that brv CLI is on PATH and a .brv/ project exists."""
        returncode, _ = await self._run_brv(
            "status", "--headless", "--format", "json",
        )
        if returncode != 0:
            raise RuntimeError(
                "brv CLI not available or .brv/ not initialized. "
                f"Run `brv init` first. (exit code {returncode})"
            )

    async def _run_brv(self, *args: str) -> tuple[int, str]:
        """Run a brv CLI command and return (returncode, stdout)."""
        proc = await asyncio.create_subprocess_exec(
            "brv", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        output = stdout.decode().strip()
        if not output:
            output = stderr.decode().strip()
        return proc.returncode or 0, output

    @staticmethod
    def _parse_query_response(
        raw_json: str,
    ) -> tuple[str, list[str]]:
        """Parse brv query JSON into (context_text, doc_ids).

        The new ``brv query`` output is structured markdown with
        ``**Details**:``, ``**Sources**:``, etc.  doc_ids are extracted
        deterministically from file paths in the Sources section.

        Returns:
            (context_text, doc_ids) — context is the Details section,
            doc_ids are topic folder names parsed from file paths.
        """
        try:
            data = json.loads(raw_json)
            result_text = data["data"]["result"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return raw_json, []

        context_text = _extract_details(result_text)
        doc_ids = _extract_doc_ids(result_text)

        return context_text, doc_ids


def _extract_details(text: str) -> str:
    """Extract the **Details** section from brv query markdown."""
    match = re.search(
        r"\*\*Details\*\*:\s*(.*?)(?=\*\*Sources\*\*|\*\*Gaps\*\*|\Z)",
        text,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return text


def _extract_doc_ids(text: str) -> list[str]:
    """Extract doc_ids from **Sources** file paths.

    Path format: .brv/context-tree/{domain}/{topic}/{file}.md
    doc_id = {topic} (the topic folder name).
    """
    sources_match = re.search(
        r"\*\*Sources\*\*:\s*(.*?)(?=\*\*Gaps\*\*|\*\*|\Z)",
        text,
        re.DOTALL,
    )
    if not sources_match:
        return []

    raw = sources_match.group(1).strip()
    if raw.lower() == "none":
        return []

    seen: set[str] = set()
    doc_ids: list[str] = []
    for path_match in _PATH_RE.finditer(raw):
        topic = path_match.group(1)
        if topic not in seen:
            seen.add(topic)
            doc_ids.append(topic)

    return doc_ids
