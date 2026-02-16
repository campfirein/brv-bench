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
    r"\.brv/context-tree/([^/]+)/([^/]+)/[^/]+\.md",
)

# Regex to extract the source identifier from a query string.
# Matches "Conversation: conv_26" or "Question ID: gpt4_2655b836".
_SOURCE_RE = re.compile(r"(?:Conversation|Question ID):\s*(\S+)")


def _extract_source_from_query(query: str) -> str | None:
    """Extract the source identifier (conversation/question ID) from a query."""
    m = _SOURCE_RE.search(query)
    return m.group(1) if m else None


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
            "query", formatted, "--format", "json",
        )
        duration_ms = (time.perf_counter() - start) * 1000

        context_text, doc_ids = self._parse_query_response(stdout, query)

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
        query: str = "",
    ) -> tuple[str, list[str]]:
        """Parse brv query JSON into (context_text, doc_ids).

        The new ``brv query`` output is structured markdown with
        ``**Details**:``, ``**Sources**:``, etc.  doc_ids are extracted
        deterministically from file paths in the Sources section.

        Args:
            raw_json: Raw JSON string from ``brv query``.
            query: Original query string used to extract the source
                identifier for domain-scoped filtering.

        Returns:
            (context_text, doc_ids) — context is the Details section,
            doc_ids are topic folder names parsed from file paths.
        """
        try:
            data = json.loads(raw_json)
            result_text = data["data"]["result"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return raw_json, []

        source = _extract_source_from_query(query)
        doc_ids = _extract_doc_ids(result_text, source=source)
        valid_topics = set(doc_ids) if source else None
        context_text = _extract_details(result_text, valid_topics=valid_topics)

        return context_text, doc_ids


def _extract_details(
    text: str,
    valid_topics: set[str] | None = None,
) -> str:
    """Extract the **Details** section from brv query markdown.

    When *valid_topics* is provided, only topic blocks whose header
    normalises to a value in the set are kept.  This prevents the
    justifier from seeing context retrieved from other source domains.

    brv query output groups each topic as::

        ### Session 2
        {content}
        ---
        ### Session 1
        {content}
    """
    match = re.search(
        r"\*\*Details\*\*:\s*(.*?)(?=\*\*Sources\*\*|\*\*Gaps\*\*|\Z)",
        text,
        re.DOTALL,
    )
    if not match:
        return text

    details = match.group(1).strip()
    if valid_topics is None:
        return details

    # Split into per-topic blocks on "---" separators and filter.
    blocks = re.split(r"\n---\n", details)
    filtered: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        header = re.match(r"###\s+(.+)", block)
        if header:
            topic = header.group(1).strip().lower().replace(" ", "_")
            if topic in valid_topics:
                filtered.append(block)
        else:
            # Keep blocks without a recognisable header (e.g. preamble).
            filtered.append(block)

    return "\n\n---\n\n".join(filtered) if filtered else details


def _extract_doc_ids(text: str, *, source: str | None = None) -> list[str]:
    """Extract doc_ids from **Sources** file paths.

    Path format: .brv/context-tree/{domain}/{topic}/{file}.md
    doc_id = {topic} (the topic folder name).

    When *source* is provided, only paths whose domain folder matches
    the source are included.
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
        domain = path_match.group(1)
        topic = path_match.group(2)
        if source and domain != source:
            continue
        if topic not in seen:
            seen.add(topic)
            doc_ids.append(topic)

    return doc_ids
