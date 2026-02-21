"""BRV CLI adapter.

Bridges brv-bench to the `brv` CLI using headless JSON mode.
Queries the context tree and returns deterministic doc_ids from file paths.
An optional AnswerJustifier synthesises a concise answer via an external LLM.

Isolated mode (``context_tree_source`` set):
    For each query the relevant domain folder is copied from the pre-curated
    source tree into ``.brv/context-tree/``, the query is run, then the
    domain folder is deleted.  This keeps the live context tree blank between
    queries and prevents any cross-question contamination.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
from pathlib import Path

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

    #: Live context tree written by brv.
    _CONTEXT_TREE = Path(".brv/context-tree")

    def __init__(
        self,
        prompt_config: PromptConfig,
        justifier: AnswerJustifier | None = None,
        context_tree_source: Path | None = None,
    ) -> None:
        self._prompt_config = prompt_config
        self._justifier = justifier
        self._context_tree_source = context_tree_source

    @property
    def name(self) -> str:
        return "brv-cli"

    @property
    def supports_warm_latency(self) -> bool:
        return False

    async def setup(self) -> None:
        """Verify brv CLI is available."""
        # await self._verify_brv()

    async def query(self, query: str, limit: int) -> QueryExecution:
        """Run a query against the brv context tree.

        In isolated mode the domain folder is copied from the source tree
        before querying and removed immediately afterwards.
        """
        source: str | None = None
        if self._context_tree_source is not None:
            source = _extract_source_from_query(query)
            if source:
                self._copy_domain(source)
            else:
                logger.warning(
                    "Isolated mode: could not extract domain from query; "
                    "context tree unchanged."
                )

        formatted = self._prompt_config.query_template.format(
            question=query,
        )

        start = time.perf_counter()
        _, stdout = await self._run_brv(
            "query", formatted, "--format", "json",
        )
        duration_ms = (time.perf_counter() - start) * 1000

        context_text, doc_ids = self._parse_query_response(stdout, query)

        # Clean up before justifying — the justifier only needs context_text.
        if self._context_tree_source is not None and source:
            self._remove_domain(source)

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
        """In isolated mode, ensure the live context tree is empty."""
        if self._context_tree_source is not None:
            if self._CONTEXT_TREE.exists():
                shutil.rmtree(self._CONTEXT_TREE)
                self._CONTEXT_TREE.mkdir(parents=True, exist_ok=True)
                logger.debug("Isolated mode: cleared live context tree")

    async def teardown(self) -> None:
        """No-op — no persistent resources to clean up."""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _copy_domain(self, domain: str) -> None:
        """Copy domain folder from source tree into the live context tree."""
        src = self._context_tree_source / domain  # type: ignore[operator]
        dst = self._CONTEXT_TREE / domain
        if not src.exists():
            logger.warning(
                "Isolated mode: source domain folder not found: %s", src
            )
            return
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        logger.debug("Isolated mode: copied %s → %s", src, dst)

    def _remove_domain(self, domain: str) -> None:
        """Delete the domain folder from the live context tree."""
        dst = self._CONTEXT_TREE / domain
        if dst.exists():
            shutil.rmtree(dst)
            logger.debug("Isolated mode: removed %s", dst)

    async def _verify_brv(self) -> None:
        """Check that brv CLI is on PATH and a .brv/ project exists."""
        returncode, _ = await self._run_brv(
            "status", "-f", "json",
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

        ### Session 2 - domain_id
        {content}
        ### Session 1 - domain_id
        {content}

    Each block may contain YAML frontmatter delimited by ``---``, so
    blocks are split on ``### Session`` headers rather than ``---`` to
    avoid fragmenting the frontmatter.
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

    # Split on ### Session headers to avoid splitting YAML frontmatter ---.
    blocks = re.split(r"(?m)^(?=### Session\b)", details)
    filtered: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        header = re.match(r"###\s+(.+)", block)
        if header:
            raw = header.group(1).strip()
            # Extract canonical "session_N" from e.g. "Session 30 - domain_id"
            # or "Session 26, Date: ..." — ignore any suffix after the number.
            session_m = re.match(r"session[_\s]+(\d+)", raw, re.IGNORECASE)
            topic = (
                f"session_{session_m.group(1)}" if session_m
                else raw.lower().replace(" ", "_")
            )
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
