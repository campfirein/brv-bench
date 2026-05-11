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

# Regex for a context-tree file path. Supports both layouts the brv writer has
# emitted across the HTML migration:
#   .brv/context-tree/{domain}/{topic}.html         → doc_id = {topic}  (post-T3, current default)
#   .brv/context-tree/{domain}/{topic}.htm          → doc_id = {topic}  (defensive: legacy html ext)
#   .brv/context-tree/{domain}/{topic}/{file}.md    → doc_id = {topic}  (legacy markdown layout, may
#                                                                       still exist on partially-migrated
#                                                                       trees)
# The bench prefers the structured `data.matchedDocs[]` event for doc-id
# extraction; this regex is the defensive fallback when the JSON event
# omits matchedDocs (cache hits, tier 0/1).
_PATH_RE = re.compile(
    r"\.brv/context-tree/([^/]+)/([^/]+?)(?:\.html?|/[^/]+\.md)",
)

# Regex to extract the source identifier from a query string.
# Matches "Conversation: conv_26" or "Question ID: gpt4_2655b836".
_SOURCE_RE = re.compile(r"(?:Conversation|Question ID):\s*(\S+)")


def _extract_source_from_query(query: str) -> str | None:
    """Extract the source identifier (conversation/question ID) from a query."""
    m = _SOURCE_RE.search(query)
    return m.group(1) if m else None


class BrvCliAdapter(RetrievalAdapter):
    """Adapter that shells out to the brv CLI in headless mode.

    For two-config orchestration (T6 `brv-bench run`), each adapter instance
    can be pinned to a specific byterover-cli checkout via `working_dir`
    (the subprocess's cwd) and `brv_bin` (the exact binary path, e.g.
    `<checkout>/bin/run.js`). This lets two different brv versions run
    side-by-side without `npm link` conflicts on the global PATH.

    Defaults are unchanged: when neither is set, `brv` is taken from PATH
    and subprocess inherits the bench's cwd — matches the single-config
    flow `brv-bench curate` / `brv-bench evaluate` have always used.
    """

    def __init__(
        self,
        prompt_config: PromptConfig,
        justifier: AnswerJustifier | None = None,
        context_tree_source: Path | None = None,
        working_dir: Path | None = None,
        brv_bin: str = "brv",
    ) -> None:
        self._prompt_config = prompt_config
        self._justifier = justifier
        self._context_tree_source = context_tree_source
        self._working_dir = Path(working_dir) if working_dir is not None else None
        self._brv_bin = brv_bin

    @property
    def working_dir(self) -> Path | None:
        """The subprocess cwd this adapter runs `brv` under, or None for inherit."""
        return self._working_dir

    @property
    def brv_bin(self) -> str:
        return self._brv_bin

    @property
    def context_tree_path(self) -> Path:
        """Absolute path to this adapter's `.brv/context-tree/`. Used by
        isolated-mode copy/remove and by the orchestrator's pre-run cleanup."""
        base = self._working_dir if self._working_dir is not None else Path(".")
        return base / ".brv" / "context-tree"

    @property
    def name(self) -> str:
        return "brv-cli"

    @property
    def supports_warm_latency(self) -> bool:
        return False

    async def setup(self) -> None:
        """No-op — brv availability is verified lazily on first query."""

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
            "query",
            formatted,
            "--format",
            "json",
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
            if self.context_tree_path.exists():
                shutil.rmtree(self.context_tree_path)
                self.context_tree_path.mkdir(parents=True, exist_ok=True)
                logger.debug("Isolated mode: cleared live context tree")

    async def teardown(self) -> None:
        """No-op — no persistent resources to clean up."""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _copy_domain(self, domain: str) -> None:
        """Copy domain folder from source tree into the live context tree."""
        src = self._context_tree_source / domain  # type: ignore[operator]
        dst = self.context_tree_path / domain
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
        dst = self.context_tree_path / domain
        if dst.exists():
            shutil.rmtree(dst)
            logger.debug("Isolated mode: removed %s", dst)

    async def _verify_brv(self) -> None:
        """Check that brv CLI is on PATH and a .brv/ project exists."""
        returncode, _ = await self._run_brv(
            "status",
            "-f",
            "json",
        )
        if returncode != 0:
            raise RuntimeError(
                "brv CLI not available or .brv/ not initialized. "
                f"Run `brv init` first. (exit code {returncode})"
            )

    async def _run_brv(self, *args: str) -> tuple[int, str]:
        """Run a brv CLI command and return (returncode, stdout).

        Subprocess runs in `self._working_dir` if set (so brv reads the
        per-config `.brv/` project), otherwise inherits the bench's cwd.
        Invokes `self._brv_bin` (default `"brv"` from PATH) so two
        different brv checkouts can be exercised side-by-side via their
        per-checkout `bin/run.js` entrypoints.
        """
        proc = await asyncio.create_subprocess_exec(
            self._brv_bin,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._working_dir) if self._working_dir is not None else None,
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

        ``brv query --format json`` emits one JSON object per line — one for
        each lifecycle event (``thinking``, ``toolCall``, ``toolResult``,
        ``response``, ``completed``). Tier 2 (direct-response, no LLM)
        emits a single ``completed`` line; Tier 3 / Tier 4 emit several
        events with ``completed`` last. We walk the stream, collect every
        valid event, and use the LAST ``event == "completed"`` payload as
        the source of truth (matches brv's "final state wins" semantics
        when retries / restarts surface multiple completions).

        Doc-id source preference (post-HTML-migration):
            1. ``data.matchedDocs[].path`` — structured event payload, present
               on every Tier 2 / Tier 3 / Tier 4 completion. Format-agnostic
               (``.md`` and ``.html`` both classify cleanly via path parsing)
               and tier-agnostic.
            2. ``data.result`` regex on the ``**Sources**:`` block — defensive
               fallback for cache-hit responses (Tier 0/1) that may omit
               matchedDocs, or for unusual response shapes.

        Context-text source: always the ``**Details**:`` block from
        ``data.result``. The optional contamination filter
        (``valid_topics``) reuses the resolved doc_ids so isolated-mode tests
        get a domain-scoped justifier context regardless of which Tier
        produced the response.

        Args:
            raw_json: Raw JSON-or-NDJSON string from ``brv query``.
            query: Original query string used to extract the source
                identifier for domain-scoped filtering.

        Returns:
            (context_text, doc_ids) — context is the Details section,
            doc_ids are topic names extracted from matchedDocs paths.
        """
        completed_payload = _find_completed_event(raw_json)
        if completed_payload is None:
            return raw_json, []

        result_text = completed_payload.get("result", "") or ""
        source = _extract_source_from_query(query)

        matched_docs = completed_payload.get("matchedDocs")
        if isinstance(matched_docs, list) and matched_docs:
            doc_ids = _doc_ids_from_matched_docs(matched_docs, source=source)
        else:
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
                f"session_{session_m.group(1)}"
                if session_m
                else raw.lower().replace(" ", "_")
            )
            if topic in valid_topics:
                filtered.append(block)
        else:
            # Keep blocks without a recognisable header (e.g. preamble).
            filtered.append(block)

    return "\n\n---\n\n".join(filtered)


def _find_completed_event(raw_stdout: str) -> dict | None:
    """Walk a brv query NDJSON stream and return the LAST `completed` event's
    `data` payload, or `None` if no completed event is present.

    `brv query --format json` emits one JSON object per line. For Tier 2 the
    stream is a single completed line; for Tier 3 / Tier 4 it interleaves
    `thinking`, `toolCall`, `toolResult`, `response` events with the final
    `completed`. A whole-stream `json.loads(raw_stdout)` raises on multi-line
    input, which is how the bench previously dropped every Tier 3 / 4 query's
    matchedDocs.

    Defensive: malformed lines (truncated chunks from a flushed pipe, stray
    blank lines) are skipped rather than aborting the parse. If multiple
    completed events appear (rare — only on retry / restart paths), the last
    one wins because it represents the final state.
    """
    completed: dict | None = None
    for line in raw_stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        payload = event.get("data")
        if not isinstance(payload, dict):
            continue
        if payload.get("event") == "completed":
            completed = payload

    return completed


def _doc_ids_from_matched_docs(
    matched_docs: list[dict],
    *,
    source: str | None = None,
) -> list[str]:
    """Extract doc_ids from the structured ``data.matchedDocs[]`` array.

    Each entry's ``path`` is relative to the context-tree root, e.g.
    ``conv_26/session_1.html`` (post-T3, the writer's flat one-file-per-topic
    layout) or ``conv_26/session_1/key_facts.md`` (legacy 3-segment layout).
    The first path segment is the domain; the topic name is the basename
    (without extension) for HTML, or the second segment for legacy markdown.

    Shared-source paths (``[alias]:rel/path.html``) are accepted but the
    alias prefix is stripped before parsing so they classify identically
    to local paths.

    When *source* is provided, only paths whose domain matches are included.
    Order is preserved (the brv search ranks paths by score; the bench's
    retrieval-quality metrics expect that order to flow through).
    """
    seen: set[str] = set()
    doc_ids: list[str] = []
    for entry in matched_docs:
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue

        # Strip shared-source `[alias]:` prefix when present.
        if raw_path.startswith("["):
            colon = raw_path.find(":")
            if colon != -1:
                raw_path = raw_path[colon + 1 :]

        parts = raw_path.split("/")
        if len(parts) < 2:
            continue
        domain = parts[0]
        if source and domain != source:
            continue

        # New 2-segment layout: domain/topic.<ext>  → topic = basename minus ext.
        # Legacy 3-segment layout: domain/topic/file.md → topic = parts[1].
        if len(parts) == 2:
            topic_with_ext = parts[1]
            dot = topic_with_ext.rfind(".")
            topic = topic_with_ext[:dot] if dot != -1 else topic_with_ext
        else:
            topic = parts[1]

        if topic and topic not in seen:
            seen.add(topic)
            doc_ids.append(topic)

    return doc_ids


def _extract_doc_ids(text: str, *, source: str | None = None) -> list[str]:
    """Extract doc_ids from **Sources** file paths.

    Defensive regex fallback for responses without ``data.matchedDocs[]``
    (cache-hit Tier 0/1, or unusual shapes). The regex accepts both the
    post-T3 2-segment HTML layout and the legacy 3-segment markdown layout;
    doc_id is always the topic name (basename minus extension for HTML;
    second path segment for legacy markdown).

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
