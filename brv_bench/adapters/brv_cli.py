"""BRV CLI adapter.

Bridges brv-bench to the `brv` CLI using headless JSON mode.
Queries the context tree to produce answers with source doc_ids.
Curation is handled separately by the curate command.
"""

import asyncio
import json
import logging
import re
import time

from brv_bench.adapters.base import RetrievalAdapter
from brv_bench.types import (
    PromptConfig,
    QueryExecution,
    SearchResult,
)

logger = logging.getLogger(__name__)


class BrvCliAdapter(RetrievalAdapter):
    """Adapter that shells out to the brv CLI in headless mode."""

    def __init__(self, prompt_config: PromptConfig) -> None:
        self._prompt_config = prompt_config

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

        answer, doc_ids = self._parse_query_response(stdout)

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
        """Parse brv query JSON response into (answer, source_doc_ids).

        Expected JSON format:
        {"command":"query","data":{"result":"...","status":"completed"},
         "success":true,"timestamp":"..."}

        The result text should contain ANSWER: and SOURCES: lines
        from the prompt template. Falls back to raw text if unparseable.
        """
        try:
            data = json.loads(raw_json)
            result_text = data["data"]["result"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return raw_json, []

        answer = ""
        doc_ids: list[str] = []

        answer_match = re.search(
            r"ANSWER:\s*(.+?)(?:\n|$)", result_text,
        )
        if answer_match:
            answer = answer_match.group(1).strip()

        sources_match = re.search(
            r"SOURCES:\s*(.+?)(?:\n|$)", result_text,
        )
        if sources_match:
            raw_sources = sources_match.group(1).strip()
            doc_ids = [
                s.strip()
                for s in raw_sources.split(",")
                if s.strip()
            ]

        if not answer:
            answer = result_text

        return answer, doc_ids
