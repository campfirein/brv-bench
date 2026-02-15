"""LLM-as-Judge metric.

Uses an LLM to evaluate whether a predicted answer is semantically
correct given the question and expected answer.  This is the primary
answer-quality metric, following the methodology of LongMemEval
(ICLR 2025, 97 % human agreement with GPT-4o judge) and Hindsight
(arXiv 2512.12818).

Scoring is binary (correct / incorrect) per question, averaged across
the evaluation set to produce a single accuracy value in [0, 1].

The metric is designed to work within the synchronous ``Metric.compute``
contract while delegating to async LLM APIs.  The sync/async bridge
uses a ``ThreadPoolExecutor`` with its own event loop so that it
operates safely even when called from an already-running async context
(as ``evaluate()`` does).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from pathlib import Path

from brv_bench.metrics._judge.client import JudgeClient, JudgeVerdict
from brv_bench.metrics._judge.prompts import (
    get_judge_prompt,
)
from brv_bench.metrics.base import Metric
from brv_bench.types import GroundTruthEntry, MetricResult, QueryExecution

logger = logging.getLogger(__name__)

# ── Cache helpers ────────────────────────────────────────────────────


def _cache_key(
    question: str,
    expected_answer: str,
    predicted_answer: str,
    category: str = "unspecified",
) -> str:
    """Deterministic cache key from the judge inputs."""
    blob = json.dumps(
        [question, expected_answer, predicted_answer, category],
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_cache(path: Path) -> dict[str, JudgeVerdict]:
    """Load verdict cache from *path*, returning empty dict on error."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
        return {
            key: JudgeVerdict(
                query=entry["query"],
                is_correct=entry["is_correct"],
                reasoning=entry["reasoning"],
            )
            for key, entry in raw.items()
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Corrupt judge cache at %s — ignoring.", path)
        return {}


def _save_cache(
    path: Path,
    verdicts: dict[str, JudgeVerdict],
) -> None:
    """Persist verdict cache to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        key: {
            "query": v.query,
            "is_correct": v.is_correct,
            "reasoning": v.reasoning,
        }
        for key, v in verdicts.items()
    }
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)


# ── Metric ───────────────────────────────────────────────────────────


class LLMJudge(Metric):
    """LLM-as-Judge answer correctness metric.

    Args:
        client: Backend-specific judge client (Anthropic / OpenAI).
        prompt_template: Prompt template with ``{question}``,
            ``{expected_answer}``, and ``{predicted_answer}``
            placeholders.
        concurrency: Maximum number of parallel LLM API calls.
        cache_path: Optional path to a JSON file for caching verdicts
            across runs.  When set, only uncached pairs trigger API
            calls, and new verdicts are appended to the cache file.
    """

    def __init__(
        self,
        client: JudgeClient,
        prompt_template: str | None = None,
        concurrency: int = 5,
        cache_path: Path | None = None,
    ) -> None:
        self._client = client
        # None ⇒ use category-specific prompts via get_judge_prompt().
        # Explicit string ⇒ override for all categories (backward compat).
        self._prompt_template = prompt_template
        self._concurrency = concurrency
        self._cache_path = cache_path
        self._last_verdicts: dict[str, JudgeVerdict] = {}

    @property
    def id(self) -> str:
        return "llm-judge"

    # ── Public API ───────────────────────────────────────────────

    def compute(
        self,
        pairs: list[tuple[QueryExecution, GroundTruthEntry]],
    ) -> list[MetricResult]:
        scorable = [
            (qe, gt) for qe, gt in pairs if qe.answer and gt.expected_answer
        ]
        if not scorable:
            return [
                MetricResult(
                    name=self.id,
                    label="LLM Judge",
                    value=0.0,
                    unit="ratio",
                )
            ]

        # Load existing cache (if configured).
        cached: dict[str, JudgeVerdict] = {}
        if self._cache_path is not None:
            cached = _load_cache(self._cache_path)

        # Partition into cached / uncached.
        uncached: list[tuple[QueryExecution, GroundTruthEntry]] = []
        cache_keys: dict[int, str] = {}  # index in scorable → key
        for idx, (qe, gt) in enumerate(scorable):
            assert gt.expected_answer is not None  # guarded above
            key = _cache_key(
                gt.query,
                gt.expected_answer,
                qe.answer,
                gt.category,
            )
            cache_keys[idx] = key
            if key not in cached:
                uncached.append((qe, gt))

        # Judge uncached pairs via async-in-thread bridge.
        if uncached:
            new_verdicts = self._run_async(self._judge_all(uncached))
            cached.update(new_verdicts)
            if self._cache_path is not None:
                _save_cache(self._cache_path, cached)

        # Store verdicts keyed by query for report enrichment.
        # Accumulate (don't reset) so category-breakdown calls don't
        # clobber verdicts from the overall compute() pass.
        for idx, (qe, gt) in enumerate(scorable):
            key = cache_keys[idx]
            self._last_verdicts[gt.query] = cached[key]

        # Aggregate scores.
        scores: list[float] = []
        for idx in range(len(scorable)):
            key = cache_keys[idx]
            verdict = cached[key]
            scores.append(1.0 if verdict.is_correct else 0.0)

        value = sum(scores) / len(scores)
        return [
            MetricResult(
                name=self.id,
                label="LLM Judge",
                value=value,
                unit="ratio",
            )
        ]

    def get_verdict(self, query: str) -> JudgeVerdict | None:
        """Return the verdict for *query* from the last ``compute()`` call."""
        return self._last_verdicts.get(query)

    # ── Async judging ────────────────────────────────────────────

    async def _judge_all(
        self,
        pairs: list[tuple[QueryExecution, GroundTruthEntry]],
    ) -> dict[str, JudgeVerdict]:
        """Judge all *pairs* concurrently, respecting the semaphore."""
        sem = asyncio.Semaphore(self._concurrency)

        async def _judge_one(
            qe: QueryExecution,
            gt: GroundTruthEntry,
        ) -> tuple[str, JudgeVerdict]:
            async with sem:
                assert gt.expected_answer is not None
                assert qe.answer is not None
                template = self._prompt_template or get_judge_prompt(
                    gt.category
                )
                prompt = template.format(
                    question=gt.query,
                    expected_answer=gt.expected_answer,
                    predicted_answer=qe.answer,
                )
                verdict = await self._client.judge(gt.query, prompt)
                key = _cache_key(
                    gt.query,
                    gt.expected_answer,
                    qe.answer,
                    gt.category,
                )
                return key, verdict

        tasks = [_judge_one(qe, gt) for qe, gt in pairs]
        results = await asyncio.gather(*tasks)
        return dict(results)

    # ── Sync / Async bridge ──────────────────────────────────────

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Return a persistent event loop running in a background thread.

        The loop is created once and reused across calls so that the
        LLM client's httpx connection pool stays bound to a single,
        *open* event loop for the lifetime of this metric instance.
        """
        if hasattr(self, "_loop") and self._loop.is_running():
            return self._loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        self._loop = loop
        self._loop_thread = thread
        return loop

    def _run_async(self, coro: object) -> dict[str, JudgeVerdict]:
        """Run an async coroutine from synchronous code.

        Uses a persistent event loop in a dedicated background thread
        so this works safely even when called from within an
        already-running loop (which is the case in ``evaluate()``).
        """
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)  # type: ignore[arg-type]
        return future.result()
