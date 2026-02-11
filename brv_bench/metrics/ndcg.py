"""NDCG@K (Normalized Discounted Cumulative Gain) metric."""

import math

from brv_bench.metrics.base import Metric
from brv_bench.types import GroundTruthEntry, MetricResult, QueryExecution


class NDCGAtK(Metric):
    """Measures ranking quality using Normalized Discounted Cumulative Gain.

    Uses binary relevance: a result is relevant (1) if its path
    appears in expected_doc_ids, otherwise irrelevant (0).

    NDCG@K = DCG@K / IDCG@K where:
      DCG@K  = sum_{i=1}^{K} rel_i / log2(i + 1)
      IDCG@K = DCG of the ideal ranking (all relevant docs first)
    """

    def __init__(self, k: int) -> None:
        self._k = k

    @property
    def id(self) -> str:
        return f"ndcg@{self._k}"

    def compute(
        self,
        pairs: list[tuple[QueryExecution, GroundTruthEntry]],
    ) -> list[MetricResult]:
        if not pairs:
            return [
                MetricResult(
                    name=self.id,
                    label=f"NDCG@{self._k}",
                    value=0.0,
                    unit="ratio",
                )
            ]

        values: list[float] = []
        for execution, truth in pairs:
            top_k = execution.results[: self._k]
            relevant = set(truth.expected_doc_ids)

            dcg = _dcg(top_k, relevant)
            ideal_hits = min(len(relevant), self._k)
            idcg = _idcg(ideal_hits)

            values.append(dcg / idcg if idcg > 0 else 0.0)

        mean = sum(values) / len(values)
        return [
            MetricResult(
                name=self.id,
                label=f"NDCG@{self._k}",
                value=mean,
                unit="ratio",
            )
        ]


def _dcg(results: tuple | list, relevant: set[str]) -> float:
    """Compute DCG for a ranked list of results (binary relevance)."""
    return sum(
        1.0 / math.log2(i + 2)  # i+2 because i is 0-based
        for i, r in enumerate(results)
        if r.path in relevant
    )


def _idcg(num_relevant: int) -> float:
    """Compute ideal DCG: all relevant docs ranked at the top."""
    return sum(1.0 / math.log2(i + 2) for i in range(num_relevant))
