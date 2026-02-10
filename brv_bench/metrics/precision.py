"""Precision@K metric."""

from brv_bench.metrics.base import Metric
from brv_bench.types import GroundTruthEntry, MetricResult, QueryExecution


class PrecisionAtK(Metric):
    """Measures the fraction of top-K results that are relevant."""

    def __init__(self, k: int) -> None:
        self._k = k

    @property
    def id(self) -> str:
        return f"precision@{self._k}"

    def compute(
        self,
        pairs: list[tuple[QueryExecution, GroundTruthEntry]],
    ) -> list[MetricResult]:
        if not pairs:
            return [
                MetricResult(
                    name=self.id, label=f"Precision@{self._k}", value=0.0, unit="ratio"
                )
            ]

        values: list[float] = []
        for execution, truth in pairs:
            top_k = execution.results[: self._k]
            relevant = set(truth.expected_docs)
            hits = sum(1 for r in top_k if r.path in relevant)
            denominator = min(self._k, max(len(relevant), 1))
            values.append(hits / denominator)

        mean = sum(values) / len(values)
        return [
            MetricResult(
                name=self.id, label=f"Precision@{self._k}", value=mean, unit="ratio"
            )
        ]
