"""Recall@K metric."""

from brv_bench.metrics.base import Metric
from brv_bench.types import GroundTruthEntry, MetricResult, QueryExecution


class RecallAtK(Metric):
    """Measures the fraction of relevant documents found in top-K results."""

    def __init__(self, k: int) -> None:
        self._k = k

    @property
    def id(self) -> str:
        return f"recall@{self._k}"

    def compute(
        self,
        pairs: list[tuple[QueryExecution, GroundTruthEntry]],
    ) -> list[MetricResult]:
        if not pairs:
            return [
                MetricResult(
                    name=self.id, label=f"Recall@{self._k}", value=0.0, unit="ratio"
                )
            ]

        values: list[float] = []
        for execution, truth in pairs:
            relevant = set(truth.expected_docs)
            if not relevant:
                values.append(1.0)
                continue
            top_k = execution.results[: self._k]
            hits = sum(1 for r in top_k if r.path in relevant)
            values.append(hits / len(relevant))

        mean = sum(values) / len(values)
        return [
            MetricResult(
                name=self.id, label=f"Recall@{self._k}", value=mean, unit="ratio"
            )
        ]
