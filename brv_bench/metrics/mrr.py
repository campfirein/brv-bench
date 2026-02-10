"""Mean Reciprocal Rank (MRR) metric."""

from brv_bench.metrics.base import Metric
from brv_bench.types import GroundTruthEntry, MetricResult, QueryExecution


class MeanReciprocalRank(Metric):
    """Measures the reciprocal rank of the first relevant result, averaged across queries."""

    @property
    def id(self) -> str:
        return "mrr"

    def compute(
        self,
        pairs: list[tuple[QueryExecution, GroundTruthEntry]],
    ) -> list[MetricResult]:
        if not pairs:
            return [MetricResult(name=self.id, label="MRR", value=0.0, unit="ratio")]

        values: list[float] = []
        for execution, truth in pairs:
            relevant = set(truth.expected_docs)
            rr = 0.0
            for i, result in enumerate(execution.results):
                if result.path in relevant:
                    rr = 1.0 / (i + 1)
                    break
            values.append(rr)

        mean = sum(values) / len(values)
        return [MetricResult(name=self.id, label="MRR", value=mean, unit="ratio")]
