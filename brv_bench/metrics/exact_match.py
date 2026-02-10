"""Exact-match metric."""

from brv_bench.metrics._normalize import normalize_answer
from brv_bench.metrics.base import Metric
from brv_bench.types import GroundTruthEntry, MetricResult, QueryExecution


class ExactMatch(Metric):
    """Normalized exact-match between predicted and expected answers."""

    @property
    def id(self) -> str:
        return "exact-match"

    def compute(
        self,
        pairs: list[tuple[QueryExecution, GroundTruthEntry]],
    ) -> list[MetricResult]:
        scores: list[float] = []
        for qe, gt in pairs:
            if qe.answer is None or gt.expected_answer is None:
                continue
            match = normalize_answer(qe.answer) == normalize_answer(gt.expected_answer)
            scores.append(1.0 if match else 0.0)

        value = sum(scores) / len(scores) if scores else 0.0
        return [
            MetricResult(
                name="Exact Match",
                label="Exact Match",
                value=value,
                unit="ratio",
            )
        ]
