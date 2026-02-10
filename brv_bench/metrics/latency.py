"""Latency metric with percentile computation."""

import math

from brv_bench.metrics.base import Metric
from brv_bench.types import (
    GroundTruthEntry,
    MetricResult,
    Percentiles,
    QueryExecution,
)


class LatencyMetric(Metric):
    """Measures query latency with p50/p95/p99 percentiles.

    Reads duration_ms from QueryExecution objects. The benchmark runner
    is responsible for providing cold or warm executions as appropriate.
    """

    def __init__(self, label: str, metric_id: str) -> None:
        self._label = label
        self._id = metric_id

    @property
    def id(self) -> str:
        return self._id

    def compute(
        self,
        pairs: list[tuple[QueryExecution, GroundTruthEntry]],
    ) -> list[MetricResult]:
        if not pairs:
            return [
                MetricResult(
                    name=self._id,
                    label=self._label,
                    value=0.0,
                    unit="ms",
                    percentiles=Percentiles(p50=0.0, p95=0.0, p99=0.0),
                )
            ]

        durations = sorted(e.duration_ms for e, _ in pairs)
        mean = sum(durations) / len(durations)

        return [
            MetricResult(
                name=self._id,
                label=self._label,
                value=mean,
                unit="ms",
                percentiles=Percentiles(
                    p50=_percentile(durations, 0.50),
                    p95=_percentile(durations, 0.95),
                    p99=_percentile(durations, 0.99),
                ),
            )
        ]


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute percentile from a pre-sorted list."""
    index = math.ceil(p * len(sorted_values)) - 1
    return sorted_values[max(0, index)]
