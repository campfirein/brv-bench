"""Base metric interface.

All metrics implement the Metric ABC. Each is stateless and self-contained:
receives query executions paired with ground truth, returns MetricResult values.
"""

from abc import ABC, abstractmethod

from brv_bench.types import GroundTruthEntry, MetricResult, QueryExecution


class Metric(ABC):
    """Abstract base class for benchmark metrics."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this metric (e.g., 'precision@5')."""

    @abstractmethod
    def compute(
        self,
        pairs: list[tuple[QueryExecution, GroundTruthEntry]],
    ) -> list[MetricResult]:
        """Compute this metric from query executions paired with ground truth.

        Args:
            pairs: List of (execution_result, ground_truth) tuples.

        Returns:
            One or more MetricResult values.
        """
