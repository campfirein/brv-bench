"""Metrics registry."""

# =============================================================================

from brv_bench.metrics.base import Metric
from brv_bench.metrics.diversity import ResultDiversity
from brv_bench.metrics.exact_match import ExactMatch
from brv_bench.metrics.f1 import F1Score
from brv_bench.metrics.latency import LatencyMetric
from brv_bench.metrics.llm_judge import LLMJudge
from brv_bench.metrics.mrr import MeanReciprocalRank
from brv_bench.metrics.ndcg import NDCGAtK
from brv_bench.metrics.precision import PrecisionAtK
from brv_bench.metrics.recall import RecallAtK

# =============================================================================

__all__ = [
    "ExactMatch",
    "F1Score",
    "LLMJudge",
    "LatencyMetric",
    "MeanReciprocalRank",
    "Metric",
    "NDCGAtK",
    "PrecisionAtK",
    "RecallAtK",
    "ResultDiversity",
    "default_metrics",
]


# =============================================================================


def default_metrics() -> list[Metric]:
    """Return the default P0 metric set."""
    return [
        PrecisionAtK(5),
        PrecisionAtK(10),
        RecallAtK(5),
        RecallAtK(10),
        NDCGAtK(5),
        NDCGAtK(10),
        MeanReciprocalRank(),
        ResultDiversity(5),
        LatencyMetric("Cold Latency", "cold-latency"),
        F1Score(),
        ExactMatch(),
    ]
