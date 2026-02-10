"""Token-overlap F1 score metric."""

from collections import Counter

from brv_bench.metrics._normalize import get_tokens
from brv_bench.metrics.base import Metric
from brv_bench.types import GroundTruthEntry, MetricResult, QueryExecution


class F1Score(Metric):
    """Token-overlap F1 between predicted and expected answers."""

    @property
    def id(self) -> str:
        return "f1"

    def compute(
        self,
        pairs: list[tuple[QueryExecution, GroundTruthEntry]],
    ) -> list[MetricResult]:
        scores: list[float] = []
        for qe, gt in pairs:
            if qe.answer is None or gt.expected_answer is None:
                continue
            pred_tokens = Counter(get_tokens(qe.answer, stem=True))
            gold_tokens = Counter(get_tokens(gt.expected_answer, stem=True))
            common = sum((pred_tokens & gold_tokens).values())
            if common == 0:
                scores.append(0.0)
                continue
            precision = common / sum(pred_tokens.values())
            recall = common / sum(gold_tokens.values())
            scores.append(2 * precision * recall / (precision + recall))

        value = sum(scores) / len(scores) if scores else 0.0
        return [MetricResult(name="F1", label="F1 Score", value=value, unit="ratio")]
