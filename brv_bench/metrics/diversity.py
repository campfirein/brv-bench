"""Result Diversity metric."""

import re

from brv_bench.metrics.base import Metric
from brv_bench.types import GroundTruthEntry, MetricResult, QueryExecution


class ResultDiversity(Metric):
    """Measures diversity of top-K results using pairwise Jaccard distance.

    Score = 1 - mean(pairwise_jaccard_similarities).
    Higher score = more diverse results (less redundancy).
    """

    def __init__(self, k: int) -> None:
        self._k = k

    @property
    def id(self) -> str:
        return f"diversity@{self._k}"

    def compute(
        self,
        pairs: list[tuple[QueryExecution, GroundTruthEntry]],
    ) -> list[MetricResult]:
        if not pairs:
            return [
                MetricResult(
                    name=self.id, label=f"Diversity@{self._k}", value=0.0, unit="ratio"
                )
            ]

        values: list[float] = []
        for execution, _truth in pairs:
            top_k = execution.results[: self._k]
            if len(top_k) < 2:
                values.append(1.0)
                continue

            word_sets = [_tokenize(r.excerpt) for r in top_k]
            similarities: list[float] = []
            for i in range(len(word_sets)):
                for j in range(i + 1, len(word_sets)):
                    similarities.append(_jaccard(word_sets[i], word_sets[j]))

            mean_sim = sum(similarities) / len(similarities)
            values.append(1.0 - mean_sim)

        mean = sum(values) / len(values)
        return [
            MetricResult(
                name=self.id, label=f"Diversity@{self._k}", value=mean, unit="ratio"
            )
        ]


_WORD_PATTERN = re.compile(r"\w{3,}")


def _tokenize(text: str) -> set[str]:
    """Extract lowercase words with 3+ characters."""
    return set(_WORD_PATTERN.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0
