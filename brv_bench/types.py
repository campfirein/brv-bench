"""Data types for brv-bench.

All dataclasses are frozen (immutable) to prevent accidental mutation
during metric computation and reporting.
"""

from dataclasses import dataclass

# =============================================================================


@dataclass(frozen=True)
class GroundTruthEntry:
    """A single ground truth entry: a query with expected relevant documents."""

    query: str
    expected_docs: tuple[str, ...]
    category: str = "unspecified"


@dataclass(frozen=True)
class GroundTruthDataset:
    """A complete ground truth dataset."""

    name: str
    entries: tuple[GroundTruthEntry, ...]


# =============================================================================


@dataclass(frozen=True)
class SearchResult:
    """A single search result returned by the retrieval system."""

    path: str
    title: str
    score: float
    excerpt: str


# =============================================================================


@dataclass(frozen=True)
class QueryExecution:
    """Raw output from a single query execution."""

    query: str
    results: tuple[SearchResult, ...]
    total_found: int
    duration_ms: float


# =============================================================================


@dataclass(frozen=True)
class Percentiles:
    """Latency percentiles."""

    p50: float
    p95: float
    p99: float


# =============================================================================


@dataclass(frozen=True)
class MetricResult:
    """Computed value for a single metric."""

    name: str
    label: str
    value: float
    unit: str
    percentiles: Percentiles | None = None


@dataclass(frozen=True)
class BenchmarkReport:
    """Full benchmark report."""

    name: str
    context_tree_docs: int
    query_count: int
    duration_ms: float
    metrics: tuple[MetricResult, ...]
