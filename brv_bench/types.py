"""Data types for brv-bench.

All dataclasses are frozen (immutable) to prevent accidental mutation
during metric computation and reporting.
"""

from dataclasses import dataclass

# =============================================================================
# Dataset types
# =============================================================================


@dataclass(frozen=True)
class CorpusDocument:
    """A document to curate into the context tree."""

    doc_id: str
    content: str
    source: str = ""


@dataclass(frozen=True)
class GroundTruthEntry:
    """A single benchmark query with expected results."""

    query: str
    expected_doc_ids: tuple[str, ...]
    category: str = "unspecified"
    expected_answer: str | None = None


@dataclass(frozen=True)
class BenchmarkDataset:
    """Complete benchmark dataset: corpus + queries + ground truth."""

    name: str
    corpus: tuple[CorpusDocument, ...]
    entries: tuple[GroundTruthEntry, ...]


# =============================================================================
# Retrieval types
# =============================================================================


@dataclass(frozen=True)
class SearchResult:
    """A single search result returned by the retrieval system."""

    path: str
    title: str
    score: float
    excerpt: str


@dataclass(frozen=True)
class QueryExecution:
    """Raw output from a single query execution."""

    query: str
    results: tuple[SearchResult, ...]
    total_found: int
    duration_ms: float
    answer: str | None = None


# =============================================================================
# Reporting types
# =============================================================================


@dataclass(frozen=True)
class Percentiles:
    """Latency percentiles."""

    p50: float
    p95: float
    p99: float


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
