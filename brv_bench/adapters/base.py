"""Base adapter interface.

Adapters provide the bridge between the benchmark harness and the
retrieval system under test.
Each adapter handles setup, querying, cache management, and cleanup
for a specific integration method.
"""

# =============================================================================

from abc import ABC, abstractmethod

from brv_bench.types import QueryExecution

# =============================================================================


class RetrievalAdapter(ABC):
    """Abstract base class for retrieval system adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name."""

    @property
    @abstractmethod
    def supports_warm_latency(self) -> bool:
        """Whether this adapter can measure warm (cached) latency."""

    @abstractmethod
    async def setup(self) -> None:
        """Prepare the adapter for benchmarking.

        Called once before any queries. Implementations may verify
        that the retrieval system is accessible, the context tree exists, etc.
        """

    @abstractmethod
    async def query(self, query: str, limit: int) -> QueryExecution:
        """Execute a single query against the retrieval system.

        Args:
            query: Natural language query string.
            limit: Maximum number of results to return.

        Returns:
            QueryExecution with results and timing.
        """

    @abstractmethod
    async def reset(self) -> None:
        """Reset retrieval state between measurement phases.

        For cold latency measurements, this should invalidate any caches.
        Implementations that don't support cache control may no-op.
        """

    @abstractmethod
    async def teardown(self) -> None:
        """Clean up all resources.

        Called once after all queries are complete.
        """
