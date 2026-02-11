"""Retrieval adapters."""

from brv_bench.adapters.base import RetrievalAdapter
from brv_bench.adapters.brv_cli import BrvCliAdapter

__all__ = ["BrvCliAdapter", "RetrievalAdapter"]
