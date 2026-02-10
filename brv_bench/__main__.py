"""Entry point: python -m brv_bench."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from brv_bench.commands.curate import curate
from brv_bench.metrics import default_metrics
from brv_bench.types import BenchmarkDataset, CorpusDocument, GroundTruthEntry

# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="brv-bench",
        description=(
            "Benchmark suite for AI agent context"
            " retrieval systems."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True
    )

    # --- curate ---
    curate_parser = subparsers.add_parser(
        "curate",
        help="Populate context tree from source files.",
    )
    curate_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory containing source files to curate.",
    )

    # --- evaluate ---
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Measure retrieval quality against ground truth.",
    )
    eval_parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to ground truth JSON file.",
    )
    eval_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max results per query (default: 10).",
    )

    return parser.parse_args(argv)


# =============================================================================


def load_dataset(path: Path) -> BenchmarkDataset:
    """Load benchmark dataset from JSON file."""
    if not path.exists():
        print(
            f"Error: dataset file not found: {path}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    corpus = tuple(
        CorpusDocument(
            doc_id=d["doc_id"],
            content=d["content"],
            source=d.get("source", ""),
        )
        for d in data.get("corpus", [])
    )

    entries = tuple(
        GroundTruthEntry(
            query=e["query"],
            expected_doc_ids=tuple(e["expected_doc_ids"]),
            category=e.get("category", "unspecified"),
            expected_answer=e.get("expected_answer"),
        )
        for e in data["entries"]
    )
    return BenchmarkDataset(
        name=data["name"], corpus=corpus, entries=entries
    )


async def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    if args.command == "curate":
        summary = await curate(args.source)
        print(
            f"Curated {summary.succeeded}/{summary.total}"
            " files."
        )
        if summary.failed > 0:
            for r in summary.results:
                if not r.success:
                    print(
                        f"  FAILED: {r.file} — {r.message}",
                        file=sys.stderr,
                    )
            return 1
        return 0

    elif args.command == "evaluate":
        _dataset = load_dataset(args.ground_truth)
        _metrics = default_metrics()

        # Adapter will be resolved here once BrvCliAdapter
        # is implemented (B3).
        print(
            "Error: No adapter configured. "
            "BrvCliAdapter implementation is pending (B3).",
            file=sys.stderr,
        )
        return 1

    return 0


# =============================================================================


def cli() -> None:
    """CLI entry point (called by `python -m brv_bench`)."""
    sys.exit(asyncio.run(main()))


# =============================================================================

if __name__ == "__main__":
    cli()
