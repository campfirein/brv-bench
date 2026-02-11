"""Entry point: python -m brv_bench."""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from brv_bench.adapters.brv_cli import BrvCliAdapter
from brv_bench.commands.curate import curate
from brv_bench.commands.evaluate import evaluate
from brv_bench.datasets.locomo import PROMPT_CONFIG as LOCOMO_PROMPT_CONFIG
from brv_bench.metrics import default_metrics
from brv_bench.reporting.terminal import format_report, save_summary
from brv_bench.types import (
    BenchmarkDataset,
    CorpusDocument,
    GroundTruthEntry,
    PromptConfig,
)

# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="brv-bench",
        description=(
            "Benchmark suite for AI agent context retrieval systems."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- curate ---
    curate_parser = subparsers.add_parser(
        "curate",
        help="Populate context tree from a benchmark dataset.",
    )
    curate_parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to benchmark dataset JSON file.",
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
    eval_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save results JSON (incremental + final).",
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
    return BenchmarkDataset(name=data["name"], corpus=corpus, entries=entries)


DATASET_PROMPT_CONFIGS: dict[str, PromptConfig] = {
    "locomo": LOCOMO_PROMPT_CONFIG,
}


def _resolve_prompt_config(dataset_name: str) -> PromptConfig:
    """Look up the prompt config for a dataset by name."""
    config = DATASET_PROMPT_CONFIGS.get(dataset_name)
    if config is None:
        raise ValueError(
            f"No prompt config for dataset '{dataset_name}'. "
            f"Known datasets: {', '.join(DATASET_PROMPT_CONFIGS)}"
        )
    return config


async def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    if args.command == "curate":
        dataset = load_dataset(args.ground_truth)
        prompt_config = _resolve_prompt_config(dataset.name)
        summary = await curate(dataset.corpus, prompt_config)
        print(f"Curated {summary.succeeded}/{summary.total} documents.")
        if summary.failed > 0:
            for r in summary.results:
                if not r.success:
                    print(
                        f"  FAILED: {r.doc_id} — {r.message}",
                        file=sys.stderr,
                    )
            return 1
        return 0

    elif args.command == "evaluate":
        dataset = load_dataset(args.ground_truth)
        metrics = default_metrics()
        prompt_config = _resolve_prompt_config(dataset.name)

        adapter = BrvCliAdapter(prompt_config=prompt_config)

        output_path = args.output
        if output_path is None:
            report_dir = Path("report")
            report_dir.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d")
            stem = f"{stamp}_{dataset.name}_{adapter.name}"
            output_path = report_dir / f"{stem}.json"

        report = await evaluate(
            adapter,
            dataset,
            metrics,
            limit=args.limit,
            output_path=output_path,
        )

        print(format_report(report))

        save_summary(report, output_path.with_suffix(".txt"))

        print(f"\nResults saved to {output_path}")
        print(f"Summary saved to {output_path.with_suffix('.txt')}")

        return 0

    return 0


# =============================================================================


def cli() -> None:
    """CLI entry point (called by `python -m brv_bench`)."""
    sys.exit(asyncio.run(main()))


# =============================================================================

if __name__ == "__main__":
    cli()
