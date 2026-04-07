"""Entry point: python -m brv_bench."""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Import dataset modules so they self-register their PromptConfigs.
import brv_bench.datasets.locomo
import brv_bench.datasets.longmemeval  # noqa: F401
from brv_bench.adapters.brv_cli import BrvCliAdapter
from brv_bench.commands.curate import curate
from brv_bench.commands.evaluate import evaluate
from brv_bench.datasets import get_prompt_config
from brv_bench.metrics import default_metrics
from brv_bench.reporting.terminal import format_report, save_summary
from brv_bench.types import (
    BenchmarkDataset,
    CorpusDocument,
    GroundTruthEntry,
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

    # LLM-as-Judge options
    eval_parser.add_argument(
        "--judge",
        action="store_true",
        default=False,
        help="Enable LLM-as-Judge answer correctness metric.",
    )
    eval_parser.add_argument(
        "--judge-backend",
        choices=["anthropic", "gemini", "ollama", "openai"],
        default="gemini",
        help="LLM backend for the judge (default: gemini).",
    )
    eval_parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model name for the judge (default: backend-specific).",
    )
    eval_parser.add_argument(
        "--judge-host",
        type=str,
        default=None,
        help=(
            "Ollama server host for the ollama judge backend "
            "(default: http://localhost:11434)."
        ),
    )
    eval_parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=5,
        help="Max parallel judge API calls (default: 5).",
    )
    eval_parser.add_argument(
        "--judge-cache",
        type=Path,
        default=None,
        help="Path to JSON file for caching judge verdicts.",
    )

    # Answer justifier options
    eval_parser.add_argument(
        "--justifier-backend",
        choices=["anthropic", "gemini", "ollama", "openai"],
        default="gemini",
        help="LLM backend for the answer justifier (default: gemini).",
    )
    eval_parser.add_argument(
        "--justifier-model",
        type=str,
        default=None,
        help="Model name for the justifier (default: backend-specific).",
    )
    eval_parser.add_argument(
        "--justifier-host",
        type=str,
        default=None,
        help=(
            "Ollama server host for the ollama justifier backend "
            "(default: http://localhost:11434)."
        ),
    )
    eval_parser.add_argument(
        "--justifier-concurrency",
        type=int,
        default=5,
        help="Max parallel justifier API calls (default: 5).",
    )

    # Isolated mode
    eval_parser.add_argument(
        "--context-tree-source",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to a pre-curated context-tree directory. "
            "When set, each query copies only its domain folder from this "
            "source into .brv/context-tree/, runs the query, then deletes "
            "it (isolated mode). The live context-tree stays blank between "
            "queries."
        ),
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
        for e in data.get("entries", [])
    )
    return BenchmarkDataset(name=data["name"], corpus=corpus, entries=entries)


async def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    if args.command == "curate":
        dataset = load_dataset(args.ground_truth)
        prompt_config = get_prompt_config(dataset.name)
        summary = await curate(dataset.corpus, prompt_config)
        print(f"\nCurated {summary.succeeded}/{summary.total} documents.")
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
        prompt_config = get_prompt_config(dataset.name)

        if args.judge:
            from brv_bench.metrics._judge.client import (
                create_judge_client,
            )
            from brv_bench.metrics.llm_judge import LLMJudge

            client = create_judge_client(
                backend=args.judge_backend,
                model=args.judge_model,
                host=args.judge_host,
            )
            judge_metric = LLMJudge(
                client=client,
                prompt_template=prompt_config.judge_template,
                concurrency=args.judge_concurrency,
                cache_path=args.judge_cache,
            )
            metrics.append(judge_metric)

        justifier = None
        if prompt_config.justifier_template:
            from brv_bench.adapters.justifier import AnswerJustifier
            from brv_bench.metrics._judge.client import create_judge_client

            justifier_client = create_judge_client(
                backend=args.justifier_backend,
                model=args.justifier_model,
                host=args.justifier_host,
            )
            justifier = AnswerJustifier(
                client=justifier_client,
                prompt_template=prompt_config.justifier_template,
            )

        adapter = BrvCliAdapter(
            prompt_config=prompt_config,
            justifier=justifier,
            context_tree_source=args.context_tree_source,
        )

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

        print("\n" + format_report(report))

        save_summary(report, output_path.with_suffix(".txt"))

        print(f"\nResults saved to {output_path}")
        print(f"Summary saved to {output_path.with_suffix('.txt')}")

        return 0


# =============================================================================


def cli() -> None:
    """CLI entry point (called by `python -m brv_bench`)."""
    sys.exit(asyncio.run(main()))


# =============================================================================

if __name__ == "__main__":
    cli()
