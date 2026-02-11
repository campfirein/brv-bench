"""Transform LongMemEval dataset to brv-bench canonical format.

Usage:
    python scripts/transform_longmemeval.py <longmemeval.json> <output.json>

The input file can be any LongMemEval variant:
    - longmemeval_oracle.json   (evidence sessions only)
    - longmemeval_s_cleaned.json (~40 sessions per question)
    - longmemeval_m_cleaned.json (~500 sessions per question)

Download from: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brv_bench.datasets.longmemeval import transform


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/transform_longmemeval.py"
            " <longmemeval.json> <output.json>"
        )
        sys.exit(1)

    raw_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    if not raw_path.exists():
        print(f"Error: {raw_path} not found")
        sys.exit(1)

    dataset = transform(raw_path)

    output = {
        "name": dataset.name,
        "corpus": [
            {
                "doc_id": d.doc_id,
                "content": d.content,
                "source": d.source,
            }
            for d in dataset.corpus
        ],
        "entries": [
            {
                "query": e.query,
                "expected_doc_ids": list(e.expected_doc_ids),
                "category": e.category,
                "expected_answer": e.expected_answer,
            }
            for e in dataset.entries
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Corpus documents: {len(dataset.corpus)}")
    print(f"Ground truth entries: {len(dataset.entries)}")
    print(f"Output: {out_path}")

    # Stats by category
    categories: dict[str, int] = {}
    for e in dataset.entries:
        categories[e.category] = categories.get(e.category, 0) + 1
    print("\nEntries by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
