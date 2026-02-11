# brv-bench

Benchmark suite for evaluating retrieval quality, latency, and diversity of AI agent context systems.

## Setup

To source virtual environment, install dependencies, run:
```bash
source scripts/source_env.sh
```

To verify the setup, run:
```bash
python -m brv_bench --help
```

## Usage

brv-bench has two commands: `curate` and `evaluate`. Both run from your project directory where `brv` is initialized.

### 1. Curate (populate the context tree)

```bash
python -m brv_bench curate --ground-truth output/locomo_benchmark.json
```

Loads the benchmark dataset, formats each corpus document with the dataset-specific prompt template, and calls `brv curate` sequentially (272 docs for LoCoMo). Run once, or whenever your curate strategy changes.

**Pre-curated context tree for LoCoMo:** To skip curation, download and extract into your `.brv/` directory:
[locomo_context_tree.zip](https://drive.google.com/file/d/1JaZWr96bSfHAlQahXcT55rHH3to38dZu/view?usp=drive_link)

### 2. Evaluate (measure retrieval quality)

```bash
python -m brv_bench evaluate --ground-truth output/locomo_benchmark.json --output output/results.json
```

Queries the context tree for each ground-truth entry (1982 for LoCoMo), computes metrics (Precision, Recall, MRR, Diversity, F1, Exact Match), and saves results.

- `--output` saves per-query results **incrementally** (crash-safe) and the final report with metrics.
- Without `--output`, results are printed to stdout only.
- Run as many times as you want -- after tuning search, changing curate strategy, etc.

### Example output

```
Benchmark: locomo
Corpus docs: 272
Queries: 1982
Duration: 125000.0ms

  Precision@5: 0.6800 ratio
  Precision@10: 0.7150 ratio
  Recall@5: 0.6800 ratio
  Recall@10: 0.8520 ratio
  MRR: 0.7600 ratio
  Diversity@5: 0.8900 ratio
  Cold Latency: 1200.0000 ms
  F1 Score: 0.4500 ratio
  Exact Match: 0.2100 ratio

Results saved to output/results.json
```

## Ground Truth Format

```json
{
  "name": "locomo",
  "corpus": [
    {
      "doc_id": "conv-26_s1",
      "content": "Session transcript...",
      "source": "session_1"
    }
  ],
  "entries": [
    {
      "query": "What career path has Caroline decided to pursue?",
      "expected_doc_ids": ["conv-26_s1", "conv-26_s4"],
      "category": "multi-hop",
      "expected_answer": "counseling or mental health for transgender people"
    }
  ]
}
```

- `corpus`: documents to curate into the context tree (used by the `curate` command)
- `expected_doc_ids`: doc IDs that contain evidence for the answer
- `expected_answer`: ground-truth answer for F1/Exact Match scoring
- `category`: optional, for per-category analysis (LoCoMo uses `single-hop`, `multi-hop`, `temporal`, `commonsense`, `adversarial`)

## Metrics

| Metric | What It Measures |
|--------|-----------------|
| Precision@K | Fraction of top-K results that are relevant |
| Recall@K | Fraction of relevant documents found in top-K |
| MRR | Reciprocal rank of the first relevant result |
| Result Diversity | 1 - mean pairwise similarity in top-K (higher = more diverse) |
| F1 Score | Token-overlap F1 between predicted and expected answers (with Porter stemming) |
| Exact Match | Normalized string equality of predicted vs expected answers |
| Cold Latency | Query time with no cache (p50/p95/p99) |
| Warm Latency | Query time with warm cache (p50/p95/p99) |

## Requirements

- Python >= 3.12
- `brv` CLI installed and authenticated (`brv login`)
- A project with `brv` initialized (`.brv/` directory exists)
