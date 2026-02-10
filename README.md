# brv-bench

Benchmark suite for evaluating retrieval quality, latency, and diversity of AI agent context systems.

## Setup

To source virtual environment, install dependencies, run:
```bash
source scripts/source_env.sh
```

To verify the setup, run:
```bash
python src/main.py
```

## Usage

brv-bench has two commands: `curate` and `evaluate`. Both run from your project directory where `brv` is initialized.

### 1. Curate (populate the context tree)

```bash
cd ~/workspace/your-project
python -m brv_bench curate --source datasets/your-project/
```

This calls `brv curate` for each source file, populating `.brv/context-tree/`.
Run once, or whenever your source material changes.

### 2. Evaluate (measure retrieval quality)

```bash
cd ~/workspace/your-project
python -m brv_bench evaluate --ground-truth datasets/your-project/ground_truth.json
```

This calls `brv query --headless --format json` for each query in the ground truth,
compares results against expected documents, and reports metrics.
Run as many times as you want -- after tuning search, changing curate strategy, etc.

### Example output

```
================================================================
  brv-bench: Context Tree Retrieval Benchmark
================================================================
  Dataset:      byterover-cli
  Context tree: 47 documents
  Queries:      30
  Runs:         3
----------------------------------------------------------------

  Quality Metrics:
  ----------------------------------------
  Precision@5          82.3%
  Precision@10         71.5%
  Recall@5             68.0%
  Recall@10            85.2%
  MRR                  0.76
  Diversity@5          0.89

  Latency Metrics:
  --------------------------------------------------------
  Metric               Mean     p50      p95      p99
  --------------------------------------------------------
  Cold Latency         1.2s     1.1s     2.3s     3.1s
  Warm Latency         0.8s     0.7s     1.5s     2.0s
================================================================
```

## Ground Truth Format

```json
{
  "name": "byterover-cli",
  "entries": [
    {
      "query": "How is OAuth authentication implemented?",
      "expected_docs": ["authentication/oauth_2_0_and_pkce_authentication.md"],
      "category": "natural-language"
    },
    {
      "query": "OAuth 2.0 and PKCE Authentication",
      "expected_docs": ["authentication/oauth_2_0_and_pkce_authentication.md"],
      "category": "exact"
    }
  ]
}
```

- `expected_docs`: paths relative to `.brv/context-tree/`
- `category`: optional, for per-category analysis (`exact`, `fuzzy`, `synonym`, `natural-language`)

## Metrics

| Metric | What It Measures |
|--------|-----------------|
| Precision@K | Fraction of top-K results that are relevant |
| Recall@K | Fraction of relevant documents found in top-K |
| MRR | Reciprocal rank of the first relevant result |
| Result Diversity | 1 - mean pairwise similarity in top-K (higher = more diverse) |
| Cold Latency | Query time with no cache (p50/p95/p99) |
| Warm Latency | Query time with warm cache (p50/p95/p99) |

## Requirements

- Python >= 3.12
- `brv` CLI installed and authenticated (`brv login`)
- A project with `brv` initialized (`.brv/` directory exists)
