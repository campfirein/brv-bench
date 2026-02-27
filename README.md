<p align="center"><img src="assets/images/logo.png" alt="ByteRover Logo" width="500" /></p>

Benchmark suite for evaluating retrieval quality, latency, and diversity of AI agent context systems. Powered for ByteRover, engineered by [ByteRover](https://www.byterover.dev/).

## Overall Accuracy
![image](assets/images/overall_accuracy.svg)
## Setup

```bash
source scripts/source_env.sh
python -m brv_bench --help
```

## Supported Datasets

| Dataset | Description | Corpus | Queries | Download | Context Tree |
|---------|-------------|--------|---------|----------|:------------:|
| LoCoMo | Long-term conversation memory QA (10 conversations, 272 sessions) | 272 docs | 1982 | [locomo10.json](https://github.com/snap-research/locomo/blob/main/data/locomo10.json) | [download](https://drive.google.com/file/d/1U6pTh7aQqfJaMCjMYgVtQUzOcQaiEL0I/view) |
| LongMemEval | Long-term interactive memory benchmark (ICLR 2025, 6 memory abilities) | 948 docs (oracle) | 500 | [HuggingFace](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | |

## Usage

### 1. Transform dataset

Pre-transformed datasets are provided in `assets/` (`locomo_sample.json`, `longmemeval_s.json`) — you can skip this step and pass those files directly to `curate`/`evaluate`.

To transform from raw sources:

```bash
# LoCoMo → produces assets/sample_data/locomo.json (already provided)
python scripts/transform_locomo.py locomo10.json assets/sample_data/locomo.json

# LongMemEval (three variants: oracle / s_cleaned ~40 sessions / m_cleaned ~500 sessions)
# → produces assets/longmemeval_s.json (already provided)
python scripts/transform_longmemeval.py longmemeval_oracle.json assets/longmemeval_s.json
```

### 2. Curate (populate context tree)

```bash
python -m brv_bench curate --ground-truth assets/sample_data/locomo.json
```

### 3. Evaluate

```bash
export GEMINI_API_KEY="your-api-key"

python -m brv_bench evaluate \
  --ground-truth assets/sample_data/locomo.json \
  --judge \
  --judge-cache report/judge_cache_locomo_gemini.json
```

The justifier is automatically enabled for LoCoMo and LongMemEval (no extra flag needed). See [LLM-as-Judge](#llm-as-judge) and [Justifier](#justifier) below for detailed configuration options.

Results are saved to `report/{yyyymmdd}_{dataset}_{memory_system}.json/.txt`. Per-query results are written incrementally (crash-safe).

#### LLM-as-Judge

Install deps and set an API key, then pass `--judge`:

```bash
pip install 'brv-bench[judge]'
export GEMINI_API_KEY="your-api-key"   # or ANTHROPIC_API_KEY / OPENAI_API_KEY

python -m brv_bench evaluate \
  --ground-truth assets/sample_data/locomo.json \
  --judge --judge-cache report/judge_cache_locomo_gemini.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--judge` | off | Enable LLM-as-Judge metric |
| `--judge-backend` | `gemini` | `gemini`, `anthropic`, or `openai` |
| `--judge-model` | `gemini-2.5-flash` / `claude-sonnet-4-6` / `gpt-4o-2024-08-06` | Model name override (default varies by backend) |
| `--judge-concurrency` | `5` | Max parallel judge API calls |
| `--judge-cache` | none | Path to JSON cache file |

#### Isolated Mode

Scopes the context tree to one question at a time to prevent cross-question contamination. Requires a pre-curated source directory.

```bash
python -m brv_bench evaluate \
  --ground-truth assets/longmemeval_s.json \
  --context-tree-source path/to/full-context-tree \
  --judge --judge-cache report/judge_cache_longmemeval_gemini.json
```

Source layout: `{context-tree-source}/{question_id}/{session_id}/key_facts.md`

#### Justifier

Automatically enabled for datasets with a `justifier_template` (LoCoMo and LongMemEval). Uses the same API key as the judge.

| Flag | Default | Description |
|------|---------|-------------|
| `--justifier-backend` | `gemini` | `gemini`, `anthropic`, or `openai` |
| `--justifier-model` | `gemini-2.5-flash` / `claude-sonnet-4-6` / `gpt-4o-2024-08-06` | Model name override (default varies by backend) |

#### Ground Truth Format

```json
{
  "name": "locomo",
  "corpus": [{ "doc_id": "session_1", "content": "...", "source": "conv-26" }],
  "entries": [{
    "query": "What career path has Caroline decided to pursue?",
    "expected_doc_ids": ["session_1", "session_4"],
    "expected_answer": "counseling or mental health for transgender people",
    "category": "multi-hop"
  }]
}
```

## Metrics

| Metric | What It Measures |
|--------|-----------------|
| LLM Judge | LLM-as-Judge binary correctness (requires `--judge`) |
| Precision@K | Fraction of top-K results that are relevant |
| Recall@K | Fraction of relevant documents found in top-K |
| NDCG@K | Ranking quality of top-K |
| MRR | Reciprocal rank of the first relevant result |
| Cold Latency | Query time with no cache (p50/p95/p99) |


## Results on LoCoMo (LLM Judge Accuracy %)

![image](assets/images/accuracy_per_category.svg)

| System | Single-Hop | Multi-Hop | Open Domain | Temporal | Overall |
|--------|:----------:|:---------:|:-----------:|:--------:|:-------:|
| **ByteRover 2.0 (Run 2 - best)** | **95.4%** | **85.1%** | 77.2% | **94.4%** | **92.2%** |
| ByteRover 2.0 (Run 1) | 93.9% | 82.6% | 77.2% | 94.4% | 90.9% |
| Hindsight (Gemini-3) | 86.2% | 70.8% | **95.1%** | 83.8% | 89.6% |
| Memobase v0.0.37 | 70.9% | 46.9% | 77.2% | 85.1% | 75.8% |
| Zep | 74.1% | 66.0% | 67.7% | 79.8% | 75.1% |
| Mem0-Graph | 65.7% | 47.2% | 75.7% | 58.1% | 68.4% |
| Mem0 | 67.1% | 51.2% | 72.9% | 55.5% | 66.9% |
| OpenAI Memory | 63.8% | 42.9% | 62.3% | 21.7% | 52.9% |

## Reproduction

To reproduce the ByteRover results above:

```bash
# For LoCoMo
python -m brv_bench evaluate \
  --ground-truth output/locomo.json \
  --judge \
  --judge-model "gemini-3-flash-preview" \
  --justifier-model "gemini-3-pro-preview"
```

## Requirements
- byterover-cli >= 2.0.0
- Python >= 3.12
- A project with `brv` initialized (`.brv/` directory exists)
