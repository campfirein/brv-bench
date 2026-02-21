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

## Supported Datasets

| Dataset | Description | Corpus | Queries | Download |
|---------|-------------|--------|---------|----------|
| LoCoMo | Long-term conversation memory QA (10 conversations, 272 sessions) | 272 docs | 1982 | [locomo10.json](https://github.com/snap-research/locomo/blob/main/data/locomo10.json) |
| LongMemEval | Long-term interactive memory benchmark (ICLR 2025, 6 memory abilities) | 948 docs (oracle) | 500 | [HuggingFace](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) |

## Preparing the Dataset

Transform a raw dataset JSON into brv-bench's canonical format.

**LoCoMo:**

```bash
python scripts/transform_locomo.py locomo10.json output/locomo_benchmark.json
```

**LongMemEval:**

```bash
python scripts/transform_longmemeval.py longmemeval_oracle.json output/longmemeval_benchmark.json
```

LongMemEval has three variants — use whichever fits your evaluation:
- `longmemeval_oracle.json` — evidence sessions only (recommended for initial testing)
- `longmemeval_s_cleaned.json` — ~40 sessions per question (~115k tokens)
- `longmemeval_m_cleaned.json` — ~500 sessions per question (~1.5M tokens)

Each transform produces a single JSON file with corpus documents (one per session) and ground-truth QA entries.

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
python -m brv_bench evaluate --ground-truth output/locomo_benchmark.json
```

Queries the context tree for each ground-truth entry (1982 for LoCoMo), computes metrics (Precision, Recall, MRR, Diversity, F1, Exact Match), and saves results.

- Results are always saved to `report/{yyyymmdd}_{dataset}_{memory_system}.json` (detailed per-query data) and `.txt` (summary).
- Use `--output path/to/results.json` to override the output location.
- Per-query results are saved **incrementally** (crash-safe).
- Use `--context-tree-source PATH` to enable isolated mode (see section 4 below).
- Run as many times as you want -- after tuning search, changing curate strategy, etc.

### 3. Evaluate with LLM-as-Judge

brv-bench supports an **LLM-as-Judge** metric that uses an LLM to evaluate whether predicted answers are semantically correct. This is the primary answer-quality metric, following the methodology of [LongMemEval](https://arxiv.org/abs/2410.10813) (ICLR 2025, 97% human agreement).

#### Setup

1. Install the judge dependencies:

```bash
pip install 'brv-bench[judge]'
```

2. Set your API key as an environment variable. 

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Then reload your shell: `source ~/.zshrc`

You can get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

For other backends, set the corresponding key instead:
- **Anthropic:** `export ANTHROPIC_API_KEY="your-key"`
- **OpenAI:** `export OPENAI_API_KEY="your-key"`

#### Run

```bash
python -m brv_bench evaluate \
  --ground-truth output/locomo_benchmark.json \
  --judge \
  --judge-cache report/judge_cache.json
```

The `--judge-cache` flag saves verdicts to a JSON file so re-runs don't repeat API calls for unchanged answers.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--judge` | off | Enable LLM-as-Judge metric |
| `--judge-backend` | `gemini` | LLM backend: `gemini`, `anthropic`, or `openai` |
| `--judge-model` | backend default | Model name override (e.g. `gemini-2.5-flash`, `gpt-4o`) |
| `--judge-concurrency` | `5` | Max parallel judge API calls |
| `--judge-cache` | none | Path to JSON cache file for verdicts |

### 4. Isolated Mode (per-query domain isolation)

By default `evaluate` queries whatever is already in `.brv/context-tree/`. When the tree contains all questions' sessions simultaneously, brv may return results from the wrong question's domain, which the source filter then discards — leaving the query with no hits.

**Isolated mode** fixes this by scoping the context tree to **one question at a time**:

1. Copy `{source}/{question_id}/` → `.brv/context-tree/{question_id}/`
2. Run `brv query` (only that question's sessions are visible)
3. Delete `.brv/context-tree/{question_id}/` before the next iteration

Use it with a **pre-curated** context tree directory via `--context-tree-source`:

```bash
python -m brv_bench evaluate \
  --ground-truth output/longmemeval_benchmark.json \
  --context-tree-source LongmemEval-bench/merged/.brv/context-tree \
  --judge --judge-cache report/judge_cache.json
```

The source directory must follow the standard layout:
```
{context-tree-source}/
├── {question_id_1}/
│   ├── session_1/
│   │   └── key_facts.md
│   └── session_2/
│       └── key_facts.md
├── {question_id_2}/
│   └── ...
```

The live `.brv/context-tree/` is cleared at the start of each run (`reset`) and stays blank between queries. This eliminates cross-question contamination without requiring any changes to curation.

**Pre-curated context trees for LongMemEval** are stored in `LongmemEval-bench/merged/.brv/context-tree/` (all 500 question domains merged from the split parts).

### 5. Answer Justifier

`brv query` performs direct search and returns raw context-tree content (key facts) — it does not synthesise answers. The **Answer Justifier** is an external LLM call that takes the retrieved context + question and produces a concise answer for F1/EM/LLM-Judge metrics.

The justifier is **automatically enabled** for datasets that define a `justifier_template` (both LoCoMo and LongMemEval do). It uses the same API key as the judge.

#### Justifier Options

| Flag | Default | Description |
|------|---------|-------------|
| `--justifier-backend` | `gemini` | LLM backend: `gemini`, `anthropic`, or `openai` |
| `--justifier-model` | backend default | Model name override |
| `--justifier-concurrency` | `5` | Reserved for future parallel queries |

### Example output

```
================================================================
  Dataset:       locomo
  Memory System: brv-cli
  Context tree:  272 documents
  Queries:       1982
----------------------------------------------------------------

  Quality Metrics (Overall):
  ----------------------------------------
  Precision@5        68.0%
  Precision@10       71.5%
  Recall@5           68.0%
  Recall@10          85.2%
  NDCG@5              0.72
  NDCG@10             0.78
  MRR                 0.76
  Diversity@5         0.89
  F1 Score           45.0%
  Exact Match        21.0%
  LLM Judge          74.5%

  Latency Metrics:
  --------------------------------------------------------
  Metric                   Mean      p50      p95      p99
  --------------------------------------------------------
  Cold Latency             1.2s     1.1s     2.3s     3.1s
================================================================

Results saved to report/20260211_locomo_brv-cli.json
Summary saved to report/20260211_locomo_brv-cli.txt
```

> **Note:** LLM Judge only appears when `--judge` is enabled.

## Ground Truth Format

```json
{
  "name": "locomo",
  "corpus": [
    {
      "doc_id": "session_1",
      "content": "Session transcript...",
      "source": "conv-26"
    }
  ],
  "entries": [
    {
      "query": "What career path has Caroline decided to pursue?",
      "expected_doc_ids": ["session_1", "session_4"],
      "category": "multi-hop",
      "expected_answer": "counseling or mental health for transgender people"
    }
  ]
}
```

- `corpus`: documents to curate into the context tree (used by the `curate` command)
- `doc_id`: matches the topic folder name in the context tree (e.g. `session_1` → `.brv/context-tree/{domain}/session_1/`)
- `source`: the domain folder name (conversation ID for LoCoMo, question ID for LongMemEval)
- `expected_doc_ids`: doc IDs that contain evidence for the answer
- `expected_answer`: ground-truth answer for F1/Exact Match/LLM Judge scoring
- `category`: optional, for per-category analysis (LoCoMo uses `single-hop`, `multi-hop`, `temporal`, `commonsense`, `adversarial`)

## Metrics

| Metric | What It Measures |
|--------|-----------------|
| LLM Judge | LLM-as-Judge binary correctness (requires `--judge` flag) |
| Precision@K | Fraction of top-K results that are relevant |
| Recall@K | Fraction of relevant documents found in top-K |
| NDCG@K | Normalized Discounted Cumulative Gain — ranking quality of top-K |
| MRR | Reciprocal rank of the first relevant result |
| Result Diversity | 1 - mean pairwise similarity in top-K (higher = more diverse) |
| F1 Score | Token-overlap F1 between predicted and expected answers (with Porter stemming) |
| Exact Match | Normalized string equality of predicted vs expected answers |
| Cold Latency | Query time with no cache (p50/p95/p99) |

## Requirements

- Python >= 3.12
- `brv` CLI installed and authenticated (`brv login`)
- A project with `brv` initialized (`.brv/` directory exists)
