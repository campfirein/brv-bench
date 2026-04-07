#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
BENCH_REPO="${BENCH_REPO:-${SCRIPT_DIR}/..}"
GROUND_TRUTH="${GROUND_TRUTH:-}"
CONTEXT_TREE_SOURCE="${CONTEXT_TREE_SOURCE:-}"
REPORT_DIR="${REPORT_DIR:-${BENCH_REPO}/report}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-1}"
JUSTIFIER_CONCURRENCY="${JUSTIFIER_CONCURRENCY:-1}"
LIMIT="${LIMIT:-32}"
WARMUP_PROMPT="Summarize the key events of World War II in 3 sentences."

# ─── Usage ───────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 <ollama-model-name>

Examples:
    GROUND_TRUTH=output/longmemeval_s_benchmark.json \\
    CONTEXT_TREE_SOURCE=LME-S-context-tree \\
        $0 qwen3.5:9b

Environment variables (required):
    GROUND_TRUTH            Path to ground truth JSON

Environment variables (optional):
    OLLAMA_HOST             Ollama API host (default: http://localhost:11434)
    BENCH_REPO              Path to brv-bench repo (default: auto-detected)
    CONTEXT_TREE_SOURCE     Path to pre-curated context tree (isolated mode)
    REPORT_DIR              Where to save reports (default: BENCH_REPO/report)
    JUDGE_CONCURRENCY       Parallel judge calls (default: 1)
    JUSTIFIER_CONCURRENCY   Parallel justifier calls (default: 1)
    LIMIT                   Max results per query (default: 32)
EOF
    exit 1
}

[[ $# -lt 1 ]] && usage
MODEL="$1"

# Validate required env vars
if [[ -z "${GROUND_TRUTH}" ]]; then
    echo "Error: GROUND_TRUTH is required. Set it to the path of your ground truth JSON file."
    exit 1
fi

# ─── Helpers ─────────────────────────────────────────────────────────────────
log() { echo "==> $(date '+%H:%M:%S') $*"; }

check_ollama() {
    if ! curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
        echo "Error: Ollama is not running at ${OLLAMA_HOST}"
        exit 1
    fi
}

# ─── Step 1: Pull model ─────────────────────────────────────────────────────
pull_model() {
    log "Pulling model: ${MODEL}"
    ollama pull "${MODEL}"
}

# ─── Step 2: Warm up & collect tok/s ─────────────────────────────────────────
collect_speed_metrics() {
    log "Running warmup prompt to collect tok/s metrics"

    local response
    response=$(curl -sf "${OLLAMA_HOST}/api/generate" \
        -d "{\"model\": \"${MODEL}\", \"prompt\": \"${WARMUP_PROMPT}\", \"stream\": false}")

    PROMPT_EVAL_COUNT=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('prompt_eval_count', 0))")
    PROMPT_EVAL_DURATION=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('prompt_eval_duration', 1))")
    EVAL_COUNT=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('eval_count', 0))")
    EVAL_DURATION=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('eval_duration', 1))")

    # Ollama durations are in nanoseconds
    GEN_TOKS=$(python3 -c "print(f'{${EVAL_COUNT} / (${EVAL_DURATION} / 1e9):.1f}')")
    PROMPT_TOKS=$(python3 -c "print(f'{${PROMPT_EVAL_COUNT} / (${PROMPT_EVAL_DURATION} / 1e9):.1f}')")

    log "Gen tok/s: ${GEN_TOKS} | Prompt tok/s: ${PROMPT_TOKS}"
}

# ─── Step 3: Collect VRAM ────────────────────────────────────────────────────
collect_vram() {
    log "Collecting VRAM usage"

    local ps_response
    ps_response=$(curl -sf "${OLLAMA_HOST}/api/ps")

    VRAM_BYTES=$(echo "$ps_response" | python3 -c "
import sys, json
model = sys.argv[1]
data = json.load(sys.stdin)
for m in data.get('models', []):
    if m.get('model', '') == model or m.get('name', '') == model:
        print(m.get('size_vram', 0))
        break
else:
    print(0)
" "${MODEL}")
    VRAM_GB=$(python3 -c "print(f'{int(${VRAM_BYTES}) / (1024**3):.1f}')")

    log "VRAM: ${VRAM_GB} GB"
}

# ─── Step 4: Run benchmark ───────────────────────────────────────────────────
run_benchmark() {
    log "Running LongMemEval-S benchmark"

    mkdir -p "${REPORT_DIR}"
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')
    local safe_model_name
    safe_model_name=$(echo "${MODEL}" | tr '/:' '_')
    BENCH_OUTPUT="${REPORT_DIR}/${timestamp}_${safe_model_name}.json"

    # shellcheck source=/dev/null
    source "${BENCH_REPO}/.venv/bin/activate"

    local ctx_flag=()
    if [[ -n "${CONTEXT_TREE_SOURCE}" ]]; then
        ctx_flag=(--context-tree-source "${CONTEXT_TREE_SOURCE}")
    fi

    python -m brv_bench evaluate \
        --ground-truth "${GROUND_TRUTH}" \
        --judge \
        --judge-backend ollama \
        --judge-model "${MODEL}" \
        --judge-concurrency "${JUDGE_CONCURRENCY}" \
        --justifier-backend ollama \
        --justifier-model "${MODEL}" \
        --justifier-concurrency "${JUSTIFIER_CONCURRENCY}" \
        "${ctx_flag[@]}" \
        --output "${BENCH_OUTPUT}" \
        --limit "${LIMIT}"

    log "Benchmark results saved to: ${BENCH_OUTPUT}"
}

# ─── Step 5: Save report ─────────────────────────────────────────────────────
save_report() {
    log "Saving metrics report"

    local safe_model_name
    safe_model_name=$(echo "${MODEL}" | tr '/:' '_')
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')
    METRICS_FILE="${REPORT_DIR}/${timestamp}_${safe_model_name}_metrics.json"

    python3 -c "
import json, sys
model, gen_toks, prompt_toks, vram_gb, bench_output, metrics_file = sys.argv[1:7]
report = {
    'model': model,
    'gen_tok_s': float(gen_toks),
    'prompt_tok_s': float(prompt_toks),
    'vram_gb': float(vram_gb),
    'benchmark_output': bench_output,
}
with open(metrics_file, 'w') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
" "${MODEL}" "${GEN_TOKS}" "${PROMPT_TOKS}" "${VRAM_GB}" "${BENCH_OUTPUT}" "${METRICS_FILE}"

    log "Metrics report saved to: ${METRICS_FILE}"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
log "Starting benchmark for model: ${MODEL}"
check_ollama
pull_model
collect_speed_metrics
collect_vram
run_benchmark
save_report
log "Done!"
