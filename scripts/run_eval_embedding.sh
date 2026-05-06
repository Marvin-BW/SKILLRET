#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Worker-pool multi-GPU embedding evaluator
#
# Evaluates all embedding models in EMBEDDING_MODELS (config.py) against the
# SkillRet benchmark (6,660 skills, auto-downloaded from HuggingFace).
# Models are loaded by HuggingFace ID and cached under $HF_HOME.
#
# Usage:
#   NUM_GPUS=8 bash scripts/run_eval_embedding.sh
#   MODELS_FILTER="KaLM,jina" bash scripts/run_eval_embedding.sh
#
# Environment variables:
#   NUM_GPUS        Number of GPUs (default: 8)
#   MODELS_FILTER   Comma-separated substrings to match model names (optional)
#   TOP_K           Number of results per query (default: 50)
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -z "${SKIP_VENV_ACTIVATE:-}" && -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

NUM_GPUS="${NUM_GPUS:-8}"
TOP_K="${TOP_K:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-results/embed}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Build model list from config.py
# ---------------------------------------------------------------------------
mapfile -t ALL_MODELS < <(
    python -c "
from skillret.config import EMBEDDING_MODELS
for m in EMBEDDING_MODELS:
    print(m)
"
)

MODELS=()
for model_id in "${ALL_MODELS[@]}"; do
    [[ "$model_id" == "BM25" ]] && continue
    if [[ -n "${MODEL_BASE_DIR:-}" ]]; then
        [[ ! -d "${MODEL_BASE_DIR}/${model_id}" ]] && { echo "WARN: skipping $model_id (not found at ${MODEL_BASE_DIR}/${model_id})"; continue; }
    fi
    if [[ -n "${MODELS_FILTER:-}" ]]; then
        match=0
        IFS=',' read -ra FILTERS <<< "$MODELS_FILTER"
        for filt in "${FILTERS[@]}"; do
            filt="$(echo "$filt" | xargs)"
            if [[ "$model_id" == *"$filt"* ]]; then
                match=1
                break
            fi
        done
        [[ $match -eq 0 ]] && continue
    fi
    MODELS+=("$model_id")
done

TOTAL=${#MODELS[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "ERROR: no models matched. Check MODELS_FILTER."
    exit 1
fi
echo "=========================================="
echo " Embedding evaluation: $TOTAL models, $NUM_GPUS GPUs"
echo " TOP_K=$TOP_K"
echo "=========================================="
printf '  %s\n' "${MODELS[@]}"
echo ""

# ---------------------------------------------------------------------------
# BM25 baseline (CPU-only, runs before GPU workers)
# ---------------------------------------------------------------------------
BM25_OUTPUT="${OUTPUT_DIR}/BM25.json"
if [[ -f "$BM25_OUTPUT" ]]; then
    echo "BM25: SKIP (exists): $BM25_OUTPUT"
else
    echo "BM25: running baseline..."
    python -c "
from skillret.eval import eval_bm25, print_results
results = eval_bm25(top_k=${TOP_K}, output_file='${BM25_OUTPUT}')
print_results(results)
" 2>&1 | tee "${OUTPUT_DIR}/BM25.log"
    echo "BM25: done"
fi
echo ""

# ---------------------------------------------------------------------------
# Queue management (flock-based)
# ---------------------------------------------------------------------------
QUEUE_DIR="$(mktemp -d)"
QUEUE_FILE="$QUEUE_DIR/queue.txt"
QUEUE_LOCK="$QUEUE_DIR/queue.lock"
DONE_FILE="$QUEUE_DIR/done.count"

printf '%s\n' "${MODELS[@]}" > "$QUEUE_FILE"
echo 0 > "$DONE_FILE"

pop_next_model() {
    (
        flock 9
        local model
        model="$(head -n1 "$QUEUE_FILE" 2>/dev/null || true)"
        if [[ -n "$model" ]]; then
            tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp"
            mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
        fi
        echo "$model"
    ) 9>"$QUEUE_LOCK"
}

increment_done() {
    (
        flock 9
        echo $(( $(cat "$DONE_FILE") + 1 )) > "$DONE_FILE"
    ) 9>"$QUEUE_LOCK"
}

get_done_count() {
    ( flock 9; cat "$DONE_FILE" ) 9>"$QUEUE_LOCK"
}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
run_eval() {
    local gpu_id="$1" model_id="$2"
    local short_name="${model_id//\//_}"
    local output_json="${OUTPUT_DIR}/${short_name}.json"
    local log="${OUTPUT_DIR}/${short_name}.log"

    if [[ -f "$output_json" ]]; then
        echo "[GPU $gpu_id] SKIP (exists): $output_json"
        return 0
    fi

    local model_ref="${MODEL_BASE_DIR:+${MODEL_BASE_DIR}/}${model_id}"
    echo "[GPU $gpu_id] Eval: $model_ref → $log"
    CUDA_VISIBLE_DEVICES="$gpu_id" python -c "
from skillret.eval import eval_retrieval, print_results
results = eval_retrieval(
    model_path='${model_ref}',
    top_k=${TOP_K},
    output_file='${output_json}',
)
print_results(results)
" 2>&1 | tee "$log"
}

# ---------------------------------------------------------------------------
# GPU worker loop
# ---------------------------------------------------------------------------
gpu_worker() {
    local gpu_id="$1"
    while true; do
        local model_id
        model_id="$(pop_next_model)"
        [[ -z "$model_id" ]] && break

        echo "[GPU $gpu_id] ▶ $model_id"
        if run_eval "$gpu_id" "$model_id"; then
            echo "[GPU $gpu_id] ✓ Done: $model_id"
        else
            echo "[GPU $gpu_id] ✗ FAILED: $model_id" >&2
        fi
        increment_done
    done
}

# ---------------------------------------------------------------------------
# Launch workers and wait
# ---------------------------------------------------------------------------
MAIN_LOG="${OUTPUT_DIR}/run_${TIMESTAMP}.log"
echo "Main log: $MAIN_LOG"

WORKER_PIDS=()
for (( gpu=0; gpu < NUM_GPUS && gpu < TOTAL; gpu++ )); do
    gpu_worker "$gpu" >> "$MAIN_LOG" 2>&1 &
    WORKER_PIDS+=($!)
done
echo "Launched $((${#WORKER_PIDS[@]})) workers"

while true; do
    done_count="$(get_done_count)"
    still_running=0
    for pid in "${WORKER_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && (( still_running++ )) || true
    done
    echo "[$(date '+%H:%M:%S')] Progress: $done_count/$TOTAL done, $still_running workers active"
    [[ $still_running -eq 0 ]] && break
    sleep 30
done

echo ""
echo "=========================================="
echo " All $TOTAL evaluations complete."
echo " Results: ${OUTPUT_DIR}/*.json"
echo "=========================================="

rm -rf "$QUEUE_DIR"
