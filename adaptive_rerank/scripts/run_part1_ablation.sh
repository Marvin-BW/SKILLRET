#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Part 1 ablation: rerank × allocator × body-ratio × seed grid.
#
# Reuses the frozen reranker code under ``skillret/`` and only writes results
# under ``adaptive_rerank/results/part1/``.
#
# Env vars
#   NUM_GPUS              parallel workers (default: 1)
#   RERANKERS             comma-separated reranker HF ids
#                         (default: jina + Qwen3-0.6B + SkillRet-0.6B)
#   ALLOCATORS            comma-separated allocator names
#                         (default: random,rank_top,rank_bottom,oracle)
#   RATIOS                comma-separated body ratios in [0, 1]
#                         (default: 0.0,0.25,0.5,0.75,1.0)
#   SEEDS                 comma-separated seeds (only used for ``random``;
#                         other allocators always use the first seed)
#                         (default: 0,1,2)
#   FIRST_STAGE           first-stage retrieval JSON path
#                         (default: results/embed/ThakiCloud_SKILLRET-Embedding-0.6B.json)
#   TOP_K                 first-stage depth (default: 20)
#   OUTPUT_DIR            (default: adaptive_rerank/results/part1)
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

if [[ -z "${SKIP_VENV_ACTIVATE:-}" && -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

NUM_GPUS="${NUM_GPUS:-1}"
TOP_K="${TOP_K:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-adaptive_rerank/results/part1}"
FIRST_STAGE="${FIRST_STAGE:-results/embed/ThakiCloud_SKILLRET-Embedding-0.6B.json}"
RERANKERS_DEFAULT="jinaai/jina-reranker-v2-base-multilingual,Qwen/Qwen3-Reranker-0.6B,ThakiCloud/SKILLRET-Reranker-0.6B"
RERANKERS="${RERANKERS:-$RERANKERS_DEFAULT}"
ALLOCATORS="${ALLOCATORS:-random,rank_top,rank_bottom,oracle}"
RATIOS="${RATIOS:-0.0,0.25,0.5,0.75,1.0}"
SEEDS="${SEEDS:-0,1,2}"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$FIRST_STAGE" ]]; then
    echo "ERROR: first-stage file not found: $FIRST_STAGE" >&2
    echo "  Run scripts/run_eval_embedding.sh first (or set FIRST_STAGE)." >&2
    exit 1
fi

IFS=',' read -ra RERANKER_LIST <<< "$RERANKERS"
IFS=',' read -ra ALLOC_LIST    <<< "$ALLOCATORS"
IFS=',' read -ra RATIO_LIST    <<< "$RATIOS"
IFS=',' read -ra SEED_LIST     <<< "$SEEDS"

JOBS=()
for reranker in "${RERANKER_LIST[@]}"; do
    for allocator in "${ALLOC_LIST[@]}"; do
        for ratio in "${RATIO_LIST[@]}"; do
            # ratios 0.0 and 1.0 are independent of allocator → only emit once.
            if [[ "$ratio" == "0.0" || "$ratio" == "1.0" ]] && [[ "$allocator" != "random" ]]; then
                continue
            fi
            if [[ "$allocator" == "random" ]]; then
                for seed in "${SEED_LIST[@]}"; do
                    JOBS+=("$reranker|$allocator|$ratio|$seed")
                done
            else
                JOBS+=("$reranker|$allocator|$ratio|0")
            fi
        done
    done
done

TOTAL=${#JOBS[@]}
echo "================================================="
echo " Part 1 ablation: $TOTAL jobs"
echo "  rerankers : ${RERANKER_LIST[*]}"
echo "  allocators: ${ALLOC_LIST[*]}"
echo "  ratios    : ${RATIO_LIST[*]}"
echo "  seeds(rnd): ${SEED_LIST[*]}"
echo "  first-stg : $FIRST_STAGE"
echo "  top-k     : $TOP_K"
echo "  out-dir   : $OUTPUT_DIR"
echo "================================================="

run_job() {
    local gpu_id="$1" reranker="$2" allocator="$3" ratio="$4" seed="$5"
    local short="${reranker//\//_}"
    local stage_short
    stage_short="$(basename "$FIRST_STAGE" .json)"
    local out="${OUTPUT_DIR}/rerank_${short}_${stage_short}_${allocator}_p${ratio}_s${seed}.json"
    local log="${out%.json}.log"

    if [[ -f "$out" ]]; then
        echo "[GPU $gpu_id] SKIP $out"
        return 0
    fi
    echo "[GPU $gpu_id] ▶ $short / $allocator / p=$ratio / seed=$seed → $out"
    CUDA_VISIBLE_DEVICES="$gpu_id" python -m adaptive_rerank.eval_ablation \
        --reranker "$reranker" \
        --first-stage "$FIRST_STAGE" \
        --allocator "$allocator" \
        --ratio "$ratio" \
        --seed "$seed" \
        --top-k "$TOP_K" \
        --output "$out" 2>&1 | tee "$log"
}

# Simple GPU pool via flock.
QUEUE_DIR="$(mktemp -d)"
Q="$QUEUE_DIR/q"; L="$QUEUE_DIR/lock"
printf '%s\n' "${JOBS[@]}" > "$Q"
pop() { ( flock 9; head -n1 "$Q" 2>/dev/null; tail -n +2 "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q" ) 9>"$L"; }

worker() {
    local gpu_id="$1"
    while :; do
        local job; job="$(pop)"
        [[ -z "$job" ]] && break
        IFS='|' read -r rr al rt sd <<< "$job"
        run_job "$gpu_id" "$rr" "$al" "$rt" "$sd" || echo "[GPU $gpu_id] FAILED: $job" >&2
    done
}

PIDS=()
for ((g=0; g<NUM_GPUS && g<TOTAL; g++)); do
    worker "$g" &
    PIDS+=($!)
done
wait "${PIDS[@]}" || true
rm -rf "$QUEUE_DIR"

echo "Part 1 ablation done. Results in: $OUTPUT_DIR"
echo "Run: python -m adaptive_rerank.analyze --results-dir $OUTPUT_DIR"
