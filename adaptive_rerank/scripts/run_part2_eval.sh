#!/usr/bin/env bash
set -euo pipefail
# Evaluate trained policies on the test split; writes JSONs that ``analyze.py``
# picks up alongside the Part 1 results.
#
# Env vars
#   RERANKER     (default: Qwen/Qwen3-Reranker-0.6B)
#   FIRST_STAGE  (default: results/embed/ThakiCloud_SKILLRET-Embedding-0.6B.json)
#   CKPT_DIR     (default: adaptive_rerank/results/part2/ckpts)
#   OUT_DIR      (default: adaptive_rerank/results/part1)   ← intentional: same dir as Part 1
#                so analyze.py picks it up as another allocator
#   THRESHOLD    decision threshold for P(use_body) (default: 0.5)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
if [[ -z "${SKIP_VENV_ACTIVATE:-}" && -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi
[[ -f "$PROJECT_DIR/.env" ]] && { set -a; source "$PROJECT_DIR/.env"; set +a; }

RERANKER="${RERANKER:-Qwen/Qwen3-Reranker-0.6B}"
FIRST_STAGE="${FIRST_STAGE:-results/embed/ThakiCloud_SKILLRET-Embedding-0.6B.json}"
CKPT_DIR="${CKPT_DIR:-adaptive_rerank/results/part2/ckpts}"
OUT_DIR="${OUT_DIR:-adaptive_rerank/results/part1}"
THRESHOLD="${THRESHOLD:-0.5}"

mkdir -p "$OUT_DIR"

for ckpt in "$CKPT_DIR"/policy_*.pt; do
    [[ -f "$ckpt" ]] || { echo "no ckpts in $CKPT_DIR"; exit 1; }
    base="$(basename "$ckpt" .pt)"
    rr="${RERANKER//\//_}"
    fs_short="$(basename "$FIRST_STAGE" .json)"
    out="${OUT_DIR}/rerank_${rr}_${fs_short}_rl_${base}_thr${THRESHOLD}.json"
    if [[ -f "$out" ]]; then
        echo "SKIP $out"; continue
    fi
    echo "▶ $base → $out"
    python -m adaptive_rerank.policy.eval_adaptive \
        --reranker "$RERANKER" \
        --first-stage "$FIRST_STAGE" \
        --policy-ckpt "$ckpt" \
        --threshold "$THRESHOLD" \
        --output "$out"
done
