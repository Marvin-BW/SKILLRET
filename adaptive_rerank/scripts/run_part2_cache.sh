#!/usr/bin/env bash
set -euo pipefail
# Build offline cache for RL policy training (train split, 10% subset).
#
# Env vars
#   RERANKER        reranker HF id  (default: Qwen/Qwen3-Reranker-0.6B)
#   FIRST_STAGE     first-stage JSON (default: results/embed/ThakiCloud_SKILLRET-Embedding-0.6B.json
#                   — if absent, falls back to running embedding fresh with EMBED_MODEL)
#   EMBED_MODEL     fallback first-stage embedder (default: ThakiCloud/SKILLRET-Embedding-0.6B)
#   SUBSET_RATIO    train subset ratio (default: 0.1)
#   SPLIT           dataset split   (default: train)
#   TOP_K           candidate depth (default: 20)
#   OUT             output cache path (default: adaptive_rerank/results/part2/cache.train10.pkl.gz)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

if [[ -z "${SKIP_VENV_ACTIVATE:-}" && -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi
[[ -f "$PROJECT_DIR/.env" ]] && { set -a; source "$PROJECT_DIR/.env"; set +a; }

RERANKER="${RERANKER:-Qwen/Qwen3-Reranker-0.6B}"
FIRST_STAGE="${FIRST_STAGE:-results/embed/ThakiCloud_SKILLRET-Embedding-0.6B.json}"
EMBED_MODEL="${EMBED_MODEL:-ThakiCloud/SKILLRET-Embedding-0.6B}"
SUBSET_RATIO="${SUBSET_RATIO:-0.1}"
SPLIT="${SPLIT:-train}"
TOP_K="${TOP_K:-20}"
OUT="${OUT:-adaptive_rerank/results/part2/cache.${SPLIT}${SUBSET_RATIO}.pkl.gz}"

mkdir -p "$(dirname "$OUT")"

ARGS=(--out "$OUT" --reranker "$RERANKER" --split "$SPLIT" --subset-ratio "$SUBSET_RATIO" --top-k "$TOP_K")
if [[ -f "$FIRST_STAGE" && "$SPLIT" == "test" ]]; then
    ARGS+=(--first-stage "$FIRST_STAGE")
else
    ARGS+=(--embed-model "$EMBED_MODEL")
fi

echo "[cache] python -m adaptive_rerank.policy.offline_cache ${ARGS[*]}"
python -m adaptive_rerank.policy.offline_cache "${ARGS[@]}"
