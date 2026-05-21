#!/usr/bin/env bash
set -euo pipefail
# Train RL policy + supervised baseline.
#
# Env vars
#   CACHE       path to cache.*.pkl.gz  (default: adaptive_rerank/results/part2/cache.train0.1.pkl.gz)
#   OUT_DIR     output dir for ckpts    (default: adaptive_rerank/results/part2/ckpts)
#   K           NDCG cutoff             (default: 10)
#   LAM         token-cost weight       (default: 0.1; sweep, e.g. 0,0.05,0.1,0.2)
#   EPOCHS_RL   (default: 10)
#   EPOCHS_SUP  (default: 20)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
if [[ -z "${SKIP_VENV_ACTIVATE:-}" && -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi
[[ -f "$PROJECT_DIR/.env" ]] && { set -a; source "$PROJECT_DIR/.env"; set +a; }

CACHE="${CACHE:-adaptive_rerank/results/part2/cache.train0.1.pkl.gz}"
OUT_DIR="${OUT_DIR:-adaptive_rerank/results/part2/ckpts}"
K="${K:-10}"
LAM="${LAM:-0.1}"
EPOCHS_RL="${EPOCHS_RL:-10}"
EPOCHS_SUP="${EPOCHS_SUP:-20}"

mkdir -p "$OUT_DIR"

echo "[train] supervised baseline"
python -m adaptive_rerank.policy.train_sup \
    --cache "$CACHE" \
    --out "$OUT_DIR/policy_sup.pt" \
    --k "$K" --epochs "$EPOCHS_SUP"

IFS=',' read -ra LAMS <<< "$LAM"
for lam in "${LAMS[@]}"; do
    echo "[train] REINFORCE lam=$lam"
    python -m adaptive_rerank.policy.train_rl \
        --cache "$CACHE" \
        --out "$OUT_DIR/policy_rl_lam${lam}.pt" \
        --k "$K" --lam "$lam" --epochs "$EPOCHS_RL"
done

echo "[train] done. Checkpoints in: $OUT_DIR"
