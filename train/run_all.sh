#!/bin/bash
# Sequential training (MAX BATCH mode): harrier-0.6b → Qwen3-8B (4 GPU DDP)
# - harrier: batch=96 + GC   (effective=384, ~5.7h)
# - qwen3-8b: batch=20 + GC  (effective=80,  ~?h)
# - GPU cleanup guaranteed between runs
# - Qwen3 runs even if harrier fails
#
# Usage:
#   cd /path/to/project
#   nohup bash train/run_all.sh > train/run_all.log 2>&1 &

set -u   # fail on undefined vars (but NOT -e so qwen3 runs even if harrier fails)

cd /path/to/project
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=skillret

# ============================================================
# Pre-flight: check no other training is running
# ============================================================
echo "[$(date)] ========== PRE-FLIGHT CHECK =========="
RUNNING=$(pgrep -f "torchrun.*train.py" | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "[$(date)] ERROR: Another training process is still running:"
    pgrep -af "torchrun.*train.py"
    echo "[$(date)] Aborting to avoid conflict."
    exit 1
fi
echo "[$(date)] No conflicting training processes. OK."

echo "[$(date)] GPU status before training:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo ""

cleanup_gpu() {
    echo "[$(date)] Cleaning up GPU processes..."
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        echo "  Killing leftover GPU process: PID $pid"
        kill -9 "$pid" 2>/dev/null
    done
    sleep 5
    echo "[$(date)] GPU status after cleanup:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    echo ""
}

wait_gpu_free() {
    local max_wait=120
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        local busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000 {print}' | wc -l)
        if [ "$busy" -eq 0 ]; then
            echo "[$(date)] All GPUs are free."
            return 0
        fi
        echo "[$(date)] GPUs still occupied ($busy busy), waiting... (${elapsed}s / ${max_wait}s)"
        sleep 10
        elapsed=$((elapsed + 10))
    done
    echo "[$(date)] WARNING: GPUs not fully free after ${max_wait}s. Force cleaning..."
    cleanup_gpu
}

# ============================================================
# Phase 1: harrier-0.6b (MAX BATCH: 96 × 4 = 384, with GC)
# ============================================================
echo "============================================"
echo "[$(date)] Phase 1: Starting harrier-0.6b (batch=96, GC)"
echo "  Expected duration: ~5.7 hours"
echo "  Output: train/4gpu-harrier-0.6b/outputs-max/"
echo "============================================"

torchrun --nproc_per_node=4 train/4gpu-harrier-0.6b/train.py
HARRIER_EXIT=$?

if [ $HARRIER_EXIT -eq 0 ]; then
    echo "[$(date)] harrier-0.6b DONE (exit code: 0)"
else
    echo "[$(date)] harrier-0.6b FAILED (exit code: $HARRIER_EXIT)"
    echo "[$(date)] Proceeding to Qwen3 anyway..."
fi

# ============================================================
# GPU Cleanup between runs
# ============================================================
echo ""
echo "[$(date)] Ensuring GPU cleanup before Qwen3..."
sleep 10
wait_gpu_free

# ============================================================
# Phase 2: Qwen3-Embedding-8B (MAX BATCH: 20 × 4 = 80, with GC)
# ============================================================
echo "============================================"
echo "[$(date)] Phase 2: Starting Qwen3-Embedding-8B (batch=20, GC)"
echo "  Output: train/4gpu-qwen3-8b/outputs/"
echo "============================================"

torchrun --nproc_per_node=4 train/4gpu-qwen3-8b/train.py
QWEN_EXIT=$?

if [ $QWEN_EXIT -eq 0 ]; then
    echo "[$(date)] Qwen3-8B DONE (exit code: 0)"
else
    echo "[$(date)] Qwen3-8B FAILED (exit code: $QWEN_EXIT)"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
echo "[$(date)] ALL TRAINING COMPLETE"
echo "  harrier-0.6b  exit code: $HARRIER_EXIT"
echo "  Qwen3-8B      exit code: $QWEN_EXIT"
echo "============================================"
