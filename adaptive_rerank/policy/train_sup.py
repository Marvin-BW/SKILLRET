"""Supervised oracle-imitation baseline.

For each query in the offline cache, compute the greedy oracle actions
(``oracle_actions`` in ``network.py``), then train the same ``PolicyMLP``
with per-candidate binary cross-entropy against those oracle labels.

This gives an upper bound on what the RL policy can achieve given the
same features and architecture.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import (
    PolicyMLP,
    QueryRecord,
    cache_to_records,
    ndcg_at_k,
    oracle_actions,
)
from .offline_cache import load_cache
from .train_rl import _featwise_normalizer, _evaluate


def _build_oracle_dataset(
    records: Sequence[QueryRecord], k: int
) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for r in records:
        a = oracle_actions(r, k=k)
        Xs.append(r.features)
        ys.append(a.astype(np.float32))
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def train(
    cache_path: str,
    *,
    out_path: str,
    k: int = 10,
    lr: float = 3e-4,
    epochs: int = 20,
    batch_size: int = 4096,
    hidden: int = 128,
    n_layers: int = 2,
    val_ratio: float = 0.1,
    seed: int = 0,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cache = load_cache(cache_path)
    records = cache_to_records(cache)
    print(f"[train_sup] {len(records)} queries (top_k={cache['top_k']})")

    rng = random.Random(seed); rng.shuffle(records)
    n_val = max(1, int(round(len(records) * val_ratio)))
    val_recs = records[:n_val]
    train_recs = records[n_val:]

    print(f"[train_sup] computing oracle labels on {len(train_recs)} train queries")
    X_train, y_train = _build_oracle_dataset(train_recs, k)
    mean, std = X_train.mean(axis=0).astype(np.float32), (X_train.std(axis=0) + 1e-6).astype(np.float32)
    X_train = (X_train - mean) / std

    device = torch.device(device_str)
    policy = PolicyMLP(hidden=hidden, n_layers=n_layers).to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)
    n = X_t.shape[0]
    history = []
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = policy(X_t[idx])
            loss = bce(logits, y_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += float(loss) * idx.numel()
        epoch_loss /= n
        val_m = _evaluate(policy, val_recs, mean, std, k, device, greedy=True)
        print(
            f"[train_sup] epoch {epoch:3d} | bce={epoch_loss:.4f} "
            f"| val NDCG@{k}={val_m['ndcg@k']:.4f} "
            f"| val tok_ratio={val_m['token_ratio']:.4f}"
        )
        history.append({"epoch": epoch, "bce": epoch_loss,
                        "val_ndcg": val_m["ndcg@k"],
                        "val_tok_ratio": val_m["token_ratio"]})

    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": policy.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "config": {"hidden": hidden, "n_layers": n_layers, "k": k,
                   "epochs": epochs, "lr": lr, "cache_path": cache_path},
        "history": history,
    }, out)
    print(f"[train_sup] saved → {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Supervised oracle-imitation baseline.")
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train(
        cache_path=args.cache,
        out_path=args.out,
        k=args.k,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        n_layers=args.n_layers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
