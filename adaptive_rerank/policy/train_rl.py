"""REINFORCE training for the adaptive body-inclusion policy.

Trains the ``PolicyMLP`` on the offline cache produced by
``offline_cache.py``. Each training "episode" is one query: the policy
samples an action vector for the top-K candidates, the resulting score
sequence is composed from the cached metadata/body scores, the NDCG@k
under that action vector is computed, and a token-cost penalty is
subtracted. A moving-average baseline is used to reduce variance.

The training loop never invokes the reranker.
"""

from __future__ import annotations

import argparse
import json
import math
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
)
from .offline_cache import load_cache


def _featwise_normalizer(records: list[QueryRecord]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([r.features for r in records], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def _evaluate(
    policy: PolicyMLP,
    records: Sequence[QueryRecord],
    mean: np.ndarray,
    std: np.ndarray,
    k: int,
    device: torch.device,
    greedy: bool = True,
) -> dict:
    policy.eval()
    ndcgs = []
    token_ratios = []
    with torch.no_grad():
        for rec in records:
            x = (rec.features - mean) / std
            xt = torch.from_numpy(x).to(device)
            p = policy.action_prob(xt).cpu().numpy()
            actions = (p >= 0.5).astype(np.int64) if greedy else (np.random.rand(len(p)) < p).astype(np.int64)
            scores = rec.combined_scores(actions)
            ndcgs.append(ndcg_at_k(scores, rec.labels, k))
            token_ratios.append(rec.token_cost(actions))
    return {
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "token_ratio": float(np.mean(token_ratios)) if token_ratios else 0.0,
        "n": len(ndcgs),
    }


def train(
    cache_path: str,
    *,
    out_path: str,
    k: int = 10,
    lam: float = 0.1,
    lr: float = 3e-4,
    epochs: int = 10,
    hidden: int = 128,
    n_layers: int = 2,
    val_ratio: float = 0.1,
    baseline_momentum: float = 0.9,
    entropy_coeff: float = 1e-3,
    seed: int = 0,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[train_rl] loading cache: {cache_path}")
    cache = load_cache(cache_path)
    records = cache_to_records(cache)
    print(f"[train_rl] {len(records)} queries (top_k={cache['top_k']})")

    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = max(1, int(round(len(records) * val_ratio)))
    val = records[:n_val]
    train_recs = records[n_val:]
    print(f"[train_rl] split: train={len(train_recs)} val={len(val)}")

    mean, std = _featwise_normalizer(train_recs)
    device = torch.device(device_str)
    policy = PolicyMLP(hidden=hidden, n_layers=n_layers).to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)

    # Sanity baselines.
    def _const_eval(action_value: int) -> dict:
        ndcgs, tr = [], []
        for r in val:
            a = np.full(len(r.labels), action_value, dtype=np.int64)
            ndcgs.append(ndcg_at_k(r.combined_scores(a), r.labels, k))
            tr.append(r.token_cost(a))
        return {"ndcg@k": float(np.mean(ndcgs)), "token_ratio": float(np.mean(tr))}

    print(f"[train_rl] baseline all-meta: {_const_eval(0)}")
    print(f"[train_rl] baseline all-body: {_const_eval(1)}")

    moving_baseline = 0.0
    step = 0
    history = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(train_recs)
        epoch_reward = 0.0
        for rec in train_recs:
            x = (rec.features - mean) / std
            xt = torch.from_numpy(x).to(device)
            logits = policy(xt)
            probs = torch.sigmoid(logits)
            dist = torch.distributions.Bernoulli(probs=probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            a_np = actions.detach().cpu().numpy().astype(np.int64)
            scores = rec.combined_scores(a_np)
            ndcg = ndcg_at_k(scores, rec.labels, k)
            tok_ratio = rec.token_cost(a_np)
            reward = ndcg - lam * tok_ratio

            # Moving-average baseline.
            moving_baseline = (
                baseline_momentum * moving_baseline
                + (1 - baseline_momentum) * reward
            )
            advantage = reward - moving_baseline

            # REINFORCE with entropy bonus.
            policy_loss = -(advantage * log_probs.sum())
            entropy = dist.entropy().sum()
            loss = policy_loss - entropy_coeff * entropy

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            epoch_reward += reward
            step += 1

        avg_reward = epoch_reward / max(len(train_recs), 1)
        val_metrics = _evaluate(policy, val, mean, std, k, device, greedy=True)
        print(
            f"[train_rl] epoch {epoch:3d} | avg_reward={avg_reward:.4f} "
            f"| val NDCG@{k}={val_metrics['ndcg@k']:.4f} "
            f"| val tok_ratio={val_metrics['token_ratio']:.4f}"
        )
        history.append({
            "epoch": epoch,
            "avg_reward": avg_reward,
            "val_ndcg": val_metrics["ndcg@k"],
            "val_tok_ratio": val_metrics["token_ratio"],
        })

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "config": {
                "hidden": hidden,
                "n_layers": n_layers,
                "k": k,
                "lam": lam,
                "epochs": epochs,
                "lr": lr,
                "cache_path": cache_path,
            },
            "history": history,
        },
        out,
    )
    print(f"[train_rl] saved policy → {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="REINFORCE train adaptive-rerank policy.")
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train(
        cache_path=args.cache,
        out_path=args.out,
        k=args.k,
        lam=args.lam,
        lr=args.lr,
        epochs=args.epochs,
        hidden=args.hidden,
        n_layers=args.n_layers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
