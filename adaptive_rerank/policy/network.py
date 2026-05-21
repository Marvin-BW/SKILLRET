"""Small MLP policy + helpers for computing NDCG from cached scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .features import FEATURE_DIM


class PolicyMLP(nn.Module):
    """Tiny 2-layer MLP returning logits for {use_metadata, use_body}.

    Sigmoid on the single output ⇒ P(use_body).
    """

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU()]
            prev = hidden
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, D) → (N,)
        return self.net(x).squeeze(-1)

    def action_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


# ---------------------------------------------------------------------------
# NDCG@k computed from a 1-D score array and 0/1 label array, both ordered
# by the cached candidate order.
# ---------------------------------------------------------------------------

def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Standard NDCG@k with binary relevance.

    Args:
        scores: predicted score per candidate (higher = better).
        labels: 0/1 relevance per candidate, parallel to ``scores``.
        k: cutoff.
    """
    n = len(scores)
    if n == 0 or labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores)
    gains = labels[order][:k].astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())
    # Ideal DCG.
    ideal = np.sort(labels)[::-1][:k].astype(np.float64)
    idcg = float((ideal * (1.0 / np.log2(np.arange(2, ideal.size + 2)))).sum())
    if idcg == 0:
        return 0.0
    return dcg / idcg


@dataclass
class QueryRecord:
    """Adapter view over one row of the offline cache."""

    qid: str
    features: np.ndarray          # (N, FEATURE_DIM)
    scores_meta: np.ndarray       # (N,)
    scores_body: np.ndarray       # (N,)
    labels: np.ndarray            # (N,) 0/1
    meta_tokens: np.ndarray       # (N,)
    body_tokens: np.ndarray       # (N,)

    def combined_scores(self, actions: np.ndarray) -> np.ndarray:
        """Return the per-candidate score under a chosen action vector.

        actions: 0/1 (0=metadata, 1=body).
        """
        return np.where(actions.astype(bool), self.scores_body, self.scores_meta)

    def token_cost(self, actions: np.ndarray) -> float:
        used = np.where(actions.astype(bool), self.body_tokens, self.meta_tokens).sum()
        full = self.body_tokens.sum()
        return float(used / max(full, 1))


def cache_to_records(cache: dict) -> list[QueryRecord]:
    out = []
    for rec in cache["records"]:
        out.append(
            QueryRecord(
                qid=rec["qid"],
                features=rec["features"],
                scores_meta=rec["scores_meta"],
                scores_body=rec["scores_body"],
                labels=rec["labels"],
                meta_tokens=rec["meta_tokens"],
                body_tokens=rec["body_tokens"],
            )
        )
    return out


# ---------------------------------------------------------------------------
# Oracle action for supervised baseline: action that maximises NDCG@k on this
# query alone (greedy over candidates by score gap).
# ---------------------------------------------------------------------------

def oracle_actions(rec: QueryRecord, k: int = 10) -> np.ndarray:
    """Greedy per-candidate oracle.

    Start with all-metadata; flip the candidate whose flip yields the
    largest NDCG@k improvement; repeat until no flip helps.
    """
    n = len(rec.labels)
    actions = np.zeros(n, dtype=np.int64)
    base_score = ndcg_at_k(rec.combined_scores(actions), rec.labels, k)
    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_idx = -1
        for i in range(n):
            actions[i] = 1 - actions[i]
            new_score = ndcg_at_k(rec.combined_scores(actions), rec.labels, k)
            actions[i] = 1 - actions[i]
            gain = new_score - base_score
            if gain > best_gain + 1e-9:
                best_gain = gain
                best_idx = i
        if best_idx >= 0:
            actions[best_idx] = 1 - actions[best_idx]
            base_score += best_gain
            improved = True
    return actions
