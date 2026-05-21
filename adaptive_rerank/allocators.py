"""Body-allocation strategies for the Part 1 ablation.

Given a list of candidate skills (already ranked by the first-stage
retriever, best first) and a body-budget ratio ``p`` ∈ [0, 1], return a
``list[bool]`` of the same length where ``True`` means *use the full body*
for that candidate and ``False`` means *metadata only*.

Strategies
----------
``random``
    Pick ``round(p * N)`` candidates uniformly at random.
``rank_top``
    Give body to the ``round(p * N)`` top-ranked candidates.
``rank_bottom``
    Give body to the ``round(p * N)`` bottom-ranked candidates.
``oracle``
    Cheating baseline that uses ground-truth labels — give body to the
    candidates that are GT positives first, then fill the remaining budget
    with rank-top candidates. This is the upper bound that the Part 2 RL
    policy will be measured against.
"""

from __future__ import annotations

import random
from typing import Iterable


def _budget(n: int, ratio: float) -> int:
    if n <= 0:
        return 0
    return int(round(max(0.0, min(1.0, ratio)) * n))


def allocate_random(n: int, ratio: float, seed: int = 0) -> list[bool]:
    rng = random.Random(seed)
    k = _budget(n, ratio)
    idxs = list(range(n))
    rng.shuffle(idxs)
    chosen = set(idxs[:k])
    return [i in chosen for i in range(n)]


def allocate_rank_top(n: int, ratio: float) -> list[bool]:
    """Top-ranked ``k`` candidates use body. Assumes input order is best-first."""
    k = _budget(n, ratio)
    return [i < k for i in range(n)]


def allocate_rank_bottom(n: int, ratio: float) -> list[bool]:
    """Bottom-ranked ``k`` candidates use body."""
    k = _budget(n, ratio)
    return [i >= n - k for i in range(n)]


def allocate_oracle(
    candidate_ids: list,
    gt_ids: Iterable,
    ratio: float,
) -> list[bool]:
    """Oracle: prefer GT positives, fill remaining budget with rank-top.

    Args:
        candidate_ids: candidate skill IDs in first-stage rank order.
        gt_ids: iterable of ground-truth positive skill IDs (as strings or ints
            — comparison done after str()).
        ratio: body-budget ratio.

    Returns:
        ``list[bool]`` aligned with ``candidate_ids``.
    """
    n = len(candidate_ids)
    k = _budget(n, ratio)
    gt_set = {str(g) for g in gt_ids}
    flags = [False] * n
    used = 0
    # Pass 1: mark GT positives in rank order.
    for i, cid in enumerate(candidate_ids):
        if used >= k:
            break
        if str(cid) in gt_set:
            flags[i] = True
            used += 1
    # Pass 2: fill from rank-top (skipping already chosen).
    if used < k:
        for i in range(n):
            if used >= k:
                break
            if not flags[i]:
                flags[i] = True
                used += 1
    return flags


def allocate(
    name: str,
    candidate_ids: list,
    ratio: float,
    *,
    seed: int = 0,
    gt_ids: Iterable | None = None,
) -> list[bool]:
    """Unified dispatcher.

    Always uses ``len(candidate_ids)`` as N; ``candidate_ids`` must be in
    first-stage rank order (best first).
    """
    n = len(candidate_ids)
    name = name.lower()
    if name == "random":
        return allocate_random(n, ratio, seed=seed)
    if name == "rank_top":
        return allocate_rank_top(n, ratio)
    if name == "rank_bottom":
        return allocate_rank_bottom(n, ratio)
    if name == "oracle":
        if gt_ids is None:
            raise ValueError("oracle allocator requires gt_ids")
        return allocate_oracle(candidate_ids, gt_ids, ratio)
    raise ValueError(f"Unknown allocator: {name!r}")


ALLOCATORS = ("random", "rank_top", "rank_bottom", "oracle")
