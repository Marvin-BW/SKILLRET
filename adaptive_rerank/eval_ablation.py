"""Part 1 entry point: rerank ablation across body-allocation strategies.

This module evaluates a reranker on the first-stage top-K results while
varying how many candidates per query are given their full body vs only
metadata. It re-uses helpers from ``skillret.eval`` without modifying any
file in that package.

Usage
-----
::

    from adaptive_rerank.eval_ablation import eval_rerank_ablation
    result = eval_rerank_ablation(
        reranker_model="Qwen/Qwen3-Reranker-0.6B",
        first_stage_file="results/embed/ThakiCloud_SKILLRET-Embedding-0.6B.json",
        allocator="oracle",
        body_ratio=0.5,
        output_file="adaptive_rerank/results/part1/run.json",
    )

The output JSON has the same shape as ``skillret.eval.eval_rerank`` but
includes an extra ``config`` block recording the ablation parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

# Re-use everything we need from the frozen ``skillret`` package.
from skillret.eval import (
    _load_reranker,
    _normalize_query_labels,
    _require_rerank_device,
    load_corpus,
    load_queries,
    trec_eval,
)
from skillret.config import RERANK_TOP_K, get_batch_size
from skillret.utils import load_json, write_json

from .allocators import allocate
from .text_builder import build_skill_text, approx_token_count


def _first_stage_for_split(first_stage_file: str, split: str) -> Dict[str, Dict[str, float]]:
    raw = load_json(first_stage_file)
    retrieval = raw.get("retrieval", raw)
    if split in retrieval:
        return retrieval[split]
    return next(iter(retrieval.values())) if retrieval else {}


def eval_rerank_ablation(
    reranker_model: str,
    first_stage_file: str,
    *,
    allocator: str,
    body_ratio: float,
    seed: int = 0,
    from_top_k: int = RERANK_TOP_K,
    output_file: str | None = None,
    rerank_batch_size: int = 0,
    split: str = "test",
) -> Dict[str, Dict[str, float]]:
    """Rerank first-stage candidates with a configurable body-allocation policy.

    Args:
        reranker_model: HF id or local path of the reranker.
        first_stage_file: JSON file produced by ``skillret.eval.eval_retrieval``.
        allocator: One of ``random``, ``rank_top``, ``rank_bottom``, ``oracle``.
        body_ratio: Fraction of candidates per query that should be given the
            full body. ``0.0`` ⇒ metadata-only for everyone (``all_meta``);
            ``1.0`` ⇒ full-body for everyone (``all_body``).
        seed: RNG seed for ``random`` allocator.
        from_top_k: Number of first-stage candidates to rerank per query.
        output_file: Optional path to write the result JSON.
        rerank_batch_size: 0 → auto-detect from skillret.config.
        split: Dataset split (default ``test``).

    Returns:
        ``{split: metrics_dict}`` — same shape as ``skillret.eval.eval_rerank``.
    """
    device = _require_rerank_device()
    print(f"[ablation] rerank device: {device}")
    if rerank_batch_size <= 0:
        rerank_batch_size = get_batch_size(reranker_model, "rerank")
        print(f"[ablation] auto rerank batch_size={rerank_batch_size} for {reranker_model}")
    print(f"[ablation] loading reranker: {reranker_model}")
    model = _load_reranker(reranker_model, device, rerank_batch_size=rerank_batch_size)

    skills = load_corpus(split=split)
    skill_map = {str(s["id"]): s for s in skills}
    print(f"[ablation] corpus: {len(skills)} skills ({split})")

    first_stage = _first_stage_for_split(first_stage_file, split)
    print(f"[ablation] loaded first-stage from {first_stage_file}")

    queries = load_queries(split=split)
    qrels: Dict[str, Dict[str, int]] = {}
    for item in queries:
        labels = _normalize_query_labels(item)
        qrels[item["id"]] = {str(x["id"]): int(x["relevance"]) for x in labels}

    # Build (query, doc_text) pairs respecting the allocator.
    use_multi = hasattr(model, "compute_rank_score_multi")
    all_pairs: list[tuple[str, str]] = []
    pair_map: list[tuple[str, list[dict], int, int, list[bool]]] = []
    body_chosen = 0
    body_total = 0
    tokens_used = 0
    tokens_full = 0

    for item in queries:
        qid = item["id"]
        cand_scores = first_stage.get(qid, {})
        sorted_cands = sorted(
            cand_scores.items(), key=lambda x: x[1], reverse=True
        )[:from_top_k]
        cand_skills = [skill_map[sid] for sid, _ in sorted_cands if sid in skill_map]
        if not cand_skills:
            continue

        gt_ids = list(qrels.get(qid, {}).keys())
        flags = allocate(
            allocator,
            [s["id"] for s in cand_skills],
            body_ratio,
            seed=seed + hash(qid) % (2**31),
            gt_ids=gt_ids if allocator == "oracle" else None,
        )

        body_chosen += sum(flags)
        body_total += len(flags)
        doc_texts = [build_skill_text(s, f) for s, f in zip(cand_skills, flags)]
        # Cost accounting (whitespace tokens — relative cost, not absolute).
        tokens_used += sum(approx_token_count(t) for t in doc_texts)
        tokens_full += sum(
            approx_token_count(build_skill_text(s, True)) for s in cand_skills
        )

        start = len(all_pairs)
        all_pairs.extend((item["query"], d) for d in doc_texts)
        pair_map.append((qid, cand_skills, start, len(all_pairs), flags))

    bs = getattr(model, "batch_size", 64) or 64
    print(
        f"[ablation] scoring {len(all_pairs)} pairs "
        f"(body_chosen={body_chosen}/{body_total}={body_chosen / max(body_total,1):.2%}, "
        f"bs={bs})"
    )

    if use_multi:
        sort_idx = sorted(
            range(len(all_pairs)),
            key=lambda i: len(all_pairs[i][0]) + len(all_pairs[i][1]),
        )
        sorted_pairs = [all_pairs[i] for i in sort_idx]
        sorted_scores = model.compute_rank_score_multi(sorted_pairs, batch_size=bs)
        all_scores: list[float] = [0.0] * len(sorted_scores)
        for orig_i, sc in zip(sort_idx, sorted_scores):
            all_scores[orig_i] = sc
    else:
        # Fallback: per-query scoring loop (no cross-query batching).
        all_scores = [0.0] * len(all_pairs)
        qmap = {it["id"]: it["query"] for it in queries}
        for qid, _cand_skills, start, end, _flags in tqdm(
            pair_map, desc="[ablation] per-query scoring"
        ):
            q_text = qmap[qid]
            doc_texts = [all_pairs[i][1] for i in range(start, end)]
            scores = model.compute_rank_score(q_text, doc_texts)
            for i, sc in enumerate(scores):
                all_scores[start + i] = float(sc)

    result: Dict[str, Dict[str, float]] = {}
    for qid, cand_skills, start, end, _flags in pair_map:
        result[qid] = {
            str(s["id"]): float(sc)
            for s, sc in zip(cand_skills, all_scores[start:end])
        }

    metrics = trec_eval(qrels=qrels, results=result)
    collection = {split: metrics}

    config = {
        "reranker": reranker_model,
        "first_stage_file": first_stage_file,
        "allocator": allocator,
        "body_ratio": body_ratio,
        "seed": seed,
        "from_top_k": from_top_k,
        "split": split,
        "body_chosen": body_chosen,
        "body_total": body_total,
        "tokens_used_approx": tokens_used,
        "tokens_full_approx": tokens_full,
        "token_ratio_approx": tokens_used / max(tokens_full, 1),
    }

    if output_file:
        write_json(
            {
                "metrics": collection,
                "retrieval": {split: result},
                "config": config,
            },
            output_file,
        )
        print(f"[ablation] wrote {output_file}")

    return collection


def main() -> None:
    """CLI entry point used by ``scripts/run_part1_ablation.sh``."""
    import argparse

    parser = argparse.ArgumentParser(description="Part 1: rerank body-ratio ablation")
    parser.add_argument("--reranker", required=True)
    parser.add_argument("--first-stage", required=True, dest="first_stage")
    parser.add_argument(
        "--allocator", required=True,
        choices=("random", "rank_top", "rank_bottom", "oracle"),
    )
    parser.add_argument("--ratio", required=True, type=float, dest="body_ratio")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=RERANK_TOP_K, dest="from_top_k")
    parser.add_argument("--batch-size", type=int, default=0, dest="rerank_batch_size")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True, dest="output_file")
    args = parser.parse_args()

    out = Path(args.output_file)
    if out.is_file():
        print(f"[ablation] SKIP (exists): {out}")
        return

    eval_rerank_ablation(
        reranker_model=args.reranker,
        first_stage_file=args.first_stage,
        allocator=args.allocator,
        body_ratio=args.body_ratio,
        seed=args.seed,
        from_top_k=args.from_top_k,
        output_file=args.output_file,
        rerank_batch_size=args.rerank_batch_size,
        split=args.split,
    )


if __name__ == "__main__":
    main()
