"""Plug the learned policy into the end-to-end rerank evaluation pipeline.

This module mirrors ``adaptive_rerank.eval_ablation`` but instead of using
a fixed allocator, it calls the trained ``PolicyMLP`` per-query to decide
which candidates get the body. It writes a JSON in the same shape as the
Part 1 results so that ``analyze.py`` can plot the RL policy as a new
``allocator="rl"`` series on the same figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

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

from .features import extract_features_for_query
from .network import PolicyMLP
from ..text_builder import build_skill_text, approx_token_count


def _first_stage(first_stage_file: str, split: str) -> Dict[str, Dict[str, float]]:
    raw = load_json(first_stage_file)
    ret = raw.get("retrieval", raw)
    if split in ret:
        return ret[split]
    return next(iter(ret.values())) if ret else {}


def eval_adaptive(
    reranker_model: str,
    first_stage_file: str,
    policy_ckpt: str,
    *,
    threshold: float = 0.5,
    from_top_k: int = RERANK_TOP_K,
    output_file: str | None = None,
    rerank_batch_size: int = 0,
    split: str = "test",
    device_str: str | None = None,
) -> Dict[str, Dict[str, float]]:
    device = _require_rerank_device()
    print(f"[adaptive] rerank device: {device}")
    if rerank_batch_size <= 0:
        rerank_batch_size = get_batch_size(reranker_model, "rerank")
    model = _load_reranker(reranker_model, device, rerank_batch_size=rerank_batch_size)

    skills = load_corpus(split=split)
    skill_map = {str(s["id"]): s for s in skills}
    queries = load_queries(split=split)

    ckpt = torch.load(policy_ckpt, map_location="cpu")
    mean = ckpt["feature_mean"]
    std = ckpt["feature_std"]
    cfg = ckpt.get("config", {})
    policy = PolicyMLP(hidden=cfg.get("hidden", 128), n_layers=cfg.get("n_layers", 2))
    policy.load_state_dict(ckpt["state_dict"])
    policy_device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy.to(policy_device).eval()
    print(f"[adaptive] policy loaded from {policy_ckpt}")

    first_stage = _first_stage(first_stage_file, split)

    qrels: Dict[str, Dict[str, int]] = {}
    for item in queries:
        labels = _normalize_query_labels(item)
        qrels[item["id"]] = {str(x["id"]): int(x["relevance"]) for x in labels}

    all_pairs: list[tuple[str, str]] = []
    pair_map = []
    body_chosen = body_total = 0
    tokens_used = tokens_full = 0

    for item in tqdm(queries, desc="[adaptive] policy decisions"):
        qid = item["id"]
        cand_scores = first_stage.get(qid, {})
        sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:from_top_k]
        cand_skills = [skill_map[sid] for sid, _ in sorted_cands if sid in skill_map]
        fs_scores = [sc for sid, sc in sorted_cands if sid in skill_map]
        if not cand_skills:
            continue
        feats = extract_features_for_query(item["query"], cand_skills, fs_scores)
        feats = (feats - mean) / std
        with torch.no_grad():
            xt = torch.from_numpy(feats).to(policy_device)
            probs = policy.action_prob(xt).cpu().numpy()
        actions = (probs >= threshold).astype(np.int64)

        flags = actions.astype(bool).tolist()
        body_chosen += int(actions.sum())
        body_total += len(flags)
        doc_texts = [build_skill_text(s, bool(f)) for s, f in zip(cand_skills, flags)]
        tokens_used += sum(approx_token_count(t) for t in doc_texts)
        tokens_full += sum(approx_token_count(build_skill_text(s, True)) for s in cand_skills)

        start = len(all_pairs)
        all_pairs.extend((item["query"], d) for d in doc_texts)
        pair_map.append((qid, cand_skills, start, len(all_pairs)))

    use_multi = hasattr(model, "compute_rank_score_multi")
    bs = getattr(model, "batch_size", 64) or 64
    if use_multi:
        order = sorted(range(len(all_pairs)),
                       key=lambda i: len(all_pairs[i][0]) + len(all_pairs[i][1]))
        sorted_pairs = [all_pairs[i] for i in order]
        sorted_scores = model.compute_rank_score_multi(sorted_pairs, batch_size=bs)
        all_scores = [0.0] * len(sorted_scores)
        for orig, sc in zip(order, sorted_scores):
            all_scores[orig] = sc
    else:
        all_scores = [0.0] * len(all_pairs)
        qmap = {it["id"]: it["query"] for it in queries}
        for qid, _cands, start, end in tqdm(pair_map, desc="[adaptive] scoring"):
            q_text = qmap[qid]
            doc_texts = [all_pairs[i][1] for i in range(start, end)]
            scores = model.compute_rank_score(q_text, doc_texts)
            for i, sc in enumerate(scores):
                all_scores[start + i] = float(sc)

    result: Dict[str, Dict[str, float]] = {}
    for qid, cand_skills, start, end in pair_map:
        result[qid] = {
            str(s["id"]): float(sc)
            for s, sc in zip(cand_skills, all_scores[start:end])
        }

    metrics = trec_eval(qrels=qrels, results=result)
    collection = {split: metrics}

    config = {
        "reranker": reranker_model,
        "first_stage_file": first_stage_file,
        "policy_ckpt": policy_ckpt,
        "threshold": threshold,
        "from_top_k": from_top_k,
        "split": split,
        "allocator": "rl",
        "body_ratio": body_chosen / max(body_total, 1),
        "body_chosen": body_chosen,
        "body_total": body_total,
        "tokens_used_approx": tokens_used,
        "tokens_full_approx": tokens_full,
        "token_ratio_approx": tokens_used / max(tokens_full, 1),
    }

    if output_file:
        write_json({"metrics": collection,
                    "retrieval": {split: result},
                    "config": config}, output_file)
        print(f"[adaptive] wrote {output_file}")
    return collection


def main() -> None:
    p = argparse.ArgumentParser(description="Adaptive (RL) rerank evaluation.")
    p.add_argument("--reranker", required=True)
    p.add_argument("--first-stage", required=True)
    p.add_argument("--policy-ckpt", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=RERANK_TOP_K, dest="from_top_k")
    p.add_argument("--batch-size", type=int, default=0, dest="rerank_batch_size")
    p.add_argument("--split", default="test")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out = Path(args.output)
    if out.is_file():
        print(f"[adaptive] SKIP (exists): {out}")
        return
    eval_adaptive(
        reranker_model=args.reranker,
        first_stage_file=args.first_stage,
        policy_ckpt=args.policy_ckpt,
        threshold=args.threshold,
        from_top_k=args.from_top_k,
        output_file=args.output,
        rerank_batch_size=args.rerank_batch_size,
        split=args.split,
    )


if __name__ == "__main__":
    main()
