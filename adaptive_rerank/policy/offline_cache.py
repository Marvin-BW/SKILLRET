"""Offline pre-computation for the RL policy training.

Builds a static dataset where for every (query, candidate) in the
first-stage top-K we cache:

* the reranker score when fed *metadata only*
* the reranker score when fed *full body*
* the cheap feature vector
* the GT label (1 if the candidate is a relevant skill for the query, else 0)
* an approximate token cost for each text variant

Once this cache exists, ``train_rl.py`` and ``train_sup.py`` can train the
policy without invoking the reranker even once.

The default first-stage retriever is ``SkillRet-Embedding-0.6B``; pass
``--first-stage`` to use a different one. The script uses ``skillret.eval``
helpers without modifying them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from skillret.eval import (
    RetModel,
    _load_reranker,
    _normalize_query_labels,
    _require_rerank_device,
    load_corpus,
    load_queries,
)
from skillret.config import RERANK_TOP_K, get_batch_size

from .features import FEATURE_DIM, extract_features_for_query
from ..text_builder import build_skill_text, approx_token_count


def _first_stage_top_k(
    embed_model_path: str,
    queries: list[dict],
    skills: list[dict],
    top_k: int,
    batch_size: int = 0,
) -> Dict[str, list[tuple[str, float]]]:
    """Run first-stage retrieval and return top-K candidates per query."""
    import faiss
    if batch_size <= 0:
        batch_size = get_batch_size(embed_model_path, "embed")
    print(f"[cache] first-stage: {embed_model_path} (bs={batch_size}, k={top_k})")
    model = RetModel(embed_model_path)
    corpus_emb = model.encode_corpus(skills, batch_size)
    dim = int(corpus_emb.shape[1])
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(corpus_emb)
    q_emb = model.encode_queries(queries, batch_size)
    dists, idxs = index.search(q_emb, top_k)
    out: Dict[str, list[tuple[str, float]]] = {}
    for q, ranks, ds in zip(queries, idxs, dists):
        out[q["id"]] = [
            (str(skills[int(r)]["id"]), float(d))
            for r, d in zip(ranks, ds) if r >= 0
        ]
    return out


def _first_stage_from_file(
    first_stage_file: str,
    split: str,
) -> Dict[str, list[tuple[str, float]]]:
    from skillret.utils import load_json
    raw = load_json(first_stage_file)
    retrieval = raw.get("retrieval", raw)
    if split in retrieval:
        per_q = retrieval[split]
    else:
        per_q = next(iter(retrieval.values()))
    out: Dict[str, list[tuple[str, float]]] = {}
    for qid, scores in per_q.items():
        out[qid] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return out


def build_cache(
    out_path: str,
    *,
    reranker_model: str,
    embed_model_path: str | None,
    first_stage_file: str | None,
    split: str = "train",
    subset_ratio: float = 0.1,
    top_k: int = RERANK_TOP_K,
    seed: int = 0,
    rerank_batch_size: int = 0,
) -> None:
    """Build and pickle the offline cache.

    Either ``embed_model_path`` (re-run first stage) or ``first_stage_file``
    (load existing results) must be provided.
    """
    rng = random.Random(seed)
    skills = load_corpus(split=split)
    skill_map = {str(s["id"]): s for s in skills}
    queries = load_queries(split=split)
    print(f"[cache] loaded {len(skills)} skills, {len(queries)} queries ({split})")

    if 0 < subset_ratio < 1.0:
        n = max(1, int(round(len(queries) * subset_ratio)))
        rng.shuffle(queries)
        queries = queries[:n]
        print(f"[cache] subset {subset_ratio:.0%} → {len(queries)} queries")

    # Ground truth.
    qrels: Dict[str, set[str]] = {}
    for item in queries:
        labels = _normalize_query_labels(item)
        qrels[item["id"]] = {
            str(x["id"]) for x in labels if int(x.get("relevance", 0)) > 0
        }

    # First-stage candidates.
    if first_stage_file:
        first_stage = _first_stage_from_file(first_stage_file, split)
        first_stage = {
            qid: first_stage[qid][:top_k]
            for qid in (q["id"] for q in queries)
            if qid in first_stage
        }
    else:
        if embed_model_path is None:
            raise ValueError("Provide either --first-stage or --embed-model")
        first_stage = _first_stage_top_k(
            embed_model_path, queries, skills, top_k
        )

    # Load reranker once.
    device = _require_rerank_device()
    if rerank_batch_size <= 0:
        rerank_batch_size = get_batch_size(reranker_model, "rerank")
    print(f"[cache] reranker: {reranker_model} (bs={rerank_batch_size})")
    rerank = _load_reranker(reranker_model, device, rerank_batch_size=rerank_batch_size)
    use_multi = hasattr(rerank, "compute_rank_score_multi")
    bs = getattr(rerank, "batch_size", 64) or 64

    # Build the two pair sets (metadata-only and full-body) sharing layout.
    pair_meta: list[tuple[str, str]] = []
    pair_body: list[tuple[str, str]] = []
    per_query: list[dict] = []

    for item in tqdm(queries, desc="[cache] feature extraction"):
        qid = item["id"]
        cand = first_stage.get(qid, [])[:top_k]
        cand_skills = [skill_map[sid] for sid, _ in cand if sid in skill_map]
        cand_scores = [sc for sid, sc in cand if sid in skill_map]
        if not cand_skills:
            continue
        feats = extract_features_for_query(item["query"], cand_skills, cand_scores)
        meta_texts = [build_skill_text(s, False) for s in cand_skills]
        body_texts = [build_skill_text(s, True) for s in cand_skills]
        meta_tokens = [approx_token_count(t) for t in meta_texts]
        body_tokens = [approx_token_count(t) for t in body_texts]
        gt = qrels.get(qid, set())
        labels = np.array(
            [1 if str(s["id"]) in gt else 0 for s in cand_skills], dtype=np.int8,
        )
        start_meta = len(pair_meta)
        pair_meta.extend((item["query"], t) for t in meta_texts)
        pair_body.extend((item["query"], t) for t in body_texts)
        per_query.append({
            "qid": qid,
            "candidate_ids": [str(s["id"]) for s in cand_skills],
            "first_stage_scores": np.asarray(cand_scores, dtype=np.float32),
            "features": feats,
            "labels": labels,
            "meta_tokens": np.asarray(meta_tokens, dtype=np.int32),
            "body_tokens": np.asarray(body_tokens, dtype=np.int32),
            "pair_start": start_meta,
            "pair_end": start_meta + len(cand_skills),
        })

    # Score both pair lists.
    def _score(pairs: list[tuple[str, str]], desc: str) -> list[float]:
        if not pairs:
            return []
        if use_multi:
            order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
            sorted_pairs = [pairs[i] for i in order]
            sorted_scores = rerank.compute_rank_score_multi(sorted_pairs, batch_size=bs)
            scores = [0.0] * len(pairs)
            for orig, sc in zip(order, sorted_scores):
                scores[orig] = float(sc)
            return scores
        # Fallback: per-query.
        scores = []
        for q, d in tqdm(pairs, desc=desc):
            scores.extend(rerank.compute_rank_score(q, [d]))
        return [float(x) for x in scores]

    print("[cache] scoring metadata-only pairs")
    scores_meta = _score(pair_meta, "[cache] meta")
    print("[cache] scoring full-body pairs")
    scores_body = _score(pair_body, "[cache] body")

    # Attach scores back to per-query records.
    sm = np.asarray(scores_meta, dtype=np.float32)
    sb = np.asarray(scores_body, dtype=np.float32)
    for rec in per_query:
        rec["scores_meta"] = sm[rec["pair_start"] : rec["pair_end"]]
        rec["scores_body"] = sb[rec["pair_start"] : rec["pair_end"]]
        del rec["pair_start"], rec["pair_end"]

    payload = {
        "feature_dim": FEATURE_DIM,
        "top_k": top_k,
        "split": split,
        "subset_ratio": subset_ratio,
        "reranker_model": reranker_model,
        "first_stage_file": first_stage_file,
        "embed_model_path": embed_model_path,
        "n_queries": len(per_query),
        "records": per_query,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wb") as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[cache] wrote {out} ({len(per_query)} queries)")


def load_cache(path: str) -> dict:
    with gzip.open(path, "rb") as fp:
        return pickle.load(fp)


def main() -> None:
    p = argparse.ArgumentParser(description="Build offline cache for RL policy training.")
    p.add_argument("--out", required=True)
    p.add_argument("--reranker", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--embed-model", default=None,
                   help="HF id of first-stage embedder; runs retrieval fresh")
    g.add_argument("--first-stage", default=None,
                   help="Path to a results/embed/*.json file")
    p.add_argument("--split", default="train")
    p.add_argument("--subset-ratio", type=float, default=0.1)
    p.add_argument("--top-k", type=int, default=RERANK_TOP_K)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rerank-batch-size", type=int, default=0)
    args = p.parse_args()

    build_cache(
        out_path=args.out,
        reranker_model=args.reranker,
        embed_model_path=args.embed_model,
        first_stage_file=args.first_stage,
        split=args.split,
        subset_ratio=args.subset_ratio,
        top_k=args.top_k,
        seed=args.seed,
        rerank_batch_size=args.rerank_batch_size,
    )


if __name__ == "__main__":
    main()
