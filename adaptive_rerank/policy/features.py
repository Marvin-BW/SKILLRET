"""Feature extraction for the per-(query, candidate) RL policy.

Features are designed to be cheap — none of them require running the
reranker. The intuition is that the policy should learn from signals that
correlate with "body helps" without ever paying the reranker forward cost
twice.

Feature vector layout (10 dims):
    0: first-stage similarity score (raw)
    1: first-stage rank within top-K (normalized to [0, 1])
    2: BM25 score on metadata
    3: BM25 score on body (skill_md)
    4: BM25(body) - BM25(metadata)
    5: 3-gram overlap ratio between query and metadata
    6: 3-gram overlap ratio between query and body
    7: log(1 + query token count)  (whitespace-token proxy)
    8: log(1 + metadata token count)
    9: log(1 + body token count)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np


FEATURE_NAMES = (
    "first_stage_score",
    "first_stage_rank_norm",
    "bm25_metadata",
    "bm25_body",
    "bm25_body_minus_meta",
    "trigram_overlap_metadata",
    "trigram_overlap_body",
    "log_query_len",
    "log_metadata_len",
    "log_body_len",
)
FEATURE_DIM = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Cheap BM25 over a small candidate set (no global IDF — we treat top-K as a
# tiny corpus and compute IDF on it; this is sufficient as a *relative*
# signal between metadata and body for the same candidate set).
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


def _ngrams(tokens: list[str], n: int = 3) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def trigram_overlap_ratio(query: str, doc: str) -> float:
    qg = _ngrams(_tokenize(query))
    dg = _ngrams(_tokenize(doc))
    if not qg:
        return 0.0
    inter = sum((qg & dg).values())
    return inter / max(sum(qg.values()), 1)


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    doc_freqs: dict[str, int],
    n_docs: int,
    avgdl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not doc_tokens:
        return 0.0
    dl = len(doc_tokens)
    tf = Counter(doc_tokens)
    score = 0.0
    for term in query_tokens:
        if term not in doc_freqs:
            continue
        df = doc_freqs[term]
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        f = tf.get(term, 0)
        denom = f + k1 * (1 - b + b * dl / max(avgdl, 1e-6))
        score += idf * (f * (k1 + 1)) / max(denom, 1e-6)
    return score


class CandidateBM25:
    """Mini BM25 index over the top-K candidate set for one query.

    Two indices are maintained: one over metadata strings, one over body
    strings. Both share the same candidate ordering.
    """

    def __init__(self, metadata_texts: list[str], body_texts: list[str]):
        self.meta_tokens = [_tokenize(t) for t in metadata_texts]
        self.body_tokens = [_tokenize(t) for t in body_texts]
        self.meta_df = _doc_freqs(self.meta_tokens)
        self.body_df = _doc_freqs(self.body_tokens)
        self.meta_avgdl = (
            sum(len(t) for t in self.meta_tokens) / max(len(self.meta_tokens), 1)
        )
        self.body_avgdl = (
            sum(len(t) for t in self.body_tokens) / max(len(self.body_tokens), 1)
        )

    def score_meta(self, query_tokens: list[str], i: int) -> float:
        return _bm25_score(
            query_tokens, self.meta_tokens[i], self.meta_df,
            len(self.meta_tokens), self.meta_avgdl,
        )

    def score_body(self, query_tokens: list[str], i: int) -> float:
        return _bm25_score(
            query_tokens, self.body_tokens[i], self.body_df,
            len(self.body_tokens), self.body_avgdl,
        )


def _doc_freqs(token_lists: list[list[str]]) -> dict[str, int]:
    df: dict[str, int] = {}
    for toks in token_lists:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1
    return df


# ---------------------------------------------------------------------------
# Top-level feature extraction
# ---------------------------------------------------------------------------

def extract_features_for_query(
    query: str,
    candidates: list[dict],
    first_stage_scores: list[float],
) -> np.ndarray:
    """Build an ``(N, FEATURE_DIM)`` matrix for one query's top-K candidates.

    Args:
        query: raw query text.
        candidates: list of skill dicts in first-stage rank order (best first).
        first_stage_scores: parallel list of raw first-stage scores.

    Returns:
        ``np.ndarray`` shape ``(len(candidates), FEATURE_DIM)``.
    """
    n = len(candidates)
    meta_texts = [
        f"{(c.get('name') or '').strip()} | {(c.get('description') or '').strip()}"
        for c in candidates
    ]
    body_texts = [(c.get("skill_md") or "").strip() for c in candidates]

    bm25 = CandidateBM25(meta_texts, body_texts)
    q_tokens = _tokenize(query)
    q_len = len(q_tokens)

    feats = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    for i in range(n):
        bm25_m = bm25.score_meta(q_tokens, i)
        bm25_b = bm25.score_body(q_tokens, i)
        feats[i, 0] = float(first_stage_scores[i])
        feats[i, 1] = i / max(n - 1, 1)
        feats[i, 2] = bm25_m
        feats[i, 3] = bm25_b
        feats[i, 4] = bm25_b - bm25_m
        feats[i, 5] = trigram_overlap_ratio(query, meta_texts[i])
        feats[i, 6] = trigram_overlap_ratio(query, body_texts[i])
        feats[i, 7] = math.log1p(q_len)
        feats[i, 8] = math.log1p(len(_tokenize(meta_texts[i])))
        feats[i, 9] = math.log1p(len(_tokenize(body_texts[i])))
    return feats
