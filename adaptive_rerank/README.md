# adaptive_rerank

Additional experiments on top of the SkillRet benchmark. **Nothing under `skillret/` is modified** — this package only imports public/private helpers from the frozen `skillret.eval` module and adds its own pipelines.

There are two parts:

| | What it answers |
|---|---|
| **Part 1** (ablation) | If we give the body of a skill to the reranker for only *some* candidates in the top-K, how does the metric change as a function of (a) the body ratio and (b) how we pick which candidates? |
| **Part 2** (RL policy) | Train a tiny per-(query, candidate) policy that decides at inference time whether each candidate should be fed body or only metadata, without touching the retriever or reranker weights. |

## Directory layout

```
adaptive_rerank/
├── __init__.py
├── text_builder.py           # build_skill_text(skill, use_body) — drop-in replacement for
│                             #   skillret.eval._rerank_skill_text but with a body switch
├── allocators.py             # random / rank_top / rank_bottom / oracle
├── eval_ablation.py          # Part 1 evaluation entry (CLI + import)
├── analyze.py                # aggregate Part 1 results → CSV + matplotlib plot
├── policy/
│   ├── features.py           # 10-dim per-(q, c) feature vector
│   ├── offline_cache.py      # pre-compute reranker scores for both text variants
│   ├── network.py            # MLP policy + NDCG helper + greedy oracle actions
│   ├── train_rl.py           # REINFORCE with moving-avg baseline
│   ├── train_sup.py          # supervised oracle imitation (upper-bound check)
│   └── eval_adaptive.py      # plug the learned policy into rerank evaluation
├── scripts/
│   ├── run_part1_ablation.sh
│   ├── run_part2_cache.sh
│   ├── run_part2_train.sh
│   └── run_part2_eval.sh
└── results/                  # outputs land here
    ├── part1/                # Part 1 JSONs (also where RL policy eval lands so analyze.py picks it up)
    └── part2/                # offline cache + policy checkpoints
```

## Prerequisites

1. The base SkillRet environment must be set up first (see `../README.md`):
   ```bash
   uv sync
   source .venv/bin/activate
   ```
2. You need at least one first-stage retrieval JSON, e.g.
   `results/embed/ThakiCloud_SKILLRET-Embedding-0.6B.json`. If you don't have it yet:
   ```bash
   NUM_GPUS=1 MODELS_FILTER="SKILLRET-Embedding-0.6B" bash scripts/run_eval_embedding.sh
   ```

## Part 1 — body-ratio ablation

Sweep the grid `reranker × allocator × body_ratio × seed`. By default it runs three rerankers (jina-reranker-v2, Qwen3-Reranker-0.6B, SkillRet-Reranker-0.6B) across body ratios `{0, 0.25, 0.5, 0.75, 1}` with four allocators.

```bash
# All defaults, single GPU
bash adaptive_rerank/scripts/run_part1_ablation.sh

# Filter to one reranker, more seeds for the random allocator
RERANKERS="Qwen/Qwen3-Reranker-0.6B" SEEDS="0,1,2,3,4" \
    bash adaptive_rerank/scripts/run_part1_ablation.sh

# Use a different first-stage retrieval JSON
FIRST_STAGE=results/embed/Qwen_Qwen3-Embedding-0.6B.json \
    bash adaptive_rerank/scripts/run_part1_ablation.sh
```

Each run writes `adaptive_rerank/results/part1/rerank_<reranker>_<firstStage>_<allocator>_p<ratio>_s<seed>.json` containing the same `metrics` / `retrieval` blocks as `skillret.eval.eval_rerank`, **plus** a `config` block recording the ablation parameters (including approximate token counts so you can plot cost-vs-quality).

**Aggregate and plot**:

```bash
python -m adaptive_rerank.analyze \
    --results-dir adaptive_rerank/results/part1 \
    --out-csv     adaptive_rerank/results/part1_summary.csv \
    --out-plot    adaptive_rerank/results/part1_ndcg10.png \
    --metric NDCG@10
```

Reads every `*.json` in the directory, dumps a wide CSV (one row per result file), and renders one subplot per reranker with one line per allocator (random/rank_top/rank_bottom/oracle), x = body ratio, y = chosen metric. RL-policy results from Part 2 are emitted into the same directory and automatically get an `rl` series on the same figure.

### Programmatic use

```python
from adaptive_rerank.eval_ablation import eval_rerank_ablation
result = eval_rerank_ablation(
    reranker_model="Qwen/Qwen3-Reranker-0.6B",
    first_stage_file="results/embed/ThakiCloud_SKILLRET-Embedding-0.6B.json",
    allocator="oracle",
    body_ratio=0.5,
    output_file="adaptive_rerank/results/part1/oracle_p50.json",
)
```

## Part 2 — RL-based adaptive body inclusion

The policy is a per-(query, candidate) contextual bandit with two actions: `metadata` vs `full body`. Reward is `NDCG@10 − λ · token_ratio`. Training is fully offline once the reranker double-scoring cache is built.

### Step 1 — build the offline cache (one-time, GPU-heavy)

For every (query, candidate) in the 10 % train subset, run the **frozen** reranker once with metadata-only doc-text and once with full-body doc-text, and cache both scores plus the 10-dim feature vector.

```bash
RERANKER=Qwen/Qwen3-Reranker-0.6B \
EMBED_MODEL=ThakiCloud/SKILLRET-Embedding-0.6B \
SUBSET_RATIO=0.1 \
    bash adaptive_rerank/scripts/run_part2_cache.sh
# → adaptive_rerank/results/part2/cache.train0.1.pkl.gz
```

If you already have a first-stage retrieval JSON for the **train** split, point `FIRST_STAGE=` at it to skip the embedding pass.

Cost rough estimate: `0.1 × 63 259 queries × 20 candidates × 2 forward passes` per reranker. Scale `SUBSET_RATIO` down for a smoke test.

### Step 2 — train policies (CPU-fine, no reranker calls)

```bash
LAM="0.0,0.05,0.1,0.2,0.5" \
    bash adaptive_rerank/scripts/run_part2_train.sh
# → adaptive_rerank/results/part2/ckpts/policy_sup.pt
# → adaptive_rerank/results/part2/ckpts/policy_rl_lam{0.0,0.05,0.1,0.2,0.5}.pt
```

Both REINFORCE and the supervised oracle baseline are trained; each prints val NDCG@10 and val token-ratio per epoch, plus the two reference baselines (all-metadata, all-full-body) at startup so you can sanity-check progress immediately.

### Step 3 — evaluate on the test split (frozen retriever + frozen reranker + learned policy)

```bash
bash adaptive_rerank/scripts/run_part2_eval.sh
# → adaptive_rerank/results/part1/rerank_..._rl_policy_*.json
```

Output JSONs go into the same `part1/` directory so the next `analyze.py` run will plot the policy curve next to the allocator curves.

Re-run `python -m adaptive_rerank.analyze ...` to refresh the plot.

## Recommended workflow

1. Run Part 1 first. Inspect the `oracle` curve at 25/50 %: if it's nearly flat against the 100 % ceiling, there is room for the RL policy to recover most of the body benefit with much less context.
2. If headroom exists, run Part 2 starting with the smallest `SUBSET_RATIO` you can stand (e.g. `0.02`) to make sure the pipeline runs end-to-end.
3. Then re-cache at 0.1 (or higher) and train multiple `LAM` values to sweep the Pareto curve.

## Notes / known limitations

- `text_builder.build_skill_text` mirrors the original `_rerank_skill_text` format `name | description | skill_md`; the metadata-only variant simply drops the trailing `| skill_md`.
- Token costs reported in `config.token_*_approx` use a whitespace split, not the reranker's actual tokenizer — adequate for *relative* comparisons, not for absolute budgeting. Swap in tiktoken `cl100k_base` if you need absolute numbers.
- The "oracle" allocator in Part 1 is GT-priority then rank-top filler; this is an upper-bound *sketch*, not the brute-force optimum. The supervised oracle in `network.oracle_actions` does a greedy search over candidate flips and is the tighter ceiling.
- The RL training loop assumes the cache's `scores_meta` / `scores_body` columns were produced by the same reranker you intend to evaluate with. If you swap rerankers between cache build and eval, rebuild the cache.

## Cleanup

If you see `adaptive_rerank/results/_smoke*` files, they are leftover fixtures from a one-time smoke test of `analyze.py` (couldn't be removed from the sandbox due to permission limits). Safe to delete:

```bash
rm -rf adaptive_rerank/results/_smoke*
```
