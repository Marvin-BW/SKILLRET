"""
Stress test to find the maximum per-device batch size for reranker training.

Matches the actual training code path:
  - Uses the Qwen3 chat template for reranker inputs
  - Computes BCE loss on yes/no logits only (not full-vocab cross_entropy)
  - Gradient checkpointing with use_reentrant=False
  - Tests at multiple max_seq_length caps for batch_size vs truncation trade-off

Usage:
    cd /path/to/skillret-benchmark
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python train/reranker-ft/stress_test.py \
        --config train/reranker-ft/configs/qwen3-reranker-0.6b-sft.yaml
"""

import argparse
import gc
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from transformers import AutoModelForCausalLM, AutoTokenizer

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "skillret.config",
    str(Path(__file__).resolve().parents[2] / "skillret/config.py"),
)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
SKILL_RERANK_INSTRUCTION = _cfg.SKILL_RERANK_INSTRUCTION
HF_DATASET_ID = _cfg.HF_DATASET_ID

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
)

YES_TOKEN_ID = 9693
NO_TOKEN_ID = 2152


def _load_hf_dataset(subset: str, split: str = "test") -> list[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load(HF_DATASET_ID, subset, split=split)
    return [dict(row) for row in ds]


def build_skill_text(skill: dict) -> str:
    name = (skill.get("name") or "").strip()
    desc = (skill.get("description") or "").strip()
    body = (skill.get("skill_md") or "").strip()
    return f"{name} | {desc} | {body}"


def format_pair(query: str, doc: str) -> str:
    return (
        f"<Instruct>: {SKILL_RERANK_INSTRUCTION}\n"
        f"<Query>: {query}\n"
        f"<Document>: {doc}"
    )


def build_all_pairs(seed: int = 42) -> list[tuple[str, str]]:
    queries = _load_hf_dataset("queries", split="train")
    skills_raw = _load_hf_dataset("skills", split="train")
    skill_lookup = {s["id"]: build_skill_text(s) for s in skills_raw}
    all_skill_ids = list(skill_lookup.keys())
    rng = random.Random(seed)

    pairs = []
    for q in queries:
        gt_ids = set(q["skill_ids"])
        for sid in q["skill_ids"]:
            text = skill_lookup.get(sid)
            if text is None:
                continue
            pairs.append((q["query"], text))
            neg_id = rng.choice(all_skill_ids)
            while neg_id in gt_ids:
                neg_id = rng.choice(all_skill_ids)
            pairs.append((q["query"], skill_lookup[neg_id]))
    return pairs


def estimate_token_lengths(
    pairs: list[tuple[str, str]],
    tokenizer,
    prefix_len: int,
    suffix_len: int,
    max_length: int,
    sample_n: int = 3000,
) -> np.ndarray:
    rng = random.Random(0)
    sample_idx = rng.sample(range(len(pairs)), min(sample_n, len(pairs)))

    sample_char_lens, sample_tok_lens = [], []
    overhead = prefix_len + suffix_len
    for i in sample_idx:
        text = format_pair(*pairs[i])
        char_len = len(text)
        tok_len = min(
            len(tokenizer.encode(text, add_special_tokens=False)) + overhead,
            max_length,
        )
        sample_char_lens.append(char_len)
        sample_tok_lens.append(tok_len)

    ratio = np.median(np.array(sample_tok_lens) / np.array(sample_char_lens))
    logger.info(f"Char-to-token ratio (median): {ratio:.3f}")

    all_lengths = np.empty(len(pairs), dtype=np.int32)
    sample_map = {idx: j for j, idx in enumerate(sample_idx)}
    for i in range(len(pairs)):
        if i in sample_map:
            all_lengths[i] = sample_tok_lens[sample_map[i]]
        else:
            char_len = len(format_pair(*pairs[i]))
            all_lengths[i] = min(int(char_len * ratio) + overhead, max_length)

    return all_lengths


def build_p99_batch(
    pairs: list[tuple[str, str]],
    lengths: np.ndarray,
    batch_size: int,
    seed: int = 0,
) -> list[tuple[str, str]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))

    batch_maxes, batch_starts = [], []
    for start in range(0, len(idx) - batch_size, batch_size):
        batch_idx = idx[start : start + batch_size]
        batch_maxes.append(lengths[batch_idx].max())
        batch_starts.append(start)

    p99_val = np.percentile(batch_maxes, 99)
    best_i = min(range(len(batch_maxes)), key=lambda i: abs(batch_maxes[i] - p99_val))
    chosen_idx = idx[batch_starts[best_i] : batch_starts[best_i] + batch_size]

    bl = lengths[chosen_idx]
    logger.info(
        f"  P99 batch: max_tok={bl.max()}, "
        f"mean_tok={bl.mean():.0f}, median_tok={int(np.median(bl))}"
    )
    return [pairs[i] for i in chosen_idx]


def try_batch(
    model, optimizer, tokenizer, prefix_tokens, suffix_tokens,
    pairs_batch: list[tuple[str, str]], max_length: int,
    vram_cap_mb: float,
) -> tuple[bool, float, int]:
    """Forward + backward + optimizer.step() matching real training."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    bs = len(pairs_batch)

    try:
        formatted = [format_pair(q, d) for q, d in pairs_batch]
        overhead = len(prefix_tokens) + len(suffix_tokens)
        inputs = tokenizer(
            formatted, padding=False, truncation=True,
            return_attention_mask=False,
            max_length=max_length - overhead,
        )
        for j, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][j] = prefix_tokens + ids + suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]

        labels = torch.randint(0, 2, (bs,), device=model.device, dtype=torch.float32)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**inputs, logits_to_keep=1)
            last_logits = outputs.logits[:, -1, :]
            scores = last_logits[:, YES_TOKEN_ID] - last_logits[:, NO_TOKEN_ID]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
            loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

        fits = peak_mb <= vram_cap_mb
        status = "OK" if fits else "OVER CAP"
        logger.info(
            f"  bs={bs}: {status} "
            f"(seq_len={seq_len}, peak={peak_mb:.0f}/{vram_cap_mb:.0f} MB cap, "
            f"{peak_mb / total_mb * 100:.1f}% of GPU)"
        )
        return fits, peak_mb, seq_len

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            logger.info(f"  bs={bs}: OOM (hard)")
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            return False, 0.0, 0
        raise


def binary_search(model, optimizer, tokenizer, prefix_tokens, suffix_tokens,
                  all_pairs, lengths, max_length, max_bs, vram_cap_mb):
    low, high, best = 1, max_bs, 1
    while low <= high:
        mid = (low + high) // 2
        logger.info(f"Testing bs={mid} [{low}, {high}]...")
        batch = build_p99_batch(all_pairs, lengths, mid)
        ok, _, _ = try_batch(
            model, optimizer, tokenizer, prefix_tokens, suffix_tokens,
            batch, max_length, vram_cap_mb,
        )
        if ok:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Find max batch_size for reranker training"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--max-batch-size", type=int, default=256,
        help="Upper bound for binary search (default: 256)",
    )
    parser.add_argument(
        "--ddp-reserve-gb", type=float, default=30.0,
        help="VRAM to reserve for DDP/Trainer/evaluator overhead (default: 30 GB)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = cfg["model"]
    max_seq_length = cfg.get("max_seq_length", 8192)

    gpu_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    total_mb = total_gb * 1024
    reserve_mb = args.ddp_reserve_gb * 1024
    vram_cap_mb = total_mb - reserve_mb
    logger.info(f"GPU: {gpu_name} ({total_gb:.1f} GB)")
    logger.info(f"DDP/Trainer reserve: {args.ddp_reserve_gb:.0f} GB")
    logger.info(f"Usable VRAM cap: {vram_cap_mb:.0f} MB ({vram_cap_mb / total_mb * 100:.0f}%)")
    logger.info(f"Model: {model_path}")
    logger.info(f"Config max_seq_length: {max_seq_length}")

    logger.info("Building all training pairs from HuggingFace dataset...")
    all_pairs = build_all_pairs()
    logger.info(f"  {len(all_pairs):,} pairs total")

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="left", trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = model.cuda().train()

    lr = cfg.get("lr", 2e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    logger.info(f"AdamW optimizer created (lr={lr})")

    logger.info("Warming up optimizer states...")
    dummy = tokenizer("dummy", return_tensors="pt").to(model.device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(**dummy, logits_to_keep=1)
        out.logits.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    opt_mem_mb = sum(
        s.element_size() * s.nelement()
        for pg in optimizer.param_groups
        for p in pg["params"]
        if p in optimizer.state
        for s in optimizer.state[p].values()
        if isinstance(s, torch.Tensor)
    ) / (1024 ** 2)
    logger.info(f"Optimizer state memory: {opt_mem_mb:.0f} MB")

    assert tokenizer.convert_tokens_to_ids("yes") == YES_TOKEN_ID
    assert tokenizer.convert_tokens_to_ids("no") == NO_TOKEN_ID

    prefix = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    overhead = len(prefix_tokens) + len(suffix_tokens)
    logger.info(f"Template overhead: {overhead} tokens")

    seq_caps = sorted(
        {cap for cap in [2048, 4096, 6144, max_seq_length] if cap <= max_seq_length},
        reverse=True,
    )

    results = []
    for cap in seq_caps:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"=== Testing max_seq_length = {cap} ===")

        logger.info("Estimating token lengths...")
        lengths = estimate_token_lengths(
            all_pairs, tokenizer, len(prefix_tokens), len(suffix_tokens), cap,
        )
        n_at_cap = int((lengths >= cap).sum())
        pct_trunc = n_at_cap / len(lengths) * 100
        logger.info(
            f"  mean={lengths.mean():.0f}, median={np.median(lengths):.0f}, "
            f"P95={np.percentile(lengths, 95):.0f}, "
            f"P99={np.percentile(lengths, 99):.0f}, max={lengths.max()}, "
            f"at_cap={n_at_cap} ({pct_trunc:.1f}%)"
        )

        best = binary_search(
            model, optimizer, tokenizer, prefix_tokens, suffix_tokens,
            all_pairs, lengths, cap, args.max_batch_size, vram_cap_mb,
        )
        results.append((cap, best, pct_trunc))
        logger.info(f"  >>> max_seq_length={cap}: max_bs={best}")

    logger.info(f"\n{'=' * 60}")
    logger.info(
        f"RESULTS (optimizer included, {args.ddp_reserve_gb:.0f} GB reserved for DDP/Trainer)"
    )
    logger.info(f"{'=' * 60}")
    logger.info(
        f"{'max_seq':>10} | {'max_bs':>8} | {'eff_bs(8gpu)':>14} | {'truncated':>10}"
    )
    logger.info(f"{'-' * 10}-+-{'-' * 8}-+-{'-' * 14}-+-{'-' * 10}")
    for cap, bs, trunc in results:
        logger.info(
            f"{cap:>10} | {bs:>8} | {bs * 8:>14} | {trunc:>9.1f}%"
        )
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
