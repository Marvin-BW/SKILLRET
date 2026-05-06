"""
Fine-tune Qwen3-Embedding-8B on SkillRet training data (4 GPU DDP).

Usage:
    cd /path/to/project
    source .venv/bin/activate
    torchrun --nproc_per_node=4 train/4gpu-qwen3-8b/train.py
"""

import logging
import os
from pathlib import Path

os.environ["WANDB_PROJECT"] = "skillret"

from datasets import Dataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "skillret.config",
    str(Path(__file__).resolve().parents[2] / "skillret/config.py"),
)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
SKILL_QUERY_PROMPT = _cfg.SKILL_QUERY_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_MODEL = "/path/to/models/Qwen3-Embedding-8B"  # TODO: set your model path
OUTPUT_DIR = "./outputs"  # TODO: set your output path
HF_DATASET = "ThakiCloud/SKILLRET"

# ── Hyperparameters (4x B200, DDP+GC, effective_batch=80, max mode) ───────
# Total steps ≈ 127,190 / 80 ≈ 1590 steps (1 epoch)
MAX_SEQ_LENGTH = 8192
BATCH_SIZE = 20         # per device → 20 × 4 GPU = 80
GRAD_ACCUM = 1
EPOCHS = 1
LR = 2e-5
WARMUP_RATIO = 0.1
EVAL_STEPS = 200        # 1590 steps → evals at 200, 400, 600, ... (7 evals)
SAVE_STEPS = 200
BF16 = True


def build_skill_text(skill: dict) -> str:
    name = (skill.get("name") or "").strip()
    desc = (skill.get("description") or "").strip()
    body = (skill.get("skill_md") or "").strip()
    return f"{name} | {desc} | {body}"


def load_hf_skills(split: str) -> dict[str, str]:
    ds = load_dataset(HF_DATASET, "skills", split=split)
    return {row["id"]: build_skill_text(row) for row in ds}


def load_hf_queries(split: str) -> list[dict]:
    ds = load_dataset(HF_DATASET, "queries", split=split)
    return [dict(row) for row in ds]


def build_train_dataset(queries: list[dict], skill_lookup: dict[str, str]) -> Dataset:
    anchors, positives, skipped = [], [], 0

    for q in queries:
        for sid in q["skill_ids"]:
            skill_text = skill_lookup.get(sid)
            if skill_text is None:
                skipped += 1
                continue
            anchors.append(q["query"])
            positives.append(skill_text)

    logger.info(f"Training pairs: {len(anchors):,} (from {len(queries):,} queries, skipped {skipped})")
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def build_evaluator(queries: list[dict], skill_lookup: dict[str, str]) -> InformationRetrievalEvaluator:
    q_dict = {q["id"]: q["query"] for q in queries}
    corpus = skill_lookup
    relevant_docs = {q["id"]: set(q["skill_ids"]) for q in queries}

    return InformationRetrievalEvaluator(
        queries=q_dict,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="v3-benchmark",
        ndcg_at_k=[5, 10, 15],
        map_at_k=[5, 10, 15],
        precision_recall_at_k=[5, 10, 15],
        mrr_at_k=[5, 10, 15],
        show_progress_bar=True,
        batch_size=32,
        query_prompt=SKILL_QUERY_PROMPT,
    )


def main():
    logger.info("Loading model...")
    model = SentenceTransformer(BASE_MODEL, trust_remote_code=True)
    model.max_seq_length = MAX_SEQ_LENGTH
    logger.info(f"  dim={model.get_sentence_embedding_dimension()}, max_seq={model.max_seq_length}")

    logger.info(f"Loading data from HuggingFace: {HF_DATASET}")

    logger.info("Loading training skill corpus...")
    train_skill_lookup = load_hf_skills("train")
    logger.info(f"  {len(train_skill_lookup):,} skills loaded")

    logger.info("Loading training queries...")
    train_queries = load_hf_queries("train")

    logger.info("Building training dataset...")
    train_dataset = build_train_dataset(train_queries, train_skill_lookup)

    logger.info("Loading evaluation data...")
    eval_skill_lookup = load_hf_skills("test")
    eval_queries = load_hf_queries("test")

    logger.info("Building evaluator (v3 benchmark)...")
    evaluator = build_evaluator(eval_queries, eval_skill_lookup)

    loss = MultipleNegativesRankingLoss(model, gather_across_devices=True)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        fp16=False,
        bf16=BF16,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="v3-benchmark_cosine_ndcg@10",
        greater_is_better=True,
        dataloader_num_workers=4,
        report_to="wandb",
        run_name="qwen3-8b-4gpu-b20-gc-skill",
        prompts={"anchor": SKILL_QUERY_PROMPT},
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=None,
        loss=loss,
        evaluator=evaluator,
    )

    logger.info(f"Starting training (4 GPU DDP, effective_batch={BATCH_SIZE * 4 * GRAD_ACCUM})...")
    trainer.train()

    logger.info("Saving final model...")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    logger.info(f"Done! Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
