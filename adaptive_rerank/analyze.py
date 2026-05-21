"""Aggregate Part 1 ablation results and produce comparison plots.

Scans a directory of result JSONs written by ``eval_ablation.py``, extracts
the ``config`` block and key metrics, dumps a wide CSV, and renders one
``NDCG@10 vs body_ratio`` figure per reranker with one curve per allocator.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

METRICS_OF_INTEREST = (
    "NDCG@5", "NDCG@10", "NDCG@15",
    "Recall@5", "Recall@10", "Recall@15",
    "Completeness@5", "Completeness@10", "Completeness@15",
)


def _iter_results(root: Path) -> Iterable[dict]:
    for fp in sorted(root.glob("*.json")):
        try:
            data = json.loads(fp.read_text())
        except json.JSONDecodeError:
            print(f"[analyze] skipping unparseable: {fp}")
            continue
        cfg = data.get("config", {})
        metrics_by_split = data.get("metrics", {})
        if not metrics_by_split:
            continue
        split = next(iter(metrics_by_split))
        m = metrics_by_split[split]
        row = {
            "file": fp.name,
            "reranker": cfg.get("reranker", ""),
            "first_stage_file": cfg.get("first_stage_file", ""),
            "allocator": cfg.get("allocator", ""),
            "body_ratio": cfg.get("body_ratio", float("nan")),
            "seed": cfg.get("seed", 0),
            "from_top_k": cfg.get("from_top_k", 0),
            "token_ratio_approx": cfg.get("token_ratio_approx", float("nan")),
            "split": split,
        }
        for k in METRICS_OF_INTEREST:
            row[k] = m.get(k, float("nan"))
        yield row


def write_csv(rows: list[dict], out_csv: Path) -> None:
    if not rows:
        print("[analyze] no rows to write")
        return
    cols = list(rows[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[analyze] wrote {len(rows)} rows → {out_csv}")


def plot_ndcg_vs_ratio(rows: list[dict], out_png: Path, metric: str = "NDCG@10") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not installed — skipping plot")
        return
    if not rows:
        print("[analyze] no data to plot")
        return

    by_reranker_allocator: Dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get(metric) is None:
            continue
        by_reranker_allocator[(r["reranker"], r["allocator"])].append(r)

    rerankers = sorted({rk for rk, _ in by_reranker_allocator})
    if not rerankers:
        print("[analyze] no rerankers found")
        return

    fig, axes = plt.subplots(
        1, len(rerankers),
        figsize=(5 * len(rerankers), 4),
        sharey=True,
    )
    if len(rerankers) == 1:
        axes = [axes]

    color_map = {
        "random": "#888888",
        "rank_top": "#1f77b4",
        "rank_bottom": "#d62728",
        "oracle": "#2ca02c",
        "rl": "#9467bd",
    }

    for ax, reranker in zip(axes, rerankers):
        for allocator in sorted({a for rk, a in by_reranker_allocator if rk == reranker}):
            pts = by_reranker_allocator[(reranker, allocator)]
            # Average over seeds.
            by_ratio: Dict[float, list[float]] = defaultdict(list)
            for r in pts:
                by_ratio[float(r["body_ratio"])].append(float(r[metric]))
            xs = sorted(by_ratio)
            ys = [sum(by_ratio[x]) / len(by_ratio[x]) for x in xs]
            ax.plot(
                xs, ys, "o-",
                color=color_map.get(allocator, None),
                label=allocator,
            )
        ax.set_title(reranker.split("/")[-1], fontsize=10)
        ax.set_xlabel("body ratio")
        ax.set_xlim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(metric)
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"[analyze] wrote plot → {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Part 1 ablation results.")
    parser.add_argument(
        "--results-dir", default="adaptive_rerank/results/part1",
        help="Directory containing *.json result files",
    )
    parser.add_argument(
        "--out-csv", default="adaptive_rerank/results/part1_summary.csv",
    )
    parser.add_argument(
        "--out-plot", default="adaptive_rerank/results/part1_ndcg10.png",
    )
    parser.add_argument("--metric", default="NDCG@10")
    args = parser.parse_args()

    rows = list(_iter_results(Path(args.results_dir)))
    write_csv(rows, Path(args.out_csv))
    plot_ndcg_vs_ratio(rows, Path(args.out_plot), metric=args.metric)


if __name__ == "__main__":
    main()
