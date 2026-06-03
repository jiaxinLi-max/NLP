"""Plotting helpers: accuracy-vs-keep_ratio and KV-MB-vs-keep_ratio."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


STRATEGY_STYLE = {
    "none":                 dict(color="black",   marker="x", linestyle="--"),
    "token_recent_distant": dict(color="#1f77b4", marker="o"),
    "token_attention":      dict(color="#ff7f0e", marker="s"),
    "head_activity":        dict(color="#2ca02c", marker="^"),
    "layer_activity":       dict(color="#d62728", marker="D"),
    "combined":             dict(color="#9467bd", marker="P"),
}


def load_results(results_dir: str | Path) -> list[dict]:
    out: list[dict] = []
    for p in Path(results_dir).glob("**/*.json"):
        if p.name == "all_results.json":
            continue
        try:
            with open(p) as f:
                obj = json.load(f)
            if "task" in obj and "strategy" in obj and "score" in obj:
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def _group(results, task: str):
    by_strat: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for r in results:
        if r["task"] != task:
            continue
        by_strat[r["strategy"]].append(
            (r["keep_ratio"], r["score"], r["kv_mb_avg"])
        )
    for k in by_strat:
        by_strat[k].sort()
    return by_strat


def plot_score_vs_ratio(results_dir: str | Path, output_dir: str | Path,
                        tasks: list[str] | None = None) -> None:
    results = load_results(results_dir)
    if not results:
        print(f"No results in {results_dir}; nothing to plot")
        return
    if tasks is None:
        tasks = sorted({r["task"] for r in results})
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for task in tasks:
        by_strat = _group(results, task)
        if not by_strat:
            continue
        plt.figure(figsize=(7, 5))
        for strat, points in by_strat.items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.plot(xs, ys, label=strat, **STRATEGY_STYLE.get(strat, {}))
        plt.xlabel("keep ratio")
        plt.ylabel("score")
        plt.title(f"{task}: score vs keep ratio")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = Path(output_dir) / f"{task}_score_vs_ratio.png"
        plt.savefig(out, dpi=130)
        plt.close()
        print(f"wrote {out}")


def plot_kv_vs_ratio(results_dir: str | Path, output_dir: str | Path,
                     tasks: list[str] | None = None) -> None:
    results = load_results(results_dir)
    if not results:
        return
    if tasks is None:
        tasks = sorted({r["task"] for r in results})
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for task in tasks:
        by_strat = _group(results, task)
        if not by_strat:
            continue
        plt.figure(figsize=(7, 5))
        for strat, points in by_strat.items():
            xs = [p[0] for p in points]
            ys = [p[2] for p in points]
            plt.plot(xs, ys, label=strat, **STRATEGY_STYLE.get(strat, {}))
        plt.xlabel("keep ratio")
        plt.ylabel("KV cache MB (avg)")
        plt.title(f"{task}: KV cache size vs keep ratio")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = Path(output_dir) / f"{task}_kv_vs_ratio.png"
        plt.savefig(out, dpi=130)
        plt.close()
        print(f"wrote {out}")


def plot_score_vs_kv(results_dir: str | Path, output_dir: str | Path,
                     tasks: list[str] | None = None) -> None:
    """Pareto view: score vs realized KV cache size."""
    results = load_results(results_dir)
    if not results:
        return
    if tasks is None:
        tasks = sorted({r["task"] for r in results})
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for task in tasks:
        by_strat = _group(results, task)
        if not by_strat:
            continue
        plt.figure(figsize=(7, 5))
        for strat, points in by_strat.items():
            xs = [p[2] for p in points]
            ys = [p[1] for p in points]
            plt.plot(xs, ys, label=strat, **STRATEGY_STYLE.get(strat, {}))
        plt.xlabel("KV cache MB (avg)")
        plt.ylabel("score")
        plt.title(f"{task}: Pareto score-vs-KV")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = Path(output_dir) / f"{task}_score_vs_kv.png"
        plt.savefig(out, dpi=130)
        plt.close()
        print(f"wrote {out}")
