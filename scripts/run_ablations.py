"""Sweep strategies × keep_ratios across GSM8K and LongBench tasks.

Loads the model once and reuses it across runs to avoid re-downloading.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.gsm8k import build_prompt, is_correct, load_gsm8k
from src.eval.longbench import load_longbench, score
from src.inference import run_with_pruning
from src.model_loader import DEFAULT_MODEL_ID, load_model
from src.prune import build_pruner


STRATEGIES = ["none", "token_recent_distant", "token_attention",
              "head_activity", "layer_activity", "combined"]
RATIOS = [0.1, 0.25, 0.5, 0.75]
LB_TASKS = ["narrativeqa", "qasper", "gov_report", "hotpotqa"]


def run_gsm8k(model, tokenizer, examples, strategy, ratio, max_new):
    pruner = None if strategy == "none" else build_pruner(strategy, ratio)
    correct = total = 0
    kv = peak = lat = 0.0
    for ex in tqdm(examples, desc=f"gsm8k {strategy}@{ratio}", leave=False):
        prompt = build_prompt(ex.question)
        t0 = time.time()
        try:
            res = run_with_pruning(
                model, tokenizer, prompt, pruner=pruner,
                max_new_tokens=max_new, eos_token_id=tokenizer.eos_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        lat += time.time() - t0
        correct += int(is_correct(res.text, ex.answer))
        total += 1
        kv += res.kv_bytes_after_prune / (1024 ** 2)
        peak += res.peak_mem_mb
    return {
        "task": "gsm8k", "strategy": strategy, "keep_ratio": ratio,
        "num_examples": total,
        "score": correct / max(total, 1),
        "kv_mb_avg": kv / max(total, 1),
        "peak_mem_mb_avg": peak / max(total, 1),
        "latency_s_avg": lat / max(total, 1),
    }


def run_longbench_task(model, tokenizer, examples, task, strategy, ratio):
    pruner = None if strategy == "none" else build_pruner(strategy, ratio)
    scores = []
    kv_l = peak_l = lat_l = 0.0
    for ex in tqdm(examples, desc=f"{task} {strategy}@{ratio}", leave=False):
        t0 = time.time()
        try:
            res = run_with_pruning(
                model, tokenizer, ex.prompt, pruner=pruner,
                max_new_tokens=ex.max_new, eos_token_id=tokenizer.eos_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        lat_l += time.time() - t0
        scores.append(score(task, res.text, ex.answers))
        kv_l += res.kv_bytes_after_prune / (1024 ** 2)
        peak_l += res.peak_mem_mb
    return {
        "task": task, "strategy": strategy, "keep_ratio": ratio,
        "num_examples": len(scores),
        "score": sum(scores) / max(len(scores), 1),
        "kv_mb_avg": kv_l / max(len(scores), 1),
        "peak_mem_mb_avg": peak_l / max(len(scores), 1),
        "latency_s_avg": lat_l / max(len(scores), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--ratios", default=",".join(str(r) for r in RATIOS))
    ap.add_argument("--tasks", default="gsm8k," + ",".join(LB_TASKS))
    ap.add_argument("--num-gsm8k", type=int, default=100)
    ap.add_argument("--num-longbench", type=int, default=30)
    ap.add_argument("--max-prompt-tokens", type=int, default=3000)
    ap.add_argument("--max-new-tokens-gsm8k", type=int, default=256)
    ap.add_argument("--output-dir", default="results")
    args = ap.parse_args()

    strategies = args.strategies.split(",")
    ratios = [float(r) for r in args.ratios.split(",")]
    tasks = args.tasks.split(",")

    model, tokenizer = load_model(model_id=args.model_id, attn_impl="eager", quant_4bit=True)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    if "gsm8k" in tasks:
        gsm_examples = load_gsm8k(split="test", limit=args.num_gsm8k)
        for strat in strategies:
            for r in (ratios if strat != "none" else [1.0]):
                summary = run_gsm8k(model, tokenizer, gsm_examples, strat, r, args.max_new_tokens_gsm8k)
                summary["model_id"] = args.model_id
                all_results.append(summary)
                with open(out_dir / f"gsm8k_{strat}_{r}.json", "w") as f:
                    json.dump(summary, f, indent=2)

    for task in tasks:
        if task == "gsm8k":
            continue
        examples = load_longbench(task, tokenizer, args.max_prompt_tokens, args.num_longbench)
        for strat in strategies:
            for r in (ratios if strat != "none" else [1.0]):
                summary = run_longbench_task(model, tokenizer, examples, task, strat, r)
                summary["model_id"] = args.model_id
                all_results.append(summary)
                with open(out_dir / f"{task}_{strat}_{r}.json", "w") as f:
                    json.dump(summary, f, indent=2)

    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {len(all_results)} results to {out_dir}/")


if __name__ == "__main__":
    main()
