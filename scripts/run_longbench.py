"""Run a LongBench task subset with a given pruning strategy. Writes JSON."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.longbench import load_longbench, score
from src.inference import run_with_pruning
from src.model_loader import DEFAULT_MODEL_ID, load_model
from src.prune import build_pruner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--strategy", default="none")
    ap.add_argument("--keep-ratio", type=float, default=1.0)
    ap.add_argument("--tasks", default="narrativeqa,qasper,gov_report,hotpotqa")
    ap.add_argument("--num-examples", type=int, default=50)
    ap.add_argument("--max-prompt-tokens", type=int, default=3500)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--quant-4bit", action="store_true", default=True)
    ap.add_argument("--no-quant", dest="quant_4bit", action="store_false")
    ap.add_argument("--output-dir", default="results/longbench")
    args = ap.parse_args()

    print(f"Loading {args.model_id} (4-bit={args.quant_4bit})...")
    model, tokenizer = load_model(
        model_id=args.model_id, attn_impl="eager", quant_4bit=args.quant_4bit,
    )
    pruner = None if args.strategy == "none" else build_pruner(args.strategy, args.keep_ratio)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for task in args.tasks.split(","):
        task = task.strip()
        examples = load_longbench(
            task, tokenizer, max_length=args.max_prompt_tokens, limit=args.num_examples,
        )
        scores: list[float] = []
        kv_mb_list: list[float] = []
        peak_list: list[float] = []
        latency_list: list[float] = []

        for ex in tqdm(examples, desc=f"{task} {args.strategy} @ {args.keep_ratio}"):
            t0 = time.time()
            try:
                res = run_with_pruning(
                    model, tokenizer, ex.prompt,
                    pruner=pruner,
                    max_new_tokens=ex.max_new,
                    chunk_size=args.chunk_size,
                    eos_token_id=tokenizer.eos_token_id,
                )
            except torch.cuda.OutOfMemoryError as e:
                print(f"OOM on {task} example: {e}")
                torch.cuda.empty_cache()
                continue
            latency_list.append(time.time() - t0)
            scores.append(score(task, res.text, ex.answers))
            kv_mb_list.append(res.kv_bytes_after_prune / (1024 ** 2))
            peak_list.append(res.peak_mem_mb)

        summary = {
            "task": task,
            "strategy": args.strategy,
            "keep_ratio": args.keep_ratio,
            "model_id": args.model_id,
            "num_examples": len(scores),
            "score": sum(scores) / max(len(scores), 1),
            "kv_mb_avg": sum(kv_mb_list) / max(len(kv_mb_list), 1),
            "peak_mem_mb_avg": sum(peak_list) / max(len(peak_list), 1),
            "latency_s_avg": sum(latency_list) / max(len(latency_list), 1),
        }
        out = Path(args.output_dir) / f"{task}_{args.strategy}_{args.keep_ratio}.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
