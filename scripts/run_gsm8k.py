"""Run GSM8K with a given pruning strategy. Writes JSON results."""

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
from src.inference import run_with_pruning
from src.model_loader import DEFAULT_MODEL_ID, load_model
from src.prune import build_pruner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--strategy", default="none",
                    choices=["none", "token_recent_distant", "token_attention",
                             "head_activity", "layer_activity", "combined"])
    ap.add_argument("--keep-ratio", type=float, default=1.0)
    ap.add_argument("--num-examples", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--quant-4bit", action="store_true", default=True)
    ap.add_argument("--no-quant", dest="quant_4bit", action="store_false")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print(f"Loading {args.model_id} (4-bit={args.quant_4bit})...")
    model, tokenizer = load_model(
        model_id=args.model_id,
        attn_impl="eager",
        quant_4bit=args.quant_4bit,
    )

    examples = load_gsm8k(split="test", limit=args.num_examples)
    pruner = None if args.strategy == "none" else build_pruner(args.strategy, args.keep_ratio)

    correct = 0
    total = 0
    kv_mb_sum = 0.0
    peak_mb_sum = 0.0
    latency_sum = 0.0

    for ex in tqdm(examples, desc=f"GSM8K {args.strategy} @ {args.keep_ratio}"):
        prompt = build_prompt(ex.question)
        t0 = time.time()
        try:
            res = run_with_pruning(
                model, tokenizer, prompt,
                pruner=pruner,
                max_new_tokens=args.max_new_tokens,
                chunk_size=args.chunk_size,
                eos_token_id=tokenizer.eos_token_id,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM on example {total}: {e}")
            torch.cuda.empty_cache()
            continue
        latency_sum += time.time() - t0
        ok = is_correct(res.text, ex.answer)
        correct += int(ok)
        total += 1
        kv_mb_sum += res.kv_bytes_after_prune / (1024 ** 2)
        peak_mb_sum += res.peak_mem_mb

    summary = {
        "task": "gsm8k",
        "strategy": args.strategy,
        "keep_ratio": args.keep_ratio,
        "model_id": args.model_id,
        "num_examples": total,
        "exact_match": correct / max(total, 1),
        "kv_mb_avg": kv_mb_sum / max(total, 1),
        "peak_mem_mb_avg": peak_mb_sum / max(total, 1),
        "latency_s_avg": latency_sum / max(total, 1),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
