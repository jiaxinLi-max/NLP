"""Quick end-to-end sanity test: load model, run one GSM8K example with two
strategies, print KV bytes and peak memory. Used to confirm the docker
pipeline works before launching the full ablation grid."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.gsm8k import build_prompt, is_correct, load_gsm8k
from src.inference import run_with_pruning
from src.model_loader import DEFAULT_MODEL_ID, load_model
from src.prune import build_pruner


def main():
    print(f"Loading {DEFAULT_MODEL_ID} in 4-bit (eager attn)...")
    t0 = time.time()
    model, tokenizer = load_model(
        model_id=DEFAULT_MODEL_ID,
        attn_impl="eager",
        quant_4bit=True,
    )
    print(f"Loaded in {time.time()-t0:.1f}s")
    if torch.cuda.is_available():
        print(f"After load: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    examples = load_gsm8k(split="test", limit=2)
    ex = examples[0]
    prompt = build_prompt(ex.question)
    print(f"\nPrompt token count: {len(tokenizer.encode(prompt))}")
    print(f"Gold answer: {ex.answer}")

    for label, strat, ratio in [
        ("baseline", "none", 1.0),
        ("token_attn@0.5", "token_attention", 0.5),
        ("head_act@0.5", "head_activity", 0.5),
        ("layer_act@0.5", "layer_activity", 0.5),
    ]:
        pruner = None if strat == "none" else build_pruner(strat, ratio)
        torch.cuda.empty_cache()
        t1 = time.time()
        res = run_with_pruning(
            model, tokenizer, prompt,
            pruner=pruner,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,
        )
        dt = time.time() - t1
        kv_mb = res.kv_bytes_after_prune / (1024 ** 2)
        ok = is_correct(res.text, ex.answer)
        gen = res.text[len(prompt):] if res.text.startswith(prompt) else res.text
        print(f"\n[{label}] {dt:.1f}s, KV={kv_mb:.1f}MB, peak={res.peak_mem_mb:.0f}MB, correct={ok}")
        print(f"  gen: {gen[:200]!r}")


if __name__ == "__main__":
    main()
