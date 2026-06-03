"""Top-level inference function: prefill -> prune -> decode -> text.

Used by every eval harness so the orchestration is in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .cache_utils import kv_bytes, measure_peak_mem
from .decode import greedy_decode
from .prefill import run_prefill
from .prune.base import Pruner


@dataclass
class InferenceResult:
    text: str
    new_tokens: List[int]
    kv_bytes_after_prune: int
    prefill_len: int
    peak_mem_mb: float


@torch.no_grad()
def run_with_pruning(
    model,
    tokenizer,
    prompt: str,
    pruner: Optional[Pruner],
    max_new_tokens: int = 128,
    chunk_size: int = 512,
    eos_token_id: Optional[int] = None,
    collect_attn: bool = True,
) -> InferenceResult:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = enc.input_ids.to(device)
    S = input_ids.size(1)

    with measure_peak_mem(device) as mem:
        cache, attn_stats, last_logits = run_prefill(
            model, input_ids, chunk_size=chunk_size, collect_attn=collect_attn,
        )

        per_layer_kv_lens = None
        per_head_masks = None
        if pruner is not None:
            res = pruner.apply(cache, attn_stats, prefill_len=S)
            cache = res.cache
            per_layer_kv_lens = res.per_layer_kv_lens
            per_head_masks = res.per_head_masks

        # Free attention stats before decode (~num_layers * H * S floats).
        del attn_stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        kv_after = kv_bytes(cache)
        seed = int(last_logits.argmax().item())

        new_ids = greedy_decode(
            model,
            cache,
            seed_token_id=seed,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            prefill_len=S,
            per_layer_kv_lens=per_layer_kv_lens,
            per_head_masks=per_head_masks,
        )

    text = tokenizer.decode([seed] + new_ids, skip_special_tokens=True)
    return InferenceResult(
        text=text,
        new_tokens=[seed] + new_ids,
        kv_bytes_after_prune=kv_after,
        prefill_len=S,
        peak_mem_mb=mem.get("peak_mb", 0.0),
    )
