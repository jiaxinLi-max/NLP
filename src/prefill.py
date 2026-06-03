"""Chunked prefill that returns the populated KV cache plus per-layer attention
statistics for prune-decision making.

Why chunking: with attn_implementation="eager", a single forward over a 4 k
context materializes [B, H, S, S] softmax matrices in fp32 — about 2 GB per
layer at S=4096. We feed the prompt in chunks of `chunk_size` tokens and
accumulate per-key-position attention sums on-the-fly so we never hold the
full softmax matrix.

`attn_stats[i]` has shape `[num_heads, prefill_len]` and stores the sum of
attention weights *received* by each key position (summed over query positions
across all chunks). Higher values = more attended to = more important to keep.
"""

from __future__ import annotations

from typing import List

import torch
from transformers.cache_utils import DynamicCache


@torch.no_grad()
def run_prefill(
    model,
    input_ids: torch.Tensor,
    chunk_size: int = 512,
    collect_attn: bool = True,
) -> tuple[DynamicCache, List[torch.Tensor], torch.Tensor]:
    """Run prefill in chunks, returning (cache, attn_stats, last_logits).

    Args:
        model: HF causal LM in eager attention mode (required for collect_attn).
        input_ids: [1, S] LongTensor of prompt tokens.
        chunk_size: tokens per forward pass.
        collect_attn: if True, accumulate per-key-position attention sums.

    Returns:
        cache: DynamicCache populated with K/V for all S tokens.
        attn_stats: list of [num_heads, S] tensors (one per layer). Empty list
            if collect_attn=False.
        last_logits: [vocab] tensor — the next-token distribution after the
            full prompt. Used to pick the seed token without re-running.
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1, "expected [1, S]"
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    S = input_ids.size(1)

    cache = DynamicCache()
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    attn_stats: List[torch.Tensor] = (
        [torch.zeros(num_heads, S, device=device, dtype=torch.float32)
         for _ in range(num_layers)]
        if collect_attn
        else []
    )

    last_logits: torch.Tensor | None = None
    pos = 0
    while pos < S:
        end = min(pos + chunk_size, S)
        chunk = input_ids[:, pos:end]
        cache_position = torch.arange(pos, end, device=device, dtype=torch.long)
        position_ids = cache_position.unsqueeze(0)

        out = model(
            input_ids=chunk,
            past_key_values=cache,
            use_cache=True,
            position_ids=position_ids,
            cache_position=cache_position,
            output_attentions=collect_attn,
            return_dict=True,
        )
        cache = out.past_key_values

        if end == S:
            last_logits = out.logits[0, -1].detach().clone()

        if collect_attn:
            for i, attn in enumerate(out.attentions):
                received = attn[0].sum(dim=1)  # [H, k_len]
                k_len = received.size(1)
                attn_stats[i][:, :k_len] += received.float()
            del out

        pos = end

    assert last_logits is not None
    return cache, attn_stats, last_logits
