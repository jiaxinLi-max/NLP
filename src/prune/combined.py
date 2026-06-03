"""Combined token-then-head pruning.

Pipeline:
  1. token_attention prune to keep_ratio_token of the prefill (uniform across
     layers). Cheap and gives most of the memory savings.
  2. head_activity prune on the survivors with keep_ratio_head, allocating
     per-head budgets within each layer.

Cumulative kept positions ≈ keep_ratio_token * keep_ratio_head.
"""

from __future__ import annotations

from typing import List

import torch
from transformers.cache_utils import DynamicCache

from .base import PruneResult, Pruner
from .token_attention import TokenAttentionPruner
from .head_activity import HeadActivityPruner


class CombinedTokenHeadPruner(Pruner):
    name = "combined"

    def __init__(
        self,
        keep_ratio_token: float,
        keep_ratio_head: float,
        recent_window_token: int = 32,
        recent_window_head: int = 16,
    ):
        self.token_pruner = TokenAttentionPruner(
            keep_ratio=keep_ratio_token, recent_window=recent_window_token,
        )
        self.head_pruner = HeadActivityPruner(
            keep_ratio=keep_ratio_head, recent_window=recent_window_head,
        )

    def apply(
        self,
        cache: DynamicCache,
        attn_stats: List[torch.Tensor],
        prefill_len: int,
    ) -> PruneResult:
        tok_res = self.token_pruner.apply(cache, attn_stats, prefill_len)
        keep_idx = tok_res.meta.get("keep_idx")
        if keep_idx is None:
            # No-op token prune (ratio==1.0); pass through as-is.
            sliced_stats = attn_stats
            kept_len = prefill_len
        else:
            kept_len = int(keep_idx.numel())
            sliced_stats = [s.index_select(1, keep_idx) for s in attn_stats]

        head_res = self.head_pruner.apply(cache, sliced_stats, kept_len)
        # head_pruner sets _seen_tokens = kept_len; restore original for RoPE.
        cache._seen_tokens = prefill_len

        return PruneResult(
            cache=cache,
            per_layer_kv_lens=head_res.per_layer_kv_lens,
            per_head_masks=head_res.per_head_masks,
            meta={
                "original": prefill_len,
                "after_token": kept_len,
                "per_layer_kv_lens": head_res.per_layer_kv_lens,
            },
        )
