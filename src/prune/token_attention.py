"""Token pruning by mean attention received (H2O / SnapKV style).

Score each prefill token by how much attention it received during prefill,
averaged across heads (and layers, for the simple variant). Keep the top-K
plus a guaranteed recent window so the most-recent local context is never
dropped (SnapKV's key insight).

Score is uniform per-token across layers — every layer keeps the same set,
so model.generate() works directly.
"""

from __future__ import annotations

from typing import List

import torch
from transformers.cache_utils import DynamicCache

from .base import PruneResult, Pruner
from ..cache_utils import select_tokens_uniform


class TokenAttentionPruner(Pruner):
    name = "token_attention"

    def __init__(self, keep_ratio: float, recent_window: int = 32):
        """Keep top-K tokens by score + always keep last `recent_window`."""
        assert 0.0 < keep_ratio <= 1.0
        self.keep_ratio = keep_ratio
        self.recent_window = recent_window

    def apply(
        self,
        cache: DynamicCache,
        attn_stats: List[torch.Tensor],
        prefill_len: int,
    ) -> PruneResult:
        S = prefill_len
        keep = max(1, int(round(S * self.keep_ratio)))
        if keep >= S:
            return PruneResult(cache=cache)

        # Score: average attention received across all layers and heads.
        device = cache.key_cache[0].device
        score = torch.zeros(S, device=device, dtype=torch.float32)
        for stat in attn_stats:
            score += stat.mean(dim=0)  # [S]
        score /= max(len(attn_stats), 1)

        recent = min(self.recent_window, S, keep)
        recent_idx = torch.arange(S - recent, S, device=device, dtype=torch.long)

        # Mask out the recent window in score so we don't double-count.
        score_masked = score.clone()
        score_masked[S - recent:] = float("-inf")
        topk_budget = keep - recent
        if topk_budget > 0:
            top_idx = torch.topk(score_masked, k=topk_budget).indices
            keep_idx = torch.cat([top_idx, recent_idx], dim=0)
        else:
            keep_idx = recent_idx
        keep_idx = keep_idx.sort().values.unique()

        select_tokens_uniform(cache, keep_idx)
        cache._seen_tokens = S

        return PruneResult(
            cache=cache,
            meta={
                "kept": int(keep_idx.numel()),
                "recent_window": recent,
                "original": S,
                "keep_idx": keep_idx,
            },
        )
