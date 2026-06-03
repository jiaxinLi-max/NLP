"""Layer-level pruning: per-layer different total budgets.

Each layer gets a budget proportional to its mean attention magnitude.
Within a layer, pruning is uniform across heads (top-K + recent window).
Different layers end up with different K/V seq_len, so decode must use
the custom loop in src/decode.greedy_decode.
"""

from __future__ import annotations

from typing import List

import torch
from transformers.cache_utils import DynamicCache

from .base import PruneResult, Pruner
from ..cache_utils import select_tokens_per_layer


class LayerActivityPruner(Pruner):
    name = "layer_activity"

    def __init__(self, keep_ratio: float, recent_window: int = 32, min_per_layer: int = 16):
        assert 0.0 < keep_ratio <= 1.0
        self.keep_ratio = keep_ratio
        self.recent_window = recent_window
        self.min_per_layer = min_per_layer

    def apply(
        self,
        cache: DynamicCache,
        attn_stats: List[torch.Tensor],
        prefill_len: int,
    ) -> PruneResult:
        S = prefill_len
        if self.keep_ratio >= 1.0:
            return PruneResult(cache=cache)

        device = cache.key_cache[0].device
        L = len(attn_stats)
        layer_scores = torch.tensor(
            [float(s.mean().item()) for s in attn_stats], device=device, dtype=torch.float32,
        )
        layer_scores = torch.clamp(layer_scores, min=1e-8)
        layer_scores = layer_scores / layer_scores.sum()

        total_budget = max(int(round(L * S * self.keep_ratio)), L * self.min_per_layer)
        raw = layer_scores * total_budget
        per_layer_budget = torch.clamp(raw.round().long(), min=self.min_per_layer, max=S)

        keep_indices_per_layer: List[torch.Tensor] = []
        for i, stat in enumerate(attn_stats):
            budget = int(per_layer_budget[i].item())
            if budget >= S:
                keep_indices_per_layer.append(torch.arange(S, device=device, dtype=torch.long))
                continue
            recent = min(self.recent_window, S, budget)
            recent_idx = torch.arange(S - recent, S, device=device, dtype=torch.long)
            score = stat.mean(dim=0).clone()  # [S]
            score[S - recent:] = float("-inf")
            extra = budget - recent
            if extra > 0:
                top = torch.topk(score, k=extra).indices
                kept = torch.cat([top, recent_idx]).unique().sort().values
            else:
                kept = recent_idx
            keep_indices_per_layer.append(kept)

        # Decode loop requires layer 0 to hold the max kv_len (HF builds the
        # global mask from layer 0). Find the layer with the most kept and
        # swap budgets so layer 0 gets that count. Easiest: ensure layer 0's
        # kept set is the largest by top-up if needed.
        max_kept = max(idx.numel() for idx in keep_indices_per_layer)
        if keep_indices_per_layer[0].numel() < max_kept:
            # Top up layer 0 with extra positions by score.
            already = set(int(p.item()) for p in keep_indices_per_layer[0])
            score0 = attn_stats[0].mean(dim=0)
            ranked = torch.argsort(score0, descending=True).tolist()
            extras = []
            for p in ranked:
                if p not in already:
                    extras.append(p)
                    if keep_indices_per_layer[0].numel() + len(extras) >= max_kept:
                        break
            extras_t = torch.tensor(extras, device=device, dtype=torch.long)
            keep_indices_per_layer[0] = (
                torch.cat([keep_indices_per_layer[0], extras_t]).unique().sort().values
            )

        select_tokens_per_layer(cache, keep_indices_per_layer)
        cache._seen_tokens = S

        per_layer_kv_lens = [int(idx.numel()) for idx in keep_indices_per_layer]
        return PruneResult(
            cache=cache,
            per_layer_kv_lens=per_layer_kv_lens,
            meta={"original": S, "per_layer_kv_lens": per_layer_kv_lens},
        )
