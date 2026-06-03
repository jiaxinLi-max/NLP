"""Head-level pruning: per-head different budgets within the same layer.

Algorithm:
- score_per_head[layer, h] = mean attention magnitude received by head h
- per_head_budget[layer, h] = round(num_heads * S * total_keep_ratio
                                    * score / sum(scores)) per layer
- For each layer, pick the union of kept positions across its heads, prune
  K/V to that union, and produce a per-head boolean mask telling decode
  which positions each head is allowed to attend to.

Decode adds an additive -inf bias on disallowed (head, key) pairs after
Q @ K.T but before softmax, so heads keep batched matmul speed.
"""

from __future__ import annotations

from typing import List

import torch
from transformers.cache_utils import DynamicCache

from .base import PruneResult, Pruner
from ..cache_utils import select_tokens_per_layer


class HeadActivityPruner(Pruner):
    name = "head_activity"

    def __init__(self, keep_ratio: float, recent_window: int = 16, min_per_head: int = 4):
        assert 0.0 < keep_ratio <= 1.0
        self.keep_ratio = keep_ratio
        self.recent_window = recent_window
        self.min_per_head = min_per_head

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
        keep_indices_per_layer: List[torch.Tensor] = []
        per_head_masks: List[torch.Tensor] = []

        for layer_idx, stat in enumerate(attn_stats):
            H, _ = stat.shape
            # Allocate budget per head proportional to its mean activity.
            head_score = stat.mean(dim=1)  # [H]
            head_score = torch.clamp(head_score, min=1e-8)
            head_score = head_score / head_score.sum()
            total_budget = max(int(round(H * S * self.keep_ratio)), H * self.min_per_head)
            raw = head_score * total_budget
            per_head_budget = torch.clamp(raw.round().long(), min=self.min_per_head, max=S)

            recent = min(self.recent_window, S)
            recent_idx = torch.arange(S - recent, S, device=device, dtype=torch.long)

            # Build per-head kept positions (always include recent window).
            kept_per_head: list[torch.Tensor] = []
            for h in range(H):
                budget = int(per_head_budget[h].item())
                if budget >= S:
                    kept = torch.arange(S, device=device, dtype=torch.long)
                else:
                    score = stat[h].clone()
                    score[S - recent:] = float("-inf")
                    extra = max(budget - recent, 0)
                    if extra > 0:
                        top = torch.topk(score, k=extra).indices
                        kept = torch.cat([top, recent_idx]).unique().sort().values
                    else:
                        kept = recent_idx
                kept_per_head.append(kept)

            # Union of all heads' kept positions for this layer.
            union = torch.cat(kept_per_head).unique().sort().values  # [U]
            keep_indices_per_layer.append(union)

            # Per-head mask of shape [H, U]: True = allowed, False = bias to -inf.
            U = union.numel()
            mask = torch.zeros(H, U, dtype=torch.bool, device=device)
            pos_to_idx = {int(p.item()): i for i, p in enumerate(union)}
            for h, kept in enumerate(kept_per_head):
                idxs = torch.tensor(
                    [pos_to_idx[int(p.item())] for p in kept],
                    device=device, dtype=torch.long,
                )
                mask[h, idxs] = True
            per_head_masks.append(mask)

        select_tokens_per_layer(cache, keep_indices_per_layer)
        cache._seen_tokens = S

        per_layer_kv_lens = [int(idx.numel()) for idx in keep_indices_per_layer]

        return PruneResult(
            cache=cache,
            per_layer_kv_lens=per_layer_kv_lens,
            per_head_masks=per_head_masks,
            meta={"original": S, "per_layer_kv_lens": per_layer_kv_lens},
        )
