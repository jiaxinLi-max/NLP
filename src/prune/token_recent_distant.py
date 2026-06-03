"""Token-level pruning: keep first N_far + last N_near tokens, drop the middle.

This is the simplest pruning baseline (StreamingLLM / "attention sinks" style):
attention sinks at the start of the sequence carry disproportionate signal,
and recent tokens carry the local context. The middle is what gets cut.

Uniform across layers and heads, so model.generate() works directly.
"""

from __future__ import annotations

from typing import List

import torch
from transformers.cache_utils import DynamicCache

from .base import PruneResult, Pruner
from ..cache_utils import select_tokens_uniform


class TokenRecentDistantPruner(Pruner):
    name = "token_recent_distant"

    def __init__(self, keep_ratio: float, sink_frac: float = 0.1):
        """Keep `keep_ratio` of tokens, split as `sink_frac` sinks + rest recent.

        Args:
            keep_ratio: fraction of prefill tokens to keep, in (0, 1].
            sink_frac: fraction of the kept budget allocated to leading sinks.
        """
        assert 0.0 < keep_ratio <= 1.0
        assert 0.0 <= sink_frac <= 1.0
        self.keep_ratio = keep_ratio
        self.sink_frac = sink_frac

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

        n_sink = max(1, int(round(keep * self.sink_frac)))
        n_recent = max(1, keep - n_sink)
        # Avoid overlap when context is tiny.
        n_sink = min(n_sink, max(1, S - n_recent))

        device = cache.key_cache[0].device
        sink_idx = torch.arange(0, n_sink, device=device, dtype=torch.long)
        recent_idx = torch.arange(S - n_recent, S, device=device, dtype=torch.long)
        keep_idx = torch.cat([sink_idx, recent_idx], dim=0).unique()

        select_tokens_uniform(cache, keep_idx)
        cache._seen_tokens = S  # important: RoPE uses original positions

        return PruneResult(
            cache=cache,
            meta={
                "kept": int(keep_idx.numel()),
                "n_sink": n_sink,
                "n_recent": n_recent,
                "original": S,
            },
        )
