"""Pruner abstract base class.

A Pruner takes a populated KV cache plus per-layer attention statistics and
returns a *modified-in-place* cache plus optional per-layer / per-head
metadata that the decode step needs.

Return contract:
    cache: the (mutated) DynamicCache. _seen_tokens is set to the ORIGINAL
        prefill length so RoPE for new decode tokens uses correct positions.
    per_layer_kv_lens: list[int] of length num_layers, the kept K/V seq_len
        per layer. None if uniform across layers (then plain generate() works).
    per_head_masks: list[Tensor] of shape [num_heads, kept_len_at_layer_i] or
        None. When non-None, decode adds -inf bias to disallowed (head, key)
        pairs. None if no per-head masking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers.cache_utils import DynamicCache


@dataclass
class PruneResult:
    cache: DynamicCache
    per_layer_kv_lens: Optional[List[int]] = None
    per_head_masks: Optional[List[torch.Tensor]] = None
    meta: dict = field(default_factory=dict)


class Pruner(ABC):
    name: str = "base"

    @abstractmethod
    def apply(
        self,
        cache: DynamicCache,
        attn_stats: List[torch.Tensor],
        prefill_len: int,
    ) -> PruneResult:
        ...
