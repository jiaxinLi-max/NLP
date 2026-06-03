"""Registry mapping strategy names + config dicts to Pruner instances."""

from __future__ import annotations

from .base import Pruner, PruneResult
from .token_recent_distant import TokenRecentDistantPruner
from .token_attention import TokenAttentionPruner
from .head_activity import HeadActivityPruner
from .layer_activity import LayerActivityPruner
from .combined import CombinedTokenHeadPruner


def build_pruner(name: str, keep_ratio: float, **kwargs) -> Pruner:
    if name == "none":
        return NoOpPruner()
    if name == "token_recent_distant":
        return TokenRecentDistantPruner(keep_ratio=keep_ratio, **kwargs)
    if name == "token_attention":
        return TokenAttentionPruner(keep_ratio=keep_ratio, **kwargs)
    if name == "head_activity":
        return HeadActivityPruner(keep_ratio=keep_ratio, **kwargs)
    if name == "layer_activity":
        return LayerActivityPruner(keep_ratio=keep_ratio, **kwargs)
    if name == "combined":
        # By convention, split the budget equally across the two stages so
        # the cumulative ratio matches `keep_ratio`.
        per_stage = keep_ratio ** 0.5
        return CombinedTokenHeadPruner(
            keep_ratio_token=per_stage,
            keep_ratio_head=per_stage,
            **kwargs,
        )
    raise ValueError(f"unknown pruner: {name}")


class NoOpPruner(Pruner):
    name = "none"

    def apply(self, cache, attn_stats, prefill_len):
        return PruneResult(cache=cache)


__all__ = [
    "build_pruner",
    "Pruner",
    "PruneResult",
    "NoOpPruner",
    "TokenRecentDistantPruner",
    "TokenAttentionPruner",
    "HeadActivityPruner",
    "LayerActivityPruner",
    "CombinedTokenHeadPruner",
]
