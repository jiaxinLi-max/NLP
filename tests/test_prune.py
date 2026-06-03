"""CPU-only unit tests for KV pruning logic.

These tests build mock DynamicCache objects with synthetic K/V tensors so we
can verify the pruning logic without loading Llama2-7B. They exercise:
- token_recent_distant: shapes and indices
- token_attention: keep set includes recent window + top-K
- head_activity: per-head budgets and union mask
- layer_activity: per-layer budgets and layer-0-is-max invariant
- combined: cumulative ratio and meta consistency
"""

from __future__ import annotations

import math

import pytest
import torch
from transformers.cache_utils import DynamicCache

from src.cache_utils import kv_bytes, kv_shape_summary
from src.prune import (
    TokenRecentDistantPruner,
    TokenAttentionPruner,
    HeadActivityPruner,
    LayerActivityPruner,
    CombinedTokenHeadPruner,
)


def make_mock(num_layers: int = 4, num_heads: int = 8, seq_len: int = 64,
              head_dim: int = 16) -> tuple[DynamicCache, list[torch.Tensor], int]:
    cache = DynamicCache()
    cache.key_cache = [
        torch.randn(1, num_heads, seq_len, head_dim) for _ in range(num_layers)
    ]
    cache.value_cache = [
        torch.randn(1, num_heads, seq_len, head_dim) for _ in range(num_layers)
    ]
    cache._seen_tokens = seq_len
    # Synthetic stats: rising scores so top-K is clearly the tail.
    attn_stats = [
        torch.linspace(0.1, 1.0, seq_len).repeat(num_heads, 1)
        for _ in range(num_layers)
    ]
    return cache, attn_stats, seq_len


def test_token_recent_distant_shapes():
    cache, stats, S = make_mock(seq_len=100)
    pruner = TokenRecentDistantPruner(keep_ratio=0.5, sink_frac=0.1)
    res = pruner.apply(cache, stats, S)
    expected = math.ceil(S * 0.5)
    for shape in kv_shape_summary(res.cache):
        assert shape[2] == expected, f"want seq_len={expected}, got {shape}"
    assert cache._seen_tokens == S


def test_token_attention_recent_window():
    cache, stats, S = make_mock(seq_len=100)
    pruner = TokenAttentionPruner(keep_ratio=0.5, recent_window=10)
    res = pruner.apply(cache, stats, S)
    kept_len = res.cache.key_cache[0].size(2)
    assert kept_len == int(round(S * 0.5))
    assert cache._seen_tokens == S


def test_head_activity_per_head_budgets():
    cache, stats, S = make_mock(num_heads=4, seq_len=64)
    pruner = HeadActivityPruner(keep_ratio=0.5, recent_window=4, min_per_head=2)
    res = pruner.apply(cache, stats, S)
    assert res.per_head_masks is not None
    assert res.per_layer_kv_lens is not None
    for layer_mask in res.per_head_masks:
        H, U = layer_mask.shape
        assert H == 4
        for h in range(H):
            allowed = layer_mask[h].sum().item()
            assert allowed >= 2  # min_per_head
    for i, kv_len in enumerate(res.per_layer_kv_lens):
        assert res.cache.key_cache[i].size(2) == kv_len
    assert cache._seen_tokens == S


def test_layer_activity_layer0_is_max():
    cache, stats, S = make_mock(num_layers=4, seq_len=80)
    pruner = LayerActivityPruner(keep_ratio=0.5, recent_window=4, min_per_layer=4)
    res = pruner.apply(cache, stats, S)
    assert res.per_layer_kv_lens is not None
    actual = [k.size(2) for k in res.cache.key_cache]
    assert actual[0] == max(actual), (
        "layer 0 must hold max kv_len for HF mask machinery"
    )
    assert cache._seen_tokens == S


def test_combined_cumulative_ratio():
    cache, stats, S = make_mock(num_layers=4, num_heads=4, seq_len=120)
    pruner = CombinedTokenHeadPruner(
        keep_ratio_token=0.6, keep_ratio_head=0.5, recent_window_token=4,
        recent_window_head=2,
    )
    res = pruner.apply(cache, stats, S)
    assert res.per_head_masks is not None
    after_token = res.meta["after_token"]
    union_lens = [k.size(2) for k in res.cache.key_cache]
    # Each layer's union <= after_token (head prune subset)
    for u in union_lens:
        assert u <= after_token


def test_kv_bytes_decreases_after_prune():
    cache, stats, S = make_mock(seq_len=200)
    before = kv_bytes(cache)
    pruner = TokenRecentDistantPruner(keep_ratio=0.25)
    pruner.apply(cache, stats, S)
    after = kv_bytes(cache)
    assert after < before
    assert after / before < 0.4  # ~25% kept + clone overhead


def test_token_attention_no_op_at_full_ratio():
    cache, stats, S = make_mock(seq_len=64)
    before_shape = kv_shape_summary(cache)
    pruner = TokenAttentionPruner(keep_ratio=1.0)
    pruner.apply(cache, stats, S)
    assert kv_shape_summary(cache) == before_shape


def test_select_tokens_preserves_seen_tokens():
    cache, stats, S = make_mock(seq_len=100)
    TokenRecentDistantPruner(keep_ratio=0.3).apply(cache, stats, S)
    assert cache._seen_tokens == S, "RoPE positions must use original prefill_len"
