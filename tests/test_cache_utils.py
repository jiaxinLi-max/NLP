"""CPU-only tests for cache_utils.kv_bytes and select_tokens helpers."""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from src.cache_utils import (
    clone_cache,
    kv_bytes,
    kv_shape_summary,
    select_tokens_per_layer,
    select_tokens_uniform,
)


def make_cache(num_layers=2, heads=4, seq=10, head_dim=8) -> DynamicCache:
    c = DynamicCache()
    c.key_cache = [torch.randn(1, heads, seq, head_dim) for _ in range(num_layers)]
    c.value_cache = [torch.randn(1, heads, seq, head_dim) for _ in range(num_layers)]
    c._seen_tokens = seq
    return c


def test_kv_bytes_matches_manual_count():
    c = make_cache()
    expected = 0
    for t in c.key_cache + c.value_cache:
        expected += t.element_size() * t.numel()
    assert kv_bytes(c) == expected


def test_clone_cache_is_independent():
    c = make_cache()
    c2 = clone_cache(c)
    c.key_cache[0][...] = 0.0
    assert not torch.allclose(c2.key_cache[0], c.key_cache[0])
    assert c2._seen_tokens == c._seen_tokens


def test_select_tokens_uniform_shapes():
    c = make_cache(seq=20)
    keep = torch.tensor([0, 5, 10, 15, 19])
    select_tokens_uniform(c, keep)
    for shape in kv_shape_summary(c):
        assert shape[2] == 5


def test_select_tokens_per_layer_different_lens():
    c = make_cache(num_layers=3, seq=20)
    idxs = [
        torch.tensor([0, 1, 2]),
        torch.tensor([5, 6, 7, 8]),
        torch.arange(20),
    ]
    select_tokens_per_layer(c, idxs)
    shapes = kv_shape_summary(c)
    assert shapes[0][2] == 3
    assert shapes[1][2] == 4
    assert shapes[2][2] == 20
