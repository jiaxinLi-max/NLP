"""KV cache utilities — sizing, cloning, and peak-memory measurement.

Targets transformers v4.46 DynamicCache: cache.key_cache / cache.value_cache
are List[Tensor] of shape [B, num_kv_heads, S, head_dim], and cache._seen_tokens
tracks the logical sequence length used by RoPE.
"""

from __future__ import annotations

import contextlib
import gc
from typing import Iterator

import torch
from transformers.cache_utils import DynamicCache


def kv_bytes(cache: DynamicCache) -> int:
    """Total bytes occupied by all key/value tensors in the cache."""
    total = 0
    for t in list(cache.key_cache) + list(cache.value_cache):
        if t is None:
            continue
        total += t.element_size() * t.numel()
    return total


def kv_mb(cache: DynamicCache) -> float:
    return kv_bytes(cache) / (1024 ** 2)


def kv_shape_summary(cache: DynamicCache) -> list[tuple[int, ...]]:
    """Return [layer_idx -> key_cache.shape] for inspection / tests."""
    return [tuple(k.shape) if k is not None else () for k in cache.key_cache]


def clone_cache(cache: DynamicCache) -> DynamicCache:
    """Deep copy a cache so prune experiments don't clobber the baseline."""
    new = DynamicCache()
    new.key_cache = [k.clone() if k is not None else None for k in cache.key_cache]
    new.value_cache = [v.clone() if v is not None else None for v in cache.value_cache]
    new._seen_tokens = cache._seen_tokens
    return new


def peak_gpu_mb(device: int | torch.device = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def reset_peak_gpu(device: int | torch.device = 0) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


@contextlib.contextmanager
def measure_peak_mem(device: int | torch.device = 0) -> Iterator[dict]:
    """Context manager that records peak GPU memory delta for the enclosed block.

    Usage:
        with measure_peak_mem() as m:
            ...
        print(m["peak_mb"], m["delta_mb"])
    """
    out: dict = {}
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.memory_allocated(device)
    else:
        start = 0
    try:
        yield out
    finally:
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated(device)
            out["peak_mb"] = peak / (1024 ** 2)
            out["delta_mb"] = (peak - start) / (1024 ** 2)
        else:
            out["peak_mb"] = 0.0
            out["delta_mb"] = 0.0


def select_tokens_per_layer(
    cache: DynamicCache, keep_indices_per_layer: list[torch.Tensor]
) -> DynamicCache:
    """Apply a per-layer token-keep mask to cache in place.

    keep_indices_per_layer[i] is a 1D LongTensor of indices into the seq axis
    of layer i's key/value. Layers may have different lengths after this.
    `_seen_tokens` is left untouched so RoPE positions for new decode tokens
    keep using the original prefill length.
    """
    assert len(keep_indices_per_layer) == len(cache.key_cache)
    for i, idx in enumerate(keep_indices_per_layer):
        k, v = cache.key_cache[i], cache.value_cache[i]
        if k is None or idx is None:
            continue
        idx = idx.to(k.device)
        cache.key_cache[i] = k.index_select(2, idx).contiguous()
        cache.value_cache[i] = v.index_select(2, idx).contiguous()
    return cache


def select_tokens_uniform(cache: DynamicCache, keep_idx: torch.Tensor) -> DynamicCache:
    """Apply the same token-keep indices to every layer.

    Required when using model.generate(...) afterwards, because HF's mask
    construction reads layer 0's seq_len and assumes it's uniform.
    """
    return select_tokens_per_layer(cache, [keep_idx] * len(cache.key_cache))
