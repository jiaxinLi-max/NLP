"""Custom greedy decode for the (possibly pruned) KV cache.

Key facts about transformers v4.46 that this code relies on:

- `_update_causal_mask` builds a single global mask using
  `past_key_values.get_seq_length()` which returns *layer 0's* actual K
  length. Each layer's eager attention then slices that mask to its own K
  length. This means: for layer-level pruning to work, layer 0 must hold the
  maximum K length across all layers. The LayerActivityPruner enforces this.

- `_seen_tokens` after pruning is the original prefill length. RoPE positions
  for new decode tokens come from the `position_ids` we pass.

- `cache_position[0]` controls the visibility threshold of the prepared mask.
  We pass `[prefill_len]` so all kept K positions (which all have logical
  position ≤ prefill_len) are unmasked.

- Per-head bias: when present, we monkey-patch each layer's self_attn.forward
  to ADD a precomputed [1, H, 1, max_kv_len_used] bias of -inf on disallowed
  (head, key) entries. The bias is sized to cover the union-pruned K length
  plus all decode-step appends; new decode tokens get bias=0 (no restriction).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional

import torch
from transformers.cache_utils import DynamicCache


@contextmanager
def _patch_attn_for_head_bias(
    model,
    per_head_masks: List[torch.Tensor],
    max_decode_steps: int,
):
    originals = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        head_mask = per_head_masks[i]  # [H, kv_len_i] bool
        H, kv_len_i = head_mask.shape
        total_len = kv_len_i + max_decode_steps + 1
        # Bias shape [1, H, 1, total_len]: -inf where disallowed, 0 elsewhere.
        bias = torch.zeros(
            1, H, 1, total_len,
            dtype=torch.float32, device=head_mask.device,
        )
        bias[:, :, :, :kv_len_i].masked_fill_(
            ~head_mask.unsqueeze(0).unsqueeze(2), float("-inf")
        )
        attn._kv_head_bias = bias
        originals.append((attn, attn.forward))

    def make_patched(orig_forward, attn_module):
        def forward(hidden_states, attention_mask=None, **kwargs):
            new_bias = attn_module._kv_head_bias
            if attention_mask is None:
                merged = new_bias
            else:
                # eager_attention_forward slices to key_states.shape[-2], so
                # both tensors get truncated to the same width by the slice;
                # we only need to ensure the bias covers all K positions.
                merged = attention_mask + new_bias.to(attention_mask.dtype)
            return orig_forward(hidden_states, attention_mask=merged, **kwargs)
        return forward

    for attn, orig in originals:
        attn.forward = make_patched(orig, attn)
    try:
        yield
    finally:
        for attn, orig in originals:
            attn.forward = orig
            if hasattr(attn, "_kv_head_bias"):
                delattr(attn, "_kv_head_bias")


@torch.no_grad()
def greedy_decode(
    model,
    cache: DynamicCache,
    seed_token_id: int,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    *,
    prefill_len: int,
    per_layer_kv_lens: Optional[List[int]] = None,
    per_head_masks: Optional[List[torch.Tensor]] = None,
) -> List[int]:
    """Greedy-decode after a (possibly pruned) prefill.

    Args:
        model: HF Llama model.
        cache: populated, possibly pruned DynamicCache. ``_seen_tokens`` must
            equal the original prefill length.
        seed_token_id: first new token to feed (typically argmax of last
            prompt logit from prefill).
        max_new_tokens: max tokens to generate after the seed.
        eos_token_id: stop token (optional).
        prefill_len: original prefill length. Sets RoPE positions and mask
            visibility threshold.
        per_layer_kv_lens: informational only (asserted at start).
        per_head_masks: per-layer [H, kv_len] bool masks for head-level prune.
    """
    device = next(model.parameters()).device
    if per_layer_kv_lens is not None:
        # Layer 0 must be the max for HF's global mask to cover every layer.
        actual = [k.size(2) for k in cache.key_cache]
        assert actual[0] == max(actual), (
            "layer 0 must hold max kv_len after pruning; got "
            f"layer0={actual[0]}, max={max(actual)}"
        )

    generated: List[int] = [seed_token_id]
    cur_pos = prefill_len  # logical RoPE position for the next token

    @contextmanager
    def maybe_patch():
        if per_head_masks is not None:
            with _patch_attn_for_head_bias(model, per_head_masks, max_new_tokens + 1):
                yield
        else:
            yield

    with maybe_patch():
        for step in range(max_new_tokens):
            input_ids = torch.tensor([[generated[-1]]], device=device, dtype=torch.long)
            position_ids = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
            cache_position = torch.tensor([cur_pos], device=device, dtype=torch.long)

            out = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                position_ids=position_ids,
                cache_position=cache_position,
                return_dict=True,
            )
            cache = out.past_key_values
            next_id = int(out.logits[0, -1].argmax().item())
            generated.append(next_id)
            cur_pos += 1
            if eos_token_id is not None and next_id == eos_token_id:
                break

    return generated[1:]
