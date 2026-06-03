"""Model loader for Llama2-7B-chat with switchable attention implementation.

Notes:
- Uses 4-bit NF4 quantization (bitsandbytes) so weights fit in ~4 GB on a T4.
- KV cache stays in fp16 (bnb only quantizes weights), giving real measurable
  memory savings when we prune.
- attn_implementation switch: "eager" exposes per-head attention weights for
  prune-decision prefill; "sdpa" is faster for plain decode.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


DEFAULT_MODEL_ID = "/root/bayes-tmp/project/models/llama2-7b-chat"


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    attn_impl: str = "eager",
    quant_4bit: bool = True,
    device_map: str | dict = "auto",
    dtype: torch.dtype = torch.float16,
):
    """Load Llama2-7B-chat (or compatible) and its tokenizer.

    Args:
        model_id: HF repo id. Use Llama-3.2-3B-Instruct as a smaller fallback.
        attn_impl: "eager" | "sdpa" | "flash_attention_2".
        quant_4bit: load weights in NF4. Required for T4-class GPUs.
        device_map: HF device_map argument; "auto" lets accelerate decide.
        dtype: compute dtype (used by bnb_4bit_compute_dtype and as torch_dtype
            when not quantizing).

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs = {
        "device_map": device_map,
        "attn_implementation": attn_impl,
    }
    if quant_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer


def switch_attn_impl(model, attn_impl: str):
    """Swap attention implementation in place after model load.

    Used to flip between "eager" (for score-collection prefill) and "sdpa"
    (for fast decode) on the same model instance.
    """
    model.config._attn_implementation = attn_impl
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = attn_impl
    return model
