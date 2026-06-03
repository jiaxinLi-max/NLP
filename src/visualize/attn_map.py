"""Attention heatmaps: pre- and post-prune visualizations.

For a single example, run prefill, dump per-layer averaged attention, then
overlay a mask showing which key positions were dropped by a given pruner.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def attention_heatmap(
    attn_stats: list[torch.Tensor],
    layer_idx: int,
    out_path: str,
    title: str = "",
) -> None:
    """Plot mean-attention-received per (head, key_pos) for a single layer."""
    stat = attn_stats[layer_idx].detach().float().cpu().numpy()  # [H, S]
    plt.figure(figsize=(10, 4))
    plt.imshow(stat, aspect="auto", cmap="viridis")
    plt.colorbar(label="attention received (sum over queries)")
    plt.xlabel("key position")
    plt.ylabel("head")
    plt.title(title or f"Layer {layer_idx} attention received")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()


@torch.no_grad()
def kept_positions_overlay(
    attn_stats: list[torch.Tensor],
    layer_idx: int,
    keep_indices: torch.Tensor,
    out_path: str,
    title: str = "",
) -> None:
    """Overlay kept-position bands on the attention heatmap."""
    stat = attn_stats[layer_idx].detach().float().cpu().numpy()
    H, S = stat.shape
    kept = np.zeros(S, dtype=bool)
    kept[keep_indices.cpu().numpy()] = True

    fig, ax = plt.subplots(2, 1, figsize=(10, 5),
                           gridspec_kw={"height_ratios": [4, 1]}, sharex=True)
    ax[0].imshow(stat, aspect="auto", cmap="viridis")
    ax[0].set_ylabel("head")
    ax[0].set_title(title or f"Layer {layer_idx} attention with prune mask")
    ax[1].imshow(kept[None, :], aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax[1].set_yticks([])
    ax[1].set_xlabel("key position (white = kept, black = dropped)")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()
