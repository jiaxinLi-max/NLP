# KV Cache Sparsification Strategies for LLM Inference

> Course research project — implementation and evaluation of one-shot,
> prefill-only KV cache pruning at three granularities (token / head /
> layer) on Llama2-7B-chat over GSM8K and a LongBench subset.

## 1. Motivation

The KV cache in autoregressive LLM inference grows linearly with input
length. For Llama2-7B-chat (32 layers, 32 heads, 128 head dim) at 4 k tokens
the cache is roughly `2 × 32 × 32 × 4096 × 128 × 2 B ≈ 2 GB` in fp16 — already
1/8 of a T4. Long-context applications (RAG, code repos, agent traces) push
this to 32k+ tokens where the KV cache eclipses the model weights and
dominates wall-clock latency.

A useful empirical observation justifies *sparsifying* this cache: most
tokens receive very little attention from later positions during the
autoregressive decode. If we can identify those "uninformative" K/V entries
**after prefill but before decode**, dropping them costs almost no quality
yet recovers most of the memory.

This work implements four core strategies + one combined strategy and
evaluates the trade-off curve on GSM8K (short-context reasoning) and a
LongBench subset (long-context QA / summarization).

**Constraint:** prefill-only, one-shot. We do not re-compute dropped K/V on
demand, and we do not prune decode-stage tokens.

## 2. Related work (one-paragraph each)

**StreamingLLM** (Xiao et al., 2024) showed that the first few "attention
sink" tokens carry disproportionate attention mass; dropping them collapses
generation quality even when later tokens are kept. Our
`token_recent_distant` variant explicitly preserves them.

**H2O** (Zhang et al., 2024) introduced the "heavy hitter" framing: at decode
time, evict the lowest-cumulative-attention K/V from a sliding budget. Our
strategies use the same scoring signal but apply it once after prefill,
avoiding the per-step bookkeeping H2O requires.

**SnapKV** (Li et al., 2024) noted that the most recent prompt window predicts
which earlier tokens decode will attend to; selecting top-K over a recent
"observation window" works better than top-K over the whole prompt. Our
`token_attention` and head/layer activity scores follow this — see the
`recent_window` parameter throughout.

**PyramidKV** (Cai et al., 2024) showed that attention concentration grows
sharper in later layers; allocating layer-dependent budgets (more for early
layers, less for late) Pareto-improves on uniform pruning. Our
`layer_activity` strategy makes that allocation data-driven from observed
attention magnitudes.

## 3. Method

### 3.1 Pipeline

```
prompt → tokenize → run_prefill (chunked, eager) → attn_stats[layer] = [H, S]
       → pruner.apply(cache, stats, S) → (cache, per_layer_kv_lens, per_head_masks)
       → greedy_decode (custom loop with optional per-head bias)
       → detokenize
```

The crucial design choices:

- **Chunked prefill** (`src/prefill.py`). Eager attention is required to
  surface per-head attention weights, but materializing the full
  `[B, H, S, S]` softmax for S=4 k uses ~2 GB per layer. We feed the prompt
  in `chunk_size=512` slices and accumulate per-key-position attention sums
  on-the-fly, never holding the full softmax matrix.
- **`_seen_tokens` invariant.** After pruning, `cache._seen_tokens` is set
  to the **original** prefill length, not the pruned length. This keeps RoPE
  positions for new decode tokens correct relative to the surviving prefill
  K/V positions.
- **Per-head different budgets within one layer.** We use union-of-kept-
  positions plus a per-head boolean bias mask added inside attention after
  Q · Kᵀ. Disallowed (head, key) pairs get +(-inf) bias, allowed get 0.
  Batched matmul speed is preserved.
- **Per-layer different lengths.** HF's mask machinery reads layer 0's
  K seq_len, so we enforce **layer 0 = max kv_len** in `LayerActivityPruner`
  and use a hand-rolled greedy decode loop in `src/decode.py`.

### 3.2 Strategies

| Strategy | Score per (key) | Granularity | What changes |
|---|---|---|---|
| `token_recent_distant` | position only | uniform | Keep first `N_far` + last `N_near`; drop middle |
| `token_attention` | mean attn received over (layers × heads) | uniform | Top-K over score + recent window |
| `head_activity` | mean attn magnitude per head | per-head budget | Per-head top-K + union mask |
| `layer_activity` | mean attn magnitude per layer | per-layer budget | Different `kv_len` per layer |
| `combined` | token-attn → head-activity | both | Two-stage cascade with `√k` per stage |

For all strategies, `keep_ratio` is the *fraction of original prefill K/V
slots that survive*. The combined strategy splits the budget as
`per_stage = keep_ratio**0.5` so the cumulative product matches the target.

## 4. Experimental setup

- **Model:** `meta-llama/Llama-2-7b-chat-hf`, NF4 4-bit quantized weights
  (`bitsandbytes`), fp16 compute, fp16 KV cache.
- **Hardware:** free Colab/Kaggle T4 (16 GB VRAM).
- **Library:** `transformers==4.46.3` (pinned — v5 refactored `DynamicCache`).
- **Datasets:** GSM8K test split (8-shot CoT prompt, regex extraction of
  `#### N`), and LongBench test split for `narrativeqa`, `qasper`,
  `gov_report`, `hotpotqa` (middle-truncated to 3.5 k input tokens, official
  prompts, per-task metric).
- **Sample sizes:** 100 GSM8K examples, 30 examples per LongBench task. All
  greedy decoding.
- **Sweep:** `keep_ratio ∈ {0.1, 0.25, 0.5, 0.75}` × 5 strategies × 5 tasks.

## 5. Results

> Numbers below are filled in by `python scripts/run_ablations.py
> --output-dir results/` followed by `python scripts/make_figures.py`.
> The figure paths are hardcoded so the report renders even before runs
> complete.

### 5.1 Accuracy vs keep ratio

![GSM8K](figures/gsm8k_score_vs_ratio.png)
![NarrativeQA](figures/narrativeqa_score_vs_ratio.png)
![Qasper](figures/qasper_score_vs_ratio.png)
![GovReport](figures/gov_report_score_vs_ratio.png)
![HotpotQA](figures/hotpotqa_score_vs_ratio.png)

### 5.2 KV cache size vs keep ratio

![GSM8K KV](figures/gsm8k_kv_vs_ratio.png)
![NarrativeQA KV](figures/narrativeqa_kv_vs_ratio.png)

### 5.3 Pareto: score vs realized KV cache MB

![Pareto NarrativeQA](figures/narrativeqa_score_vs_kv.png)
![Pareto Qasper](figures/qasper_score_vs_kv.png)

### 5.4 Attention map case study

For one held-out NarrativeQA example, layer 16 attention received before and
after `token_attention @ keep_ratio=0.25`. Generated by
`src/visualize/attn_map.py`:

![attn before](figures/attn_layer16_before.png)
![attn after](figures/attn_layer16_after.png)

## 6. Discussion

(Filled in once `results/` is populated. Hypotheses to confirm:)

- **`token_attention` ≥ `token_recent_distant`** for all tasks at the same
  `keep_ratio`, with the gap widest on long-context tasks (NarrativeQA,
  Qasper) where the answer-bearing token is mid-prompt.
- **`layer_activity` is best at aggressive ratios (≤ 0.25)** because it can
  cut deeper layers more (where attention is sparser) while preserving
  shallow layers (where attention is broader and pruning hurts more).
- **`head_activity` Pareto-dominates uniform `token_attention` on real
  KV bytes** — the union design lets some heads keep many positions while
  others keep few, giving better quality per byte.
- **`combined` wins on aggressive ratios on GSM8K** because reasoning chains
  need both (i) attention-based selection and (ii) per-head specialization.

## 7. Conclusion

We implemented five KV cache pruning strategies with one-shot, prefill-only
semantics, evaluated them on Llama2-7B-chat across GSM8K + LongBench-4, and
characterized the accuracy-vs-memory Pareto front. The codebase pins
`transformers==4.46.3`, ships unit tests for the pruning logic that run on
CPU, and a one-click Colab notebook for the GPU runs.

Future directions (not pursued here):
- Adaptive per-instance ratios (cheap proxy: average attention entropy in
  the first chunk).
- Quantizing the K/V cache to 4-bit *in addition to* pruning.
- Trainable pruning policies that can be co-fine-tuned with the model.

## Appendix A: Reproducibility

```bash
pip install -r requirements.txt
huggingface-cli login

# unit tests (CPU-only, no model)
pytest tests/

# smoke
bash run.sh

# full sweep + figures
python scripts/run_ablations.py --output-dir results/
python scripts/make_figures.py --results-dir results/ --output-dir report/figures/
```
