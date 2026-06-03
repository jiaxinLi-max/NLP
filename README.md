# KV Cache Sparsification for LLM Inference

Course research project on KV cache pruning strategies for Llama2-7B-chat.
Implements three granularities of one-shot, prefill-only pruning (token / head /
layer) plus a combined strategy, and evaluates them on GSM8K and a LongBench
subset.

## Setup

Need a CUDA GPU with at least 16 GB VRAM (free Colab/Kaggle T4 works).

```bash
pip install -r requirements.txt
huggingface-cli login   # Llama2 weights are gated
```

If you don't have access to Llama2, swap in `meta-llama/Llama-3.2-3B-Instruct`
via `--model-id` on the run scripts (note the substitution in the report).

## Quick smoke test

```bash
bash run.sh
```

Loads the model in 4-bit, runs 5 GSM8K examples without pruning and 5 with
50% token pruning, prints EM and KV-cache MB for each.

## Full evaluation

```bash
# GSM8K (200 examples, ~30 min on T4)
python scripts/run_gsm8k.py --strategy token_attention --keep-ratio 0.25 --num-examples 200

# LongBench subset (NarrativeQA, Qasper, GovReport, HotpotQA)
python scripts/run_longbench.py --strategy head_activity --keep-ratio 0.5 --tasks narrativeqa,qasper,gov_report,hotpotqa

# Full ablation sweep — strategies × ratios × tasks (~4-6 h on T4)
python scripts/run_ablations.py --output-dir results/

# Generate figures
python scripts/make_figures.py --results-dir results/ --output-dir report/figures/
```

## Strategies

| Granularity | Strategy | Idea |
|---|---|---|
| Token | `token_recent_distant` | keep first `N_far` and last `N_near` tokens |
| Token | `token_attention` | keep top-K tokens by mean attention received |
| Head | `head_activity` | per-head budget proportional to mean attention magnitude |
| Layer | `layer_activity` | per-layer budget proportional to mean attention magnitude |
| Combined | `combined` | token-prune first, then head-prune the survivors |

## Project structure

See `report/report.md` for design rationale. Key modules:

- `src/model_loader.py` — 4-bit Llama2-7B-chat loader, switchable attn impl
- `src/cache_utils.py` — KV byte counting, peak-mem context, prune helpers
- `src/prefill.py` — chunked prefill that returns cache + per-layer attention stats
- `src/decode.py` — custom greedy decode supporting per-layer kv_lens and per-head bias
- `src/prune/` — five pruner implementations
- `src/eval/` — GSM8K and LongBench harnesses
- `src/visualize/` — accuracy/memory plots and attention heatmaps
- `tests/` — pytest suite using mock tensors (CPU-only, no model required)

## Design notes

- Pinned to `transformers==4.46.3` because v5 refactored `DynamicCache`. All
  published baselines (H2O, SnapKV, PyramidKV) target the v4 API.
- `_seen_tokens` is set to the **original** prefill length after pruning so
  RoPE positions for new decode tokens stay correct.
- Attention scores require `attn_implementation="eager"`; we chunk the prefill
  to avoid materializing full `[B,H,S,S]` softmax matrices.
- Per-head different budgets within one layer use union-of-kept-positions
  plus a per-head boolean bias mask, preserving batched matmul.
- Per-layer different sequence lengths break `model.generate()`, so head- and
  layer-prune go through `src/decode.greedy_decode`.
