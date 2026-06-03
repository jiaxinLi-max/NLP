---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
header: 'KV Cache Pruning &mdash; Course Research'
footer: 'NLP Course Project'
style: |
  section { font-size: 24px; }
  section.lead h1 { font-size: 56px; }
  section.lead h2 { color: #555; font-weight: 400; }
  h1 { color: #1a3a6c; }
  h2 { color: #1a3a6c; }
  code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
  table { font-size: 20px; }
  .small { font-size: 18px; }
  .tiny { font-size: 16px; }
  .red { color: #c0392b; font-weight: bold; }
  .muted { color: #777; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
---

<!-- _class: lead -->

# One-Shot Prefill-Only KV Cache Pruning

## Token / Head / Layer Granularity on Llama-2-7B-chat

NLP Course Research Project

---

# Why KV Cache Matters

The KV cache scales **linearly with prompt length**:

$$
\text{KV bytes} = 2 \times L \times H \times S \times d_h \times \text{dtype}
$$

For Llama-2-7B-chat ($L{=}32, H{=}32, d_h{=}128$, fp16):

| Prompt length | KV cache size |
|---|---|
| 4 k tokens | $\approx$ 2 GB &nbsp; <span class="muted">(1/8 of T4)</span> |
| 32 k tokens | $\approx$ 16 GB &nbsp; <span class="red">> model weights</span> |
| 128 k tokens | $\approx$ 64 GB &nbsp; <span class="red">dominates latency</span> |

> **Long-context inference is bound by the cache, not the weights.**

---

# Empirical Hook

Attention is **sparse**: most prompt positions receive a tiny share of the
attention mass during decode.

If we could identify uninformative $(K, V)$ entries
**after prefill but before decode** $\Rightarrow$ drop them once,
recover memory, retain quality.

This is the bet of KV cache pruning &mdash; we test it on five strategies
under one comparable knob.

---

# Three Constraints (Project Scope)

<div class="two-col">

**(C1) One-shot**
Pruning happens once after prefill.
No recomputation, no restoration.

**(C2) Prefill-only**
Decode-stage tokens are never pruned.
The cache shrinks, then only grows.

</div>

**(C3) Single knob `keep_ratio`**
Every strategy exposes the same scalar so accuracy-vs-memory curves are
directly comparable across granularities.

<span class="muted">These constraints come straight from the course brief and bound the design space.</span>

---

# Five Strategies, Three Granularities

| Strategy | Granularity | Score | Key idea |
|---|---|---|---|
| `token_recent_distant` | uniform / token | position | Keep first $N_\text{far}$ + last $N_\text{near}$ |
| `token_attention` | uniform / token | mean attn over $L \times H$ | Top-K + recent window |
| `head_activity` | per-head budget | mean attn per head | Union mask, per-head bias |
| `layer_activity` | per-layer budget | mean attn per layer | Different `kv_len` per layer |
| `combined` | token $\to$ head | cascade | Two stages, $\sqrt{\text{ratio}}$ each |

All five share `keep_ratio` $\in [0, 1]$ as the **only** external knob.

---

# Pipeline

```
prompt
  ├─► tokenise
  ├─► chunked prefill ──► (cache, attn_stats[L][H,S], last_logits)
  ├─► pruner.apply(cache, stats, S)
  │       └─► (cache_pruned, per_layer_kv_lens, per_head_masks)
  ├─► greedy decode (custom loop, optional per-head bias)
  └─► detokenise
```

The pruner is the **only stage** that varies between strategies; everything
else is shared infrastructure.

---

# Engineering Invariant 1 &mdash; Chunked Prefill

Eager attention is required to surface per-head softmax weights, but
materialising the full $[B, H, S, S]$ matrix at $S{=}4{\rm k}$ uses
$\approx$ 2 GB **per layer**.

**Fix.** Feed the prompt in `chunk_size=512` slices and accumulate
**per-key-position attention sums** on the fly:

$$
A^{(\ell)}_{h, j} \mathrel{+}= \sum_{i \in \text{chunk}} \alpha^{(\ell)}_{h, i, j}
$$

Never instantiate the full softmax matrix.

---

# Engineering Invariant 2 &mdash; `_seen_tokens`

After pruning, a naïve implementation sets
`cache._seen_tokens = pruned_length`. Then RoPE of new decode tokens
shifts back, **misaligning with the surviving prefill positions**.

**Fix.** Set `cache._seen_tokens` to the **original** prefill length, not
the pruned length:

```python
cache._seen_tokens = original_prefill_len   # not len(survivors)
```

This is silent on success, catastrophic on failure &mdash; one of the easiest
bugs to ship.

---

# Engineering Invariant 3 &mdash; Per-Head Heterogeneous Budgets

Different heads in one layer want different sets of kept positions.
Naïve fix: per-head $(K, V)$ tensors. **Kills batched matmul.**

**Our fix.** Layer-level kept set is the **union** of per-head sets;
per-head bool mask becomes a bias added inside attention after $QK^\top$:

$$
\text{logits}_{h, i, j} \mathrel{+}= \begin{cases} 0 & \text{if } j \in S_h \\ -\infty & \text{otherwise} \end{cases}
$$

Speed of batched matmul is preserved; semantics of per-head budgets is correct.

---

# Engineering Invariant 4 &mdash; Layer 0 = Max KV Length

HuggingFace's attention-mask machinery reads sequence length **from layer 0**.

If `LayerActivityPruner` allocates the smallest budget to layer 0, the mask
truncates everything else and decode produces garbage.

**Fix.** Force layer 0 to receive the **largest** budget; the pruner
re-sorts the per-layer allocation.

```python
order = argsort(scores, descending=True)   # decreasing budgets
allocation[order[0]], allocation[0] = allocation[0], allocation[order[0]]
```

<span class="muted">Plus a hand-rolled greedy-decode loop in `src/decode.py` that bypasses HF's
length-consistency assumption.</span>

---

# Experimental Setup

- **Model.** Llama-2-7B-chat, NF4 4-bit weights (`bitsandbytes`), fp16 KV cache.
- **Hardware.** Single Colab/Kaggle T4 (16 GB VRAM).
- **Library.** `transformers==4.46.3` (v5 refactored `DynamicCache`).
- **Datasets.**
  - GSM8K: 100 examples, 8-shot CoT, EM on `#### N`.
  - LongBench: NarrativeQA, Qasper, GovReport, HotpotQA &mdash; 30 / task,
    middle-truncated to 3.5 k tokens, official metrics.
- **Sweep.** `keep_ratio` $\in \{0.10, 0.25, 0.50, 0.75\} \times$ 5 strategies $\times$ 5 tasks
  $= 100$ runs + 5 no-pruning baselines.
- **Reported.** Task metric, **realised KV bytes** (not nominal ratio), prefill peak memory.

---

# Hypotheses (To Confirm or Refute)

1. `token_attention` $\geq$ `token_recent_distant` for all tasks at the
   same `keep_ratio`; gap **widest on long-context** tasks where the
   answer-bearing token sits mid-prompt.
2. `layer_activity` is **best at aggressive ratios** ($\leq 0.25$) &mdash; can
   cut deep layers harder while preserving shallow ones.
3. `head_activity` Pareto-dominates uniform `token_attention` on
   **realised bytes** &mdash; union design lets some heads keep many positions
   while others keep few.
4. `combined` wins on **GSM8K at aggressive ratios** &mdash; reasoning chains
   need both attention selection and per-head specialisation.

---

# Results &mdash; Accuracy vs Keep Ratio &nbsp; <span class="muted">[placeholder]</span>

<div style="border: 2px dashed #aaa; padding: 80px; text-align: center; color: #999; margin: 20px 60px;">
[ figure: accuracy-vs-keep_ratio per task, five strategies overlaid ]
<br><br>
generated by <code>scripts/make_figures.py</code>
</div>

| Strategy | 0.10 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| `token_recent_distant` | TODO | TODO | TODO | TODO | <span class="muted">baseline</span> |
| `token_attention`      | TODO | TODO | TODO | TODO | |
| `head_activity`        | TODO | TODO | TODO | TODO | |
| `layer_activity`       | TODO | TODO | TODO | TODO | |
| `combined`             | TODO | TODO | TODO | TODO | |

GSM8K exact match. Filled after Colab sweep.

---

# Results &mdash; KV Bytes vs Keep Ratio &nbsp; <span class="muted">[placeholder]</span>

<div style="border: 2px dashed #aaa; padding: 80px; text-align: center; color: #999; margin: 20px 60px;">
[ figure: realised KV cache MB vs keep_ratio, all strategies ]
</div>

Why this matters: nominal `keep_ratio` and **realised bytes** diverge for
per-head and per-layer strategies. Uniform strategies sit on the diagonal;
others can spend their budget unevenly.

---

# Results &mdash; Pareto Front &nbsp; <span class="muted">[placeholder]</span>

<div style="border: 2px dashed #aaa; padding: 80px; text-align: center; color: #999; margin: 20px 60px;">
[ figure: accuracy vs realised KV bytes, scatter, strategy=colour, ratio=marker size ]
</div>

This is the **headline plot**: which strategy gives the best accuracy at
a given memory budget. Up-and-to-the-left wins.

---

# Results &mdash; Attention Map Case Study &nbsp; <span class="muted">[placeholder]</span>

<div class="two-col">

<div style="border: 2px dashed #aaa; padding: 60px; text-align: center; color: #999;">
[ before: layer-16 attention received ]
</div>

<div style="border: 2px dashed #aaa; padding: 60px; text-align: center; color: #999;">
[ after: same, post token_attention @ ratio=0.25 ]
</div>

</div>

NarrativeQA single example. The "after" panel should show concentrated
mass on retained positions; sink tokens (front) and recent window (back)
are visible.

---

# Discussion (Pending Numbers)

- Confirm or refute each hypothesis with the actual sweep data.
- Note where the four engineering invariants would have caused silent
  errors had we not pinned them.
- Quantify the gap between **nominal** and **realised** memory savings
  for non-uniform strategies.
- Limitations: small sample size (no variance reporting yet); single model
  family; no decode-time pruning baseline (e.g. H2O).

---

# Conclusion

- Five pruning strategies covering token, head, layer granularity, plus
  a two-stage cascade &mdash; all under one `keep_ratio` knob.
- Four engineering invariants that are necessary for correctness on
  HuggingFace Transformers but rarely written down.
- A reproducible Pareto-front evaluation on Llama-2-7B-chat over
  GSM8K + LongBench-4.

**Future work.** Adaptive per-instance ratios (entropy proxy);
combine with KV quantisation (KIVI / KVQuant); trainable pruning policies.

---

<!-- _class: lead -->

# Thank you

## Questions?

<span class="muted tiny">Code: <code>D:/course_research/NLP/project/</code> &nbsp; | &nbsp; Pinned: <code>transformers==4.46.3</code> &nbsp; | &nbsp; Tests: 12/12 CPU</span>
