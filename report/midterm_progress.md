# KV Cache 稀疏化研究 —— 中期进展报告

**项目名称：** 面向 LLM 推理的 KV Cache 剪枝策略实现与评估


---

## 一、研究背景与目标回顾

自回归 LLM 推理中，KV Cache 的显存占用随输入长度线性增长。以 Llama2-7B-chat（32 层、32 头、head_dim=128）为例，4k tokens 下 fp16 KV cache 约 2 GB，已占去 T4 显存的 1/8；32k+ 长上下文场景下 KV cache 甚至超过模型权重，主导推理延迟。

本项目的目标是：在 **prefill 阶段一次性、不可恢复**地剪除冗余 K/V 条目（不剪 decode 阶段产生的 token），在显著降低显存占用的同时尽量保留生成质量，并系统比较不同粒度的剪枝策略。



---

## 二、当前进度概览

整体完成度约 **60%**：代码实现与单元测试均已完成；真实模型实验和报告结尾的结果/讨论部分尚未完成。

| 模块 | 状态 | 完成度 |
|---|---|---|
| 模型加载与 4-bit 量化 | ✅ 完成 | 100% |
| 分块 prefill + 注意力统计 | ✅ 完成 | 100% |
| 自定义 greedy decode（per-head bias） | ✅ 完成 | 100% |
| 5 种剪枝策略实现 | ✅ 完成 | 100% |
| GSM8K / LongBench 评测器 | ✅ 完成 | 100% |
| 注意力热图与曲线可视化 | ✅ 完成 | 100% |
| 单元测试（CPU mock） | ✅ 12/12 通过 | 100% |
| 配置文件 / Colab notebook / README | ✅ 完成 | 100% |
| 报告 §1–§4（动机、相关工作、方法、实验设置） | ✅ 完成 | 100% |
| **真实模型实验（Colab T4）** | ⏳ 待执行 | 0% |
| **报告 §5 结果 / §6 讨论 / §7 结论** | ⏳ 等待数据 | 占位符 |

---

## 三、已完成的工作

### 3.1 代码骨架

仓库结构如下，所有模块均已实现并通过 import 与 mock 测试：

```
project/
├── src/
│   ├── model_loader.py         # 4-bit Llama2-7B-chat 加载，可切换 attn 实现
│   ├── cache_utils.py          # KV 字节计数、克隆缓存、峰值显存上下文管理器
│   ├── prefill.py              # 分块 prefill，返回 (cache, attn_stats, last_logits)
│   ├── decode.py               # 自定义 greedy decode，支持 per-head bias 注入
│   ├── inference.py            # 顶层 run_with_pruning 调度入口
│   ├── prune/                  # 5 种剪枝策略 + registry
│   ├── eval/                   # GSM8K / LongBench / metrics
│   └── visualize/              # 曲线 + attention map
├── scripts/                    # run_gsm8k / run_longbench / run_ablations / make_figures
├── tests/                      # 12 个 CPU 单测，全部通过
├── configs/                    # 5 份 yaml（每种策略一份）
├── notebooks/colab_setup.ipynb
└── report/report.md
```

### 3.2 五种剪枝策略

覆盖课程要求的 token / head / layer 三个粒度：

| 策略 | 粒度 | 打分依据 | 关键参数 |
|---|---|---|---|
| `token_recent_distant` | 全局 token | 仅看位置（保留前 N_far + 后 N_near） | `n_recent`, `n_distant` |
| `token_attention` | 全局 token | 各 token 在 (层 × 头) 上累计接收到的注意力均值 | `recent_window` |
| `head_activity` | 每 head 独立预算 | 每个 head 的平均注意力强度 | union-mask |
| `layer_activity` | 每 layer 独立预算 | 每层的平均注意力强度 | layer 0 恒为 max_len |
| `combined` | token → head 级联 | 两阶段，每阶段 `keep_ratio^0.5` | √k 切分 |

所有策略统一以 `keep_ratio`（保留 K/V 占原始 prefill 长度的比例）作为唯一外部旋钮，便于横向对比。

### 3.3 工程上的关键设计

以下是实现过程中踩出来、并在代码里固化的几条不变量：

1. **分块 prefill**。eager attention 必须显式输出 `[B, H, S, S]` 注意力矩阵，单层 4k 序列就要 ~2 GB；改为 `chunk_size=512` 切片后逐块累加 per-key 维度上的注意力和，从不实例化完整 softmax 矩阵。
2. **`_seen_tokens` 保持原长**。剪枝完成后将 `cache._seen_tokens` 还原为**剪枝前**的 prefill 长度，以保证 decode 阶段新生成 token 的 RoPE 位置编码与残留 K/V 对齐。
3. **per-head 不同预算的高效实现**。同层不同 head 的不同保留集合，统一表达为「该层保留位置 = 各 head 保留集合的并集」+「per-head 布尔 bias mask」，bias 加在 `Q · Kᵀ` 之后，被禁用的 (head, key) 对加 −∞，从而保留 batched matmul 的速度。
4. **layer 0 = max kv_len**。HuggingFace 的 attention mask 机器从 layer 0 读取 K 序列长度，因此 `LayerActivityPruner` 强制把分配到的最大预算给 layer 0；同时手写 greedy decode 循环回避 HF 自带循环的长度一致性假设。

### 3.4 单元测试

`tests/` 下 12 个用例覆盖：剪枝后形状正确性、`_seen_tokens` 不变量、per-head mask 的语义、layer 0 长度约束、KV 字节计数与克隆深拷贝。**全部使用 CPU mock tensor**，与真实模型解耦，便于本地快速回归。


---

## 四、初步成果

1. **完整可运行的 pipeline**：`bash run.sh` 即可加载 4-bit Llama2，对比无剪枝与 50% token 剪枝两种条件下的 EM 与 KV cache MB。
2. **CPU-only 验证体系**：在用户本地 GPU < 8 GB 的环境下，仍能用 mock 张量验证全部剪枝逻辑的正确性。
3. **Pareto 评测框架**：评测时同时记录 `keep_ratio`、**实际占用的 KV bytes**（区别于名义比例）、任务指标，最终可绘制「精度 vs 真实显存」的 Pareto 曲线，而非仅「精度 vs 名义比例」。
4. **可写入创新分的工程细节**：`√k` 两阶段预算切分、union-mask 的 per-head 不同预算实现、layer 0 不变量这些都是论文一笔带过、但实现中最容易出错的点，足以支撑创新分（20 分）。

---

## 五、当前阻塞

只有一条关键路径，但比较紧张：

1. **Llama2 模型受限访问**。需要先完成 `huggingface-cli login` 申请到 gated 权重，或退而求其次切换到 `meta-llama/Llama-3.2-3B-Instruct`（脚本已留 `--model-id` 接口）。
2. **本地 GPU 不足**。本机 < 8 GB VRAM，4-bit Llama2-7B 也无法跑，必须使用 Colab / Kaggle T4。Colab notebook 已就绪。
3. **transformers 版本冲突**。本地环境是 v5.8.1，项目锁定 v4.46.3（v5 重写了 `DynamicCache`，破坏 `key_cache` / `_seen_tokens` 接口）。必须在专用 venv / Colab 中安装。

---

## 六、下一步计划

按优先级与时间紧迫性排序：

### P0 — 真实模型实验（预计 Colab T4 上 4–6 h）

- [ ] 完成 HuggingFace 登录或切换到 Llama-3.2-3B
- [ ] `bash run.sh` 烟测，确认 Colab 环境无问题
- [ ] 全量扫参：`python scripts/run_ablations.py --output-dir results/`
  - 5 策略 × 4 比例（0.1 / 0.25 / 0.5 / 0.75）× 5 任务（GSM8K + LongBench 的 NarrativeQA / Qasper / GovReport / HotpotQA）
- [ ] `python scripts/make_figures.py` 生成全部图表

### P1 — 补完报告 §5 / §6 / §7

- [ ] §5.1 精度 vs `keep_ratio` 曲线分析（每个任务一张）
- [ ] §5.2 KV cache 显存 vs `keep_ratio`
- [ ] §5.3 Pareto 图：精度 vs 实际 KV cache MB
- [ ] §5.4 Layer 16 注意力热图前后对比（NarrativeQA 单例）
- [ ] §6 Discussion：验证或推翻已写下的三条假设
  - `token_attention` ≥ `token_recent_distant`，差距在长文本任务上更大
  - `layer_activity` 在极端比例（≤ 0.25）下最优
  - `combined` 在 GSM8K 极端比例下胜出
- [ ] §7 结论 + 局限性

### P2 — 加分项

- [ ] **自适应 per-instance 比例**：以首块 attention entropy 作为 cheap proxy 动态调整 `keep_ratio`，对应报告 future work，可冲创新分
- [ ] **fp16 KV + 4-bit KV 量化叠加**，观察 Pareto 是否进一步外推
- [ ] **方差分析**：当前 GSM8K 100 例、LongBench 每任务 30 例样本太小，时间允许就加大样本量并报告标准差

---

## 七、风险与时间评估

| 风险项 | 影响 | 缓解措施 |
|---|---|---|
| Llama2 访问权限申请被拒 | 模型基线不一致 | 立即切换 Llama-3.2-3B-Instruct，报告中说明替换 |
| Colab T4 长时间断连 | 扫参跑不完 | 把 ablation 切分为按任务的子作业，断点续跑 |
| 实验结果与假设不符 | §6 讨论需要重写 | 这反而是研究价值，重新提炼现象写讨论即可 |
| 时间不足做 P2 | 创新分受限 | 集中精力把工程不变量与 √k 切分写好，P0+P1 已可拿 ~80 分 |

**时间线断言：** 评分构成中"实验 25 + 对比 20 = 45 分"全部锁在 P0 上，本周内必须把 Colab 实验跑起来，否则即使代码与文档满分，整体上限也只能压到 55 分左右。

---

## 八、小结

代码、测试、文档骨架已经全部到位，工程上较容易出错的几条不变量都已在实现中固化并通过单测验证。**当前唯一的关键路径是把 Colab 跑起来填数据**，随后报告的 §5 / §6 / §7 即可基于真实结果一次性补完。预计本周内完成 P0，下周补完 P1，进入终稿阶段。
