# Gemma4 26B MoE — weight-layout 修复分支验证 Wiki

- 日期：2026-07-22
- 测试分支：`fix-gemma4-moe-weight-layout`（`origin/fix-gemma4-moe-weight-layout`）
  - 基线：`ce229dc2d [Feature] Support AlltoAll + EP + LoRA in non‑quantized mode (#12451)`（在 upstream `main` `44fc51ffb` 之上）
  - 修复 commit：`f0a7948d7 fix: preserve unquantized MoE weight stride layout`
  - 改动文件：`vllm_ascend/ops/fused_moe/fused_moe.py`、`vllm_ascend/_310p/fused_moe/fused_moe.py` + 2 个测试
- 目的：验证 MoE 权重 stride 布局修复是否能消除 [[Gemma4 26B MoE 重复退化分析报告]](gemma4_26b_moe_repetition_analysis.md) 中的重复 loop 问题。

## 1. 结论

**`f0a7948d7` weight-layout 修复未解决重复 loop 问题**，loop 率与准确率均与未修复时基本一致。但更深入的数据拆分揭示了一个更重要的结论：

> **26B MoE 的"非 loop"样本准确率高达 ~80%，反超 31B dense 的 76.8%。模型推理能力本身没问题，重复 loop 是唯一拖累总分的因素。** 若能消除 loop，26B 总分可升至 ~80%。

- weight-layout 分支：loop 率 27.3%、总 acc 0.601；其中 loop 样本 acc 仅 9.3%，非 loop 样本 acc 79.2%
- 之前 main+PR12530：loop 率 30.8%、总 acc 0.5859；loop 样本 acc 8.2%，非 loop 样本 acc 81.0%
- 两次 run 的 loop 题有 **70.4% 重叠**（54 题中 38 题两次都 loop）——**特定输入稳定触发**，指向确定性的 MoE routing/expert bug，而非随机采样退化

两条 MoE 修复线（PR12530 routing indices、f0a7948d7 weight-layout）均未触及 loop 根因。服务端评测期间零报错。

## 2. 关键数据拆分：loop vs 非 loop 样本

将每条样本按是否触达 `max_tokens`（8192）拆分为 loop / clean 两组（31B 按 32768 阈值，实际 0 触顶）：

| run | loop 率 | loop 样本 acc | **非 loop 样本 acc** | 总 acc |
|---|---|---|---|---|
| 26B main+PR12530（temp 0.95） | 30.8%（61/198） | 8.2%（5/61） | **81.0%（111/137）** | 0.5859 |
| **26B weight-layout（temp 1.0）** | **27.3%（54/198）** | **9.3%（5/54）** | **79.2%（114/144）** | **0.601** |
| 31B dense（temp 0.95） | 0%（0/198） | — | **76.8%（152/198）** | 0.7677 |

要点：
- **loop 样本几乎全错**（acc 8-9%，低于 4 选 1 随机 25%——因 loop 样本常连答案字母都没输出），loop 一旦发生基本判死。
- **非 loop 样本 acc ~80%**，两次 26B run 一致，且**高于 31B dense 的 76.8%**——26B MoE 的有效推理能力并不弱，甚至略强。
- 因此 26B 总分低（0.59-0.60）**完全由 loop 率（~28-31%）造成**，而非推理能力不足。loop 是唯一杠杆。

## 3. loop 触发的输入稳定性

两次 26B run（temp 0.95 vs 1.0、PR12530 vs weight-layout 两条不同修复线）的 loop 题重叠：

| | run0（main+PR12530） | run1（weight-layout） |
|---|---|---|
| loop 题数 | 61 | 54 |
| 两次都 loop 的题 | 38 | 38 |
| 仅本次 loop | 23 | 16 |
| Jaccard 相似度 | 0.494 | |
| 占 run1 loop 题比例 | — | 70.4% |

**70% 的 loop 题在不同温度、不同 MoE 修复下都稳定触发**。若是高温随机采样退化，重叠率应很低；高重叠表明这些输入**确定性地**驱动 26B MoE 进入重复——强烈指向某个 MoE 路径（routing/expert 选择/merge）在特定 token 模式下产生退化输出的 bug。这批 ~38 个"必 loop 题"可作为定位 bug 的最小复现集。

## 4. 测试配置

- 服务：`/home/xty/gemma4/26B`，TP4 + `--enable-expert-parallel` + `FULL_DECODE_ONLY`，devices 4-7
- 服务启动无报错，KV cache 41.32 GiB / 14.79x 并发；fused_moe.py 改动触发 torch.compile 重编译（正常）
- 评测：evalscope，GPQA Diamond 全量 198，0-shot，`max_tokens=8192`，**`temperature=1.0`**，top_p 0.95，top_k 64，thinking 关闭，batch 8，seed 42

## 5. 最终数据（198/198 完成）

- **mean_acc = 0.601（60.1%）**，n=198
- 触顶 8192 的 **54 条（27.3%）**，触顶样本为退化性重复（`The resistance of 0%` / `The endocyclic.` / `log, log, log` / `Let X.` / `...The...` 等短语重复至 max_tokens）
- 非 loop 样本（144 条）输出中位 797 token，集中于 < 2k（107 条在 500-1k）
- 平均延迟 65.6s（max 196.3s），TPOT 23.12ms，吞吐 43.0 tok/s
- 评测日志：`outputs/gpqa_26b_weightlayout_run.log`；服务日志：`outputs/serve26b_weightlayout.log`

## 6. 跨修复线对比总表

| 分支 / 修复 | temp | max_tokens | n | 总 acc | loop 率 | 非 loop acc |
|---|---|---|---|---|---|---|
| main（无修复） | 0.95 | 32768 | 10 | 0.8 | 2/10 | — |
| main + PR12530（routing indices） | 0.95 | 8192 | 198 | 0.5859 | 30.8% | 81.0% |
| **weight-layout（f0a7948d7）** | **1.0** | **8192** | **198** | **0.601** | **27.3%** | **79.2%** |
| 31B dense 对照 | 0.95 | 32768 | 198 | 0.7677 | 0% | 76.8% |

两条 MoE 修复线（PR12530 routing indices、f0a7948d7 weight-layout）均未降低 loop 率。26B 非 loop acc（~80%）反超 31B（76.8%）——loop 是唯一短板。

## 7. 根因推测（更新）

1. **确定性 MoE 路径 bug**（最可能）：70% loop 题跨温度/跨修复稳定触发，符合"特定输入→特定 routing→退化输出"的确定性 bug 特征，而非随机采样退化。PR12530（unpermute 索引）和 weight-layout（权重 stride）都未命中，bug 可能在 moe_block 内的 permute/merge、expert 选择或 top-k 路由的某个 Ascend 算子路径。
2. 已排除：权重 stride 布局错误（本次验证）、unpermute 索引丢失（PR12530 已修但仍 loop）、纯随机采样退化（重叠率不支持）。
3. 待排除：模型固有（小 active 容量）退化——但 31B dense 同配置零 loop、26B 非 loop acc 更高，不支持"26B 推理弱"的解释。

## 8. 待办

- [ ] 用 ~38 个"必 loop 题"作最小复现集，`ASCEND_LAUNCH_BLOCKING=1` 抓 MoE routing/expert 命中分布，对比 loop token vs 正常 token 的路由差异
- [ ] 排查 moe_block 内 permute/unpermute/merge 及 top-k 路由的 Ascend 算子路径
- [ ] 生成侧缓解验证（仅用于隔离根因，非根治）：加 `repetition_penalty=1.1~1.2` 或开启 thinking，看 loop 率能否压到 0
- [ ] 若生成侧能压到 0 且 acc 升至 ~80%，则确认根因为 MoE 路径 bug 而非模型固有

## 9. 复现

```bash
git checkout fix-gemma4-moe-weight-layout
# 服务
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=256 \
vllm serve /home/xty/gemma4/26B \
  --served-model-name gemma-4-26B-A4B-it \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":2,"audio":1,"video":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
# 评测关键参数
# datasets=gpqa_diamond(full), temperature=1.0, top_p=0.95, top_k=64,
# max_tokens=8192, thinking=disabled, batch=8, seed=42
# 脚本：outputs/eval_gpqa_26b_weightlayout.py
```
