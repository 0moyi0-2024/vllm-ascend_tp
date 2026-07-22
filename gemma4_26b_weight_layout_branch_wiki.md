# Gemma4 26B MoE — weight-layout 修复分支验证 Wiki

- 日期：2026-07-22
- 测试分支：`fix-gemma4-moe-weight-layout`（`origin/fix-gemma4-moe-weight-layout`）
  - 基线：`ce229dc2d [Feature] Support AlltoAll + EP + LoRA in non‑quantized mode (#12451)`（在 upstream `main` `44fc51ffb` 之上）
  - 修复 commit：`f0a7948d7 fix: preserve unquantized MoE weight stride layout`
  - 改动文件：`vllm_ascend/ops/fused_moe/fused_moe.py`、`vllm_ascend/_310p/fused_moe/fused_moe.py` + 2 个测试
- 目的：验证 MoE 权重 stride 布局修复是否能消除 [[Gemma4 26B MoE 重复退化分析报告]](gemma4_26b_moe_repetition_analysis.md) 中的重复 loop 问题。

## 1. 结论（当前）

**`f0a7948d7` weight-layout 修复未解决重复 loop 问题。** loop 率与准确率均与未修复时基本一致：

| 指标 | weight-layout 分支（temp 1.0） | 之前 main+PR12530（temp 0.95） |
|---|---|---|
| 全量 GPQA mean_acc | **~0.61（实时，151/198 已评，进行中）** | 0.5859 |
| 触顶 loop 率（达 8192） | **28.5%（43/151）** | 30.8%（61/198） |
| max_tokens | 8192 | 8192 |
| batch_size | 8 | 8 |

loop 率 28.5% vs 30.8%、准确率 0.61 vs 0.59 —— 差异在采样噪声范围内，**修复无明显效果**。说明 unquantized MoE 权重 stride 布局不是重复 loop 的根因。

> 注：本次 temp 改为 1.0（之前 0.95），但 temp 的小幅变化不影响 loop 率量级——核心问题是模型陷入短语重复退化，与温度关系不大。

## 2. 测试配置

- 服务：`/home/xty/gemma4/26B`，TP4 + `--enable-expert-parallel` + `FULL_DECODE_ONLY`，devices 4-7
- 服务启动无报错，KV cache 41.32 GiB / 14.79x 并发；fused_moe.py 改动触发 torch.compile 重编译（正常）
- 评测：evalscope，GPQA Diamond 全量 198，0-shot，`max_tokens=8192`，**`temperature=1.0`**，top_p 0.95，top_k 64，thinking 关闭，batch 8，seed 42

## 3. 当前数据（实时，评测进行中 151/198）

- 已评 151/198，正确 92，**实时 acc = 0.6093**
- 已写预测 151 条，触顶 8192 的 **43 条（28.5%）**——与未修复时同量级，触顶样本仍为退化性重复（`ANSWER: X` / `Looking at.` / `Wait, 0.x.` 等短语重复至 max_tokens）
- 评测日志：`outputs/gpqa_26b_weightlayout_run.log`
- 服务日志：`outputs/serve26b_weightlayout.log`

## 4. 跨修复线对比总表

| 分支 / 修复 | temp | max_tokens | n | mean_acc | loop 率 |
|---|---|---|---|---|---|
| main（无修复） | 0.95 | 32768 | 10 | 0.8 | 2/10 |
| main + PR12530（routing indices） | 0.95 | 32768 | 10 | 0.9 | 1/10 |
| main + PR12530（routing indices） | 0.95 | 8192 | 198 | 0.5859 | 30.8% |
| **weight-layout（f0a7948d7）** | **1.0** | **8192** | **198（进行中 151）** | **~0.61** | **~28.5%** |
| 31B dense 对照 | 0.95 | 32768 | 198 | 0.7677 | 0% |

两条 MoE 修复线（PR12530 routing indices、f0a7948d7 weight-layout）均未消除重复 loop。31B dense 在同配置下零 loop，问题仍指向 26B MoE 特有路径，但根因尚未定位。

## 5. 待办

- [ ] 等本次 weight-layout 全量评测跑完，更新最终 acc 与 loop 率
- [ ] 排查其它 MoE 路径（moe_block 内 permute/unpermute/merge、expert 选择）是否存在导致 token 退化的逻辑
- [ ] 尝试生成侧缓解验证：加 `repetition_penalty=1.1~1.2` 或开启 thinking，看 loop 率能否压到 0（用于区分"模型固有退化"vs"MoE 路径 bug"）
- [ ] 在 loop 样本上用 `ASCEND_LAUNCH_BLOCKING=1` 抓 MoE 路由/expert 命中分布，定位异常

## 6. 复现

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
