# Gemma4 26B MoE 重复 loop 根因定位 — FIA decode mask

- 日期：2026-07-22
- 状态：**根因已定位**
- 相关：[[重复退化分析报告]](gemma4_26b_moe_repetition_analysis.md)、[[weight-layout 验证 wiki]](gemma4_26b_weight_layout_branch_wiki.md)

## 1. 结论（TL;DR）

**上游 `070825300 [BugFix] Fix FIA decode mask handling on main (#11266)` 是 Gemma4 26B MoE GPQA 重复 loop 的根因。** 在 `validate-gemma4-fia-decode-mask` 分支回退该修复后：

- **loop 率从 ~27-31% 暴跌到 2.5%**（54/198 → 5/198）
- **总 acc 从 0.59-0.60 跃升到 0.7222**，接近 31B dense 的 0.7677
- 平均延迟 65.6s → 27.0s（不再 ramble 到 max_tokens）

之前验证的两条 MoE 修复线（PR#12530 routing indices、`f0a7948d7` weight-layout）均未命中根因，因为它们修的不是根因——根因在 **FIA decode mask 的 decode 注意力路径**。

## 2. 验证分支

`validate-gemma4-fia-decode-mask`（`origin/validate-gemma4-fia-decode-mask`）
- 基线：`ce229dc2d [Feature] Support AlltoAll + EP + LoRA in non‑quantized mode (#12451)`（upstream `main` `44fc51ffb` 之上）
- commit：`9a966880d [Test] Revert FIA decode mask handling for Gemma4 validation`
- 改动：`vllm_ascend/attention/attention_v1.py`（回退 #11266 的 FIA decode mask 处理，净 -23 行）
- 假设：上游 #11266 的 FIA decode mask 修复引入了 26B MoE 的退化 → 回退验证

## 3. 全量 GPQA 结果（temp 1.0, max_tokens 8192, 198 题, 0-shot, batch 8, seed 42）

| 指标 | **fia-decode-mask（回退）** | weight-layout | main+PR12530 |
|---|---|---|---|
| **总 acc** | **0.7222** | 0.601 | 0.5859 |
| **loop 率（达 8192）** | **2.5%（5/198）** | 27.3%（54/198） | 30.8%（61/198） |
| loop 样本 acc | 0%（0/5） | 9.3% | 8.2% |
| 非 loop 样本 acc | 74.1%（143/193） | 79.2% | 81.0% |
| 平均输出 tokens | 1273（中位 914） | 2819 | 3079 |
| 平均延迟 | 27.0s | 65.6s | 70.2s |
| 服务端报错 | 无 | 无 | 无 |

## 4. 跨分支对比总表（含 31B 对照）

| 分支 / 修复 | temp | loop 率 | 总 acc | 非 loop acc |
|---|---|---|---|---|
| main（无修复，前 10 题） | 0.95 | 2/10 | 0.8 | — |
| main + PR12530（routing indices） | 0.95 | 30.8% | 0.5859 | 81.0% |
| weight-layout（f0a7948d7） | 1.0 | 27.3% | 0.601 | 79.2% |
| **fia-decode-mask 回退（9a966880d）** | **1.0** | **2.5%** | **0.7222** | **74.1%** |
| 31B dense 对照 | 0.95 | 0% | 0.7677 | 76.8% |

回退 FIA decode mask 是唯一显著降低 loop 率的改动。loop 率 2.5% 已接近 31B 的 0%，总分 0.7222 也接近 31B 的 0.7677。

## 5. 根因机制推测

- 上游 #11266 "Fix FIA decode mask handling" 修改了 decode 阶段 FIA（Full Instruction Attention）的 mask 处理逻辑。
- 该修改在 26B MoE 的特定输入下导致 decode 注意力退化，token 陷入短语重复（`ANSWER: B`×1764、`Looking at.`×506 等）直至 max_tokens。
- 这与早期观察一致：70% loop 题跨温度/跨 MoE 修复稳定触发（确定性 bug 特征），且仅 26B MoE 出现、31B dense 零 loop。
- 回退后非 loop acc 从 ~80% 略降到 74.1%——说明 #11266 的 mask 改动对部分正常题的 decode 注意力也有影响（可能 #11266 修的是某个真实问题，但实现引入了 26B MoE 的退化）。net effect 强烈正向（+0.12 总分）。

## 6. 残留问题

- 回退后仍有 2.5%（5/198）loop，可能是 #11266 之外的小问题，或回退不彻底。
- 非 loop acc 略降（74.1% vs 80%）：回退对部分正常题有副作用，需权衡。理想方案是**针对性修复 #11266 引入退化的部分**，而非整体回退——既保住 loop 修复又不损失正常题准确率。

## 7. 后续

- [ ] 对比 #11266 改动前后 `attention_v1.py` 的 FIA decode mask 差异，定位具体哪几行引入退化
- [ ] 尝试只回退/修正引入退化的部分，保留 #11266 的合理修复，验证能否同时达到低 loop 率 + 高非 loop acc
- [ ] 复查剩余 2.5% loop 题，确认是否同源
- [ ] 上游反馈：#11266 在 Gemma4 26B MoE 上引入重复退化，建议 revert 或修正

## 8. 复现

```bash
git checkout validate-gemma4-fia-decode-mask
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=256 \
vllm serve /home/xty/gemma4/26B \
  --served-model-name gemma-4-26B-A4B-it \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":2,"audio":1,"video":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
# 评测: gpqa_diamond(full), temp=1.0, top_p=0.95, top_k=64, max_tokens=8192,
# thinking=disabled, batch=8, seed=42  (脚本: outputs/eval_gpqa_26b_fia.py)
```

数据产物：
- 评测报告：`outputs/gpqa_26b_fia/<ts>/reports/gemma-4-26B-A4B-it/gpqa_diamond.json`
- 服务日志：`outputs/serve26b_fia.log`；评测日志：`outputs/gpqa_26b_fia_run.log`
