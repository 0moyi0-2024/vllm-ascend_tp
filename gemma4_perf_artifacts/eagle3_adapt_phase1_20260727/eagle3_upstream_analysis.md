# Gemma4 31B EAGLE3 图模式适配 — 上库分析与实现方案

- 日期：2026-07-27
- 仓库：`vllm-ascend`（分支 `gemma4-eagle3-graph`）
- 模型：Target `/home/xty/gemma4/31B`（bf16），Draft `/home/xty/eagle3/gemma-4-31b-it-eagle3`（Eagle3DraftModel）
- 环境：Ascend 910B3 ×8，vllm 0.23.0，vllm-ascend dev

---

## 1. 问题

Gemma4 31B + EAGLE3 spec decoding 在 vllm-ascend 上无法启动：

```
File ".../vllm_ascend/spec_decode/llm_base_proposer.py", line 374, in load_model
    self.model.config.image_token_index = model.config.image_token_index
AttributeError: 'Gemma4Config' object has no attribute 'image_token_index'
```

## 2. 根因

`AscendSpecDecodeBaseProposer.load_model` 对多模态 target 模型设置 draft 的 `image_token_index`。该逻辑按模型 arch 名匹配：
- 命中名单 → `image_token_index = model.config.image_token_id`
- 未命中 → `else: image_token_index = model.config.image_token_index`（假设 target 有该属性）

Gemma4 是多模态（`Gemma4ForConditionalGeneration`），但其 config 用 `image_token_id`（无 `image_token_index`），且不在名单中 → 走 else → `AttributeError`。

**上游 vLLM 已修复**（`vllm/v1/spec_decode/llm_base_proposer.py:1371` 已含 `Gemma4ForConditionalGeneration`/`Gemma4UnifiedForConditionalGeneration`），vllm-ascend 版本未同步。

## 3. 修复（Mod 1 — 唯一代码改动）

`vllm_ascend/spec_decode/llm_base_proposer.py`，在 VL `image_token_index` 名单中补两个 gemma4 arch：

```diff
                 "Qwen3_5ForConditionalGeneration",
                 "Qwen3_5MoeForConditionalGeneration",
                 "Step3p7ForConditionalGeneration",
+                # Align with upstream vLLM: Gemma4 VL models use image_token_id
+                # (not image_token_index), so map it explicitly here. Without
+                # this the else-branch raises AttributeError on Gemma4Config.
+                "Gemma4ForConditionalGeneration",
+                "Gemma4UnifiedForConditionalGeneration",
             ]:
                 self.model.config.image_token_index = model.config.image_token_id
```

**+5 行，纯新增，0 删除。** 与上游 vLLM 行为完全对齐。通用（不限 NPU，所有 gemma4 + EAGLE3 受益）。

## 4. 图模式选择（配置，非代码）

Gemma4 有 512-dim 全局注意力头。`using_paged_attention()` 在 `head_size==512` 时返回 True（PA 回退，因 FIA-TND 不支持 512-dim）。但 spec decoding 下 `using_paged_attention` 返回 False（spec guard），强制 FIA。

| 图模式 | spec off | spec on | 原因 |
|---|---|---|---|
| FULL_DECODE_ONLY | ✅（PA 捕获） | ❌ 561002 | spec 下走 FIA-TND，512-dim 不支持（CANN: TND 仅 64/128/192） |
| PIECEWISE | ✅ | ✅ | FIA 用 BNSD（支持 512-dim），捕获成功 |

**结论**：Gemma4 EAGLE3 图模式用 **PIECEWISE**（`--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`）。FULL_DECODE_ONLY 被 512-dim TND 阻断，需 CANN 支持 TND-512。

## 5. 正确性验证

### 5.1 启动命令
```bash
cd <vllm-ascend>
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=256 VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve /home/xty/gemma4/31B --host 0.0.0.0 --port 8000 \
  --served-model-name gemma4-eagle3 --tensor-parallel-size 4 --max-num-seqs 8 \
  --max-model-len 8192 --block-size 128 \
  --speculative-config '{"model":"/home/xty/eagle3/gemma-4-31b-it-eagle3","method":"eagle3","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
```
（`--max-model-len 8192 --block-size 128` 是 gemma4 必需：默认 256k OOMs；默认 block_size 16 → "No common block size"）

### 5.2 Greedy token 一致性（spec-on vs spec-off）
4 个 prompt，spec-on (EAGLE3+PIECEWISE) 与 spec-off (Target-only+PIECEWISE) greedy 输出 **完全一致**（ALL MATCH）：Paris / 1-10 / ML 解释 / 多语言 hello。

### 5.3 稳定性
5 轮 × 4 prompt = 20 次连续请求，greedy 输出完全一致 → STABILITY PASS。

## 6. 接受率

num_speculative_tokens=3，自然输出（128 token），稳态：

| pos0 | pos1 | pos2 | mean acceptance length | avg draft acceptance |
|---|---|---|---|---|
| ~0.71 | ~0.48 | ~0.30 | ~2.5 / 3 | ~50% |

典型 EAGLE3 衰减 pattern，draft 质量正常。

## 7. 性能收益

### 7.1 Speedup 曲线（spec-on vs spec-off，同 PIECEWISE，TP4，自然输出 128 tok）

| parallel | spec-on (tok/s) | spec-off (tok/s) | speedup |
|---|---|---|---|
| 1 | 28.1 | 24.6 | +14.2%（峰值） |
| 2 | 51.0 | 48.3 | +5.6% |
| 4 | 91.6 | 96.0 | ~flat |
| 8 | 188.1 | 186.0 | ~flat |
| 16 | 177.6 | 360.5 | -50.8% |
| 32 | 264.3 | 519.9 | -49.2% |

### 7.2 瓶颈分析（插桩实测）

- **draft forward = 5.7ms/步**（可忽略，非瓶颈）
- **target spec-verify = 125ms（4 token, FIA BNSD 512-dim, graph-captured）**
- 非 spec PA decode = 40.7ms（1 token）
- verify 4-token = 3.07× PA 1-token：FIA 512-dim attention 按 token 线性增长，低并发下 NPU overhead-bound 未摊销
- 高并发（p16+）target 饱和：4× verify 工作被 acceptance 3.5 抵不住 → -50%

### 7.3 论文 3× 的 CANN 依赖

论文（GPU）：4-token verify 摊销到 ~1-token（compute-bound）→ spec-step ≈ 46ms → 75 tok/s → 3×。
NPU：4-token verify = 125ms（不摊销）→ +14%。

**对齐 3× 需 CANN**：
1. FIA TND 支持 512-dim（解锁 FULL_DECODE_ONLY + BNSD→TND 提速）
2. FIA 512-dim 多 token 摊销（4-token verify ≈ 1-token 成本）

两项均为 CANN/内核级，非 vllm-ascend 代码可解。

## 8. 上库建议

| 项 | 上库 | 说明 |
|---|---|---|
| **Mod 1（image_token_index）** | ✅ | 对齐上游、必要、+5 行、通用 |
| PIECEWISE 配置 | 文档 | gemma4 EAGLE3 图模式推荐，非代码 |
| adaptive spec / draft 重叠 | ❌ | CANN workaround，Ascend-specific，收益小，上游会拒 |
| CANN 需求（TND-512 + 摊销） | 给 CANN 团队 | 3× 的真路径 |

**代码只上 Mod 1。** 其余为配置推荐 + CANN 需求。

## 9. 已尝试但未保留的优化

| 优化 | 结果 | 处理 |
|---|---|---|
| Mod 2（force PA under spec, `attention/utils.py`） | FDO 捕获成功但输出乱码（PA+spec+graph 不兼容） | 已回退 |
| `--async-scheduling` | 无改善（draft 仍串行） | 配置，未保留 |
| draft `draft_tensor_parallel_size=4` | 无改善（小 draft，TP4 通信抵消） | 配置，未保留 |
| `num_speculative_tokens=2` | 更差（每步接受 token 少） | 配置，未保留 |

## 10. 原始证据

- 代码 diff：`vllm_ascend/spec_decode/llm_base_proposer.py`（+5 行）
- 正确性：`token_compare/compare_target_vs_eagle3.txt`（ALL MATCH）
- 接受率：`token_compare/eagle3_piecewise_acceptance.txt`
- 性能曲线：`perf_opt/`（tn2/tn3 脚本 + server logs）
- 插桩：`perf_opt/server_prof.log`（[PROF_EAGLE3] draft 5.7ms）、`server_tgt_prof.log`（[PROF_TGT] verify 125ms）
- 详细报告：`eagle3_adapt_phase1_report.md`、`eagle3_perf_extreme_analysis.md`、`eagle3_perf_rootcause_deep.md`
