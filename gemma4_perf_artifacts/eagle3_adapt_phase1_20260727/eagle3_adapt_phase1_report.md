# Gemma4 31B EAGLE3 图模式适配 — Phase 1 报告

- 日期：2026-07-27
- 代码：`/home/vllm-ascend_tp/vllm-ascend_tp`（分支 `gemma4-eagle3-graph`，基线 commit `a62f90757`）
- 模型：Target `/home/xty/gemma4/31B`（bf16），Draft `/home/xty/eagle3/gemma-4-31b-it-eagle3`（Eagle3DraftModel）
- TP=4（Target），Draft TP=1，文本 only，低并发（max_num_seqs=8），短上下文（max_model_len 8192）

## 结论：EAGLE3 图模式适配成功（PIECEWISE）

Gemma4 31B EAGLE3 在 **PIECEWISE 图模式**下跑通：greedy token 与 Target-only 完全一致、5 轮 20 请求稳定、接受率正常（~45-80%，mean acceptance length 3.40）。仅 1 处代码修改（Mod 1，对齐上游）。

## 修改记录

### Mod 1（已合入，对齐上游）— `vllm_ascend/spec_decode/llm_base_proposer.py`
- **假设**：EAGLE3 draft 加载时 `llm_base_proposer.load_model` 对多模态模型无条件复制 `model.config.image_token_index`，但 `Gemma4Config` 用 `image_token_id`（无 `image_token_index`）→ `AttributeError`。
- **对照上游**：上游 vllm `v1/spec_decode/llm_base_proposer.py` 已把 `Gemma4ForConditionalGeneration`/`Gemma4UnifiedForConditionalGeneration` 加入 VL 名单，映射 `image_token_index = target_model.config.image_token_id`。vllm-ascend 版本较旧缺此条目。
- **改动**：在 vllm-ascend 同位置补这两个 arch（+5 行）。
- **before**：`AttributeError: 'Gemma4Config' object has no attribute 'image_token_index'`（`logs/eagle3_eager_before_mod1.log`）。
- **after**：draft 正常加载，EAGLE3 eager 启动成功。

### Mod 2（已回退，方向错误）— `vllm_ascend/attention/utils.py`
- **假设**：`using_paged_attention` 在 `speculative_config is not None` 时返回 False，强制 512-dim 走 FIA-TND（561002）。把 512-dim PA 回退移到 spec guard 之前，让 gemma4 在 spec 下用 PA。
- **结果**：FDO 捕获成功（33/33），但**输出乱码且不稳定** —— PA 是 decode-only，与 spec decoding 的多 token target forward 不兼容（PA graph replay 不处理 draft/bonus token 形状）。`using_paged_attention` 的 spec guard 正是为此而设。已回退。

## Phase 1.1 — Target-only BF16 图模式 TP4（FDO）✅
- 命令：`scripts/start_target_tp4.sh`（FULL_DECODE_ONLY，无 spec）。`--max-model-len 8192 --block-size 128`（gemma4 必需）。
- 结果：FULL_DECODE_ONLY 捕获成功；greedy 正确（Paris / 1-10 / ML 解释 / 多语言 hello）；2 轮稳定。
- 参考 token：`token_compare/target_greedy.json`。
- 备注：无 spec 时 `using_paged_attention` 对 512-dim 返回 True → 走 PA → TP4 下 PA 捕获成功。

## Phase 1.2 — 通用 EAGLE3（eager）✅（Mod 1）
- 命令：`scripts/start_eagle3_eager.sh`（`--enforce-eager`，spec on）。
- 结果（Mod 1 后）：draft 加载成功；greedy 与 Target 参考一致；2 轮稳定；接受率 75.8% / 51.5%（mean acceptance length 3.27）。
- 日志：`logs/eagle3_eager_after_mod1.log`；接受率：`token_compare/eagle3_eager_acceptance.txt`。

## Phase 1.3 — EAGLE3 + 图模式

### FULL_DECODE_ONLY ❌（被 CANN FIA-TND 512-dim 限制阻断）
- spec on → `using_paged_attention` 返回 False → 512-dim 全局头走 FIA-TND。
- `aclnnFusedInferAttentionScoreV3` 561002：**"input_layout is TND, only headDim = 64/128/192 supported, but got 512"**（CANN 限制）。
- eager（Phase 1.2）用 BNSD 布局能跑 512-dim；FULL_DECODE_ONLY 编译路径强制 TND → 崩。
- 低并发（max_num_seqs 8）仍崩 → 非 shape 问题，是 TND 布局硬限制。
- Mod 2（强制 PA）能捕获但乱码（PA+spec 不兼容）。
- 日志：`logs/eagle3_fdo_before_mod2.log`。

### PIECEWISE ✅（适配成功）
- 命令：`scripts/start_eagle3_piecewise.sh`（`cudagraph_mode:PIECEWISE`，spec on，max_num_seqs 8）。
- 结果：PIECEWISE 捕获成功（7/7，15s）；greedy 与 Target 参考完全一致（ALL MATCH）；5 轮 20 请求稳定（STABILITY PASS）；接受率 80% / 44.9%（mean acceptance length 3.40）。
- 原因：PIECEWISE 编译路径不强制 TND for 512-dim（走 BNSD，与 eager 一致），绕开 CANN 561002。
- 证据：`logs/eagle3_piecewise_after.log`、`token_compare/eagle3_piecewise_greedy.json`、`token_compare/eagle3_piecewise_acceptance.txt`、`token_compare/compare_target_vs_eagle3.txt`。

## 功能门槛核对
| 门槛 | 结果 |
|---|---|
| greedy token 完全一致（spec-on vs spec-off） | ✅ ALL MATCH（`compare_target_vs_eagle3.txt`） |
| 接受率正常 | ✅ ~45-80%，mean acceptance length 3.40 |
| 连续请求稳定 | ✅ 5 轮 × 4 prompt = 20 请求全一致 |

## 关键结论
- **Gemma4 EAGLE3 图模式 = PIECEWISE**（不是 FULL_DECODE_ONLY）。FULL_DECODE_ONLY 被 gemma4 512-dim 全局头 + CANN FIA-TND（TND 仅 64/128/192）阻断，且 PA 与 spec 不兼容。PIECEWISE 编译路径用 BNSD 支持 512-dim，是当前构建下唯一可用的 EAGLE3 图模式。
- 唯一代码改动：Mod 1（对齐上游，补 gemma4 arch 到 VL image_token_index 名单）。
- 性能优化待正确性通过后进行（下一阶段）。

## 复现（EAGLE3 + PIECEWISE 图模式）
```bash
cd /home/vllm-ascend_tp/vllm-ascend_tp
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=256 VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve /home/xty/gemma4/31B --host 0.0.0.0 --port 8000 \
  --served-model-name gemma4-eagle3 --tensor-parallel-size 4 --max-num-seqs 8 \
  --max-model-len 8192 --block-size 128 \
  --speculative-config '{"model":"/home/xty/eagle3/gemma-4-31b-it-eagle3","method":"eagle3","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
# 验证（需 NO_PROXY=127.0.0.1 绕过 http_proxy）：
python3 scripts/greedy_token_test.py --port 8000 --model gemma4-eagle3 --rounds 5 --max-tokens 128
```
