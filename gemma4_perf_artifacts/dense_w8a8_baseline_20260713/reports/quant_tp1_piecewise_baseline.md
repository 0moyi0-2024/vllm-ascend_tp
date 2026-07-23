# Gemma4 31B W8A8 量化 — TP1 PIECEWISE 图模式性能基线

- 日期：2026-07-13
- 权重：`/home/quant_ms/gemma4_w8a8`（W8A8，已修复 lm_head，2004 描述键 / 1999 张量）
- 路径：`/home/vllm-ascend_tp/vllm-ascend_tp`（vllm 0.23.0 + vllm-ascend @ a83b055e）
- 启动脚本：`scripts/start_quant_tp1_piecewise.sh`

## 关键结论

TP1 单卡，用 **显式 PIECEWISE 图模式** 达成两组评测 TPOT ≤ 50ms：

| 评测 | 输入→输出 | TPOT | TTFT | Decode tok/s | 成功率 |
|---|---|---|---|---|---|
| 文本（random） | 4096→1536 | **37.1ms** ✅ | 1000ms | 26.93 | 100% |
| 图片（random_vl, 1920×1080） | 297→256 | **36.0ms** ✅ | 458ms | 27.77 | 100% |

与之前测试（rollback 权重, v0/0.20.2, PIECEWISE）的 39.1ms / 37.0ms 基本一致。

## 启动命令

```bash
cd /home/vllm-ascend_tp/vllm-ascend_tp
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=256 VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /home/quant_ms/gemma4_w8a8 \
  --served-model-name gemma4_w8a8 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 --max-model-len 6144 --port 8080 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
```

PIECEWISE 捕获成功：`Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100% 35/35`，`Graph capturing finished in 18 secs`。

## 根因与方法（未改代码）

1. **v0 不可用**：vllm 0.23.0 报 `Unknown vLLM environment variable detected: VLLM_USE_V1`，v0 引擎已移除。之前测试用的 v0 路径走不通。
2. **默认图模式失败**：TP1 不指定 `--compilation-config` 时默认 `FULL_AND_PIECEWISE`，其 FULL decode 部分在 gemma4 512-dim 全局头走 PagedAttention（`full_graph_pa` → `graph_task_group`）捕获时崩溃（ACL 107033 "task group status error"）。`FULL_DECODE_ONLY` 同样在 PA 捕获失败。
3. **PIECEWISE 显式指定可绕过**：`--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` 走 piecewise 捕获（按 attention 边界切分），不触发 `full_graph_pa` 的 PA `graph_task_group` 路径，35 个图全部捕获成功。
4. **关于 force-disable**：`vllm_ascend/platform.py:61`（commit 028e538c）`envs_vllm.VLLM_USE_BREAKABLE_CUDAGRAPH=False` 只是禁用"breakable cudagraph"的**自动启用**（针对 DeepSeek V4 的 workaround），**不阻止显式 `cudagraph_mode:PIECEWISE`**——显式指定仍然生效。所以无需改代码，显式 PIECEWISE 即可用。
5. **eager 不达标**：TP1 eager 模式 generation 仅 6.5 tok/s（TPOT ~154ms），远超 50ms；必须图模式。图模式下 26.5 tok/s（TPOT ~37ms）。

## 评测脚本
- 文本：`scripts/evalpertext.sh`（evalscope perf, parallel 1, number 10, 4096→1536, ignore_eos）
- 图片：`scripts/evalperimage.sh`（evalscope perf, parallel 1, number 12, 1920×1080, 30→256）

## 原始日志
- 服务：`logs/server_pw.log`（PIECEWISE 捕获成功）
- 文本评测：`logs/eval_text.log` + `logs/eval_text_summary.txt`
- 图片评测：`logs/eval_image.log` + `logs/eval_image_summary.txt`

## 注意
- `--gpu-memory-utilization 0.92`（KV cache 16.87 GiB，6144 tok 并发 5.11x）。之前测试用 0.70 在 0.23.0 上 KV 仅 3.46 GiB，放不下 4096→1536 请求；改为 0.92。
- 之前测试的"PIECEWISE 自动选择"依赖 v0/0.20.2；在 0.23.0 v1 上需**显式** `--compilation-config '{"cudagraph_mode":"PIEWISE"}'`。
