# Gemma4 性能/特性验证产物

本目录存放 Gemma4 (dense 31B / MoE 26B-A4B) 在 Ascend 910B3 上的特性验证与量化性能基线产物。

## 子目录
- `feature_verify_20260711/` — 5 特性（chunked prefill / prefix cache / weight_nz / async scheduling / CPU binding）在 31B 与 26B-MoE 上的支持验证 + flashcomm1 调查。报告：`reports/gemma4_vllm_ascend_feature_report.md`、`reports/flashcomm1_verification_report.md`、`reports/fdo_cudagraph_capture_rootcause.md`。
- `dense_w8a8_baseline_20260713/` — 31B Dense W8A8 (`/home/quant_ms/gemma4_w8a8`) TP1 PIECEWISE 基线。报告：`reports/quant_tp1_piecewise_baseline.md`。文本 TPOT 37.1ms / 图片 36.0ms。
- `moe_w8a8_baseline_20260713/` — 26B MoE W8A8 only-experts (`/home/wangminghua/gemma4_moe_w8a8_only_experts`) TP1 PIECEWISE 基线。报告：`reports/quant_moe_tp1_piecewise_baseline.md`。文本 TPOT 22.2ms / 图片 22.9ms。

## 关键结论
- 两份量化权重均用 **v1 引擎 + TP1 + 显式 `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`** 跑通，两组评测 TPOT ≤ 50ms，未改 vllm-ascend 代码。
- v0 引擎在 vllm 0.23.0 已移除；TP1 默认/`FULL_DECODE_ONLY` 图模式在 gemma4 512-dim 全局头 PA 捕获时 ACL 107033 崩溃，需显式 PIECEWISE 绕过。
- 复现命令见各子目录 `scripts/` 与 `reports/`。
