# Gemma4 性能/特性验证产物

本目录存放 Gemma4 (dense 31B / MoE 26B-A4B) 在 Ascend 910B3 上的特性验证与量化性能基线产物。

## 子目录
- `feature_verify_20260711/` — 5 特性（chunked prefill / prefix cache / weight_nz / async scheduling / CPU binding）在 31B 与 26B-MoE 上的支持验证（eager 模式）+ flashcomm1 调查。报告：`reports/gemma4_vllm_ascend_feature_report.md`、`reports/flashcomm1_verification_report.md`、`reports/fdo_cudagraph_capture_rootcause.md`。
- `feature_graphmode_piecewise_20260723/` — 5 特性 × **PIECEWISE 图模式**（TP2）叠加验证。报告：`reports/feature_graphmode_piecewise_report.md`。
- `feature_graphmode_fdo_20260723/` — 5 特性 × **FULL_DECODE_ONLY 图模式**（TP4）叠加验证 + 三模式汇总。报告：`reports/feature_graphmode_fdo_report.md`、`reports/gemma4_feature_graphmode_consolidated.md`。
- `dense_w8a8_baseline_20260713/` — 31B Dense W8A8 (`/home/quant_ms/gemma4_w8a8`) TP1 PIECEWISE 基线。报告：`reports/quant_tp1_piecewise_baseline.md`。文本 TPOT 37.1ms / 图片 36.0ms。
- `moe_w8a8_baseline_20260713/` — 26B MoE W8A8 only-experts (`/home/wangminghua/gemma4_moe_w8a8_only_experts`) TP1 PIECEWISE 基线。报告：`reports/quant_moe_tp1_piecewise_baseline.md`。文本 TPOT 22.2ms / 图片 22.9ms。

## 关键结论
- **5 特性 + chunked+prefix 组合** 在 eager / PIECEWISE(TP2) / FULL_DECODE_ONLY(TP4) 三种图模式下均可叠加（42 case 全 OK），未改 vllm-ascend 代码。汇总见 `feature_graphmode_fdo_20260723/reports/gemma4_feature_graphmode_consolidated.md`。
- **量化基线**：两份 W8A8 权重均用 v1 + TP1 + 显式 PIECEWISE 跑通，两组评测 TPOT ≤ 50ms。
- gemma4 512-dim 全局头：TP1/TP2 下 FULL/FULL_DECODE_ONLY 走 PA 捕获崩（ACL 107033），FIA-TND 不支持 512-dim（561002）；解法 = 显式 PIECEWISE（TP2 可用）或 FULL_DECODE_ONLY + TP4（拆 head_dim 到 128）。v0 引擎在 vllm 0.23.0 已移除。
- 复现命令见各子目录 `scripts/` 与 `reports/`。

