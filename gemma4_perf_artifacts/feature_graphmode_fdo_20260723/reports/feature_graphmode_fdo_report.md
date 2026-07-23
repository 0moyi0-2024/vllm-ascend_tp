# Gemma4 特性 × FULL_DECODE_ONLY 图模式 叠加验证报告

- 日期：2026-07-23
- 模型：Gemma4 31B (dense, bf16) / 26B-A4B (MoE, bf16)，**TP=4**，910B3
- 代码：`/home/vllm-ascend_tp/vllm-ascend_tp`（vllm 0.23.0 + vllm-ascend @ a83b055e），未改代码
- 目录：`/home/xty/gemma4_feature_fdo_20260723_071510/`

## 关键约束：FULL_DECODE_ONLY 必须 TP4

gemma4 有 512-dim 全局头。`using_paged_attention()` 在 `head_size==512` 时强制走 PagedAttention（`full_graph_pa` → `graph_task_group`），其图捕获在当前 ACL 下崩（107033）。**TP1/TP2 时 head_size 仍为 512 → FULL_DECODE_ONLY 不可用**（见 `fdo_cudagraph_capture_rootcause.md`）。

**TP4 把 head_dim 拆到 128**（512/4），不再触发 512-dim PA 回退，改走 FIA-TND（可捕获）→ FULL_DECODE_ONLY 图捕获成功（35/35）。

> 对照：FIA-TND 直接处理 512-dim 会 ACL 561002（不支持），所以只能靠 TP 拆到 ≤256。TP4 是 FULL_DECODE_ONLY 在 gemma4 上可用的最低 TP。

## 启动模板
```
vllm serve <model> --tensor-parallel-size 4 \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --limit-mm-per-prompt '{"image":2,"audio":1,"video":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --max-model-len 8192 --block-size 128 \
  <feature flags> --additional-config '<json>'
```
每 case `Capturing CUDA graphs (decode, FULL)` 35/35 成功（~37s）。

## 结论：5 特性 + chunked+prefix 组合 均可与 FULL_DECODE_ONLY（TP4）叠加

| 模型 | baseline | chunked | prefix | weight_nz | async | cpu_binding | chunked+prefix |
|---|---|---|---|---|---|---|---|
| 31B (TP4) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 26B MoE (TP4) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

14 case 全部：启动成功 + FULL_DECODE_ONLY 捕获成功 + 12/12 功能请求成功 + 配置生效证据齐全。

## 叠加生效证据（每项确认 FULL_DECODE_ONLY + 特性同时 active）

- **FULL_DECODE_ONLY**：每 case `cudagraph_mode: <CUDAGraphMode.FULL_DECODE_ONLY>` + `Capturing CUDA graphs (decode, FULL)` 35/35。
- **async scheduling**：31B_async `FULL_DECODE_ONLY` + `Asynchronous scheduling is enabled.`（兼容）。
- **CPU binding**：26B_cpubind `FULL_DECODE_ONLY` + `[cpu_bind_mode] mode=topo_affinity`（兼容）。
- **weight_nz**：26B_weightnz `FULL_DECODE_ONLY` + `weight_nz_mode value 2`（MoE 上兼容）。
- **chunked / prefix / chunked+prefix**：flag 生效 + FULL 捕获成功。

## 三种图模式对比（gemma4 bf16）

| 模式 | TP1/TP2 | TP4 | 说明 |
|---|---|---|---|
| eager (`--enforce-eager`) | ✅ | ✅ | 无图，最慢 |
| PIECEWISE（显式） | ✅（TP2） | ✅ | 按 attention 边界切分，避开 full_graph_pa |
| FULL_DECODE_ONLY | ❌（PA 107033） | ✅ | 需 TP4 拆 head_dim 到 128 |

## case 汇总（func 12/12 全 OK，TP4）

| case | lat_mean_s | case | lat_mean_s |
|---|---|---|---|
| 31B_baseline | 0.630 | 26B_baseline | 0.486 |
| 31B_chunked | 0.606 | 26B_chunked | 0.474 |
| 31B_prefix | 0.600 | 26B_prefix | 0.476 |
| 31B_weightnz | 0.636 | 26B_weightnz | 0.470 |
| 31B_async | 0.542 | 26B_async | 0.444 |
| 31B_cpubind | 0.599 | 26B_cpubind | 0.463 |
| 31B_chunked_prefix | 0.644 | 26B_chunked_prefix | 0.455 |

## 原始日志
- 每 case：`cases/<case>/`（command.sh, server.log, runner.log, func_verify.json, config_evidence.txt, metrics.prom, status.txt）
- 调度：`logs/run_all.log`（2 slot × TP4）；汇总：`reports/case_summary.tsv`

## 结论
gemma4 (31B/26B-MoE) 的 5 特性 + chunked+prefix 组合，**在 TP4 下全部可与 FULL_DECODE_ONLY 图模式叠加**，启动/捕获/请求均成功，未改代码。FULL_DECODE_ONLY 在 gemma4 上仅 TP4 可用（TP1/TP2 因 512-dim PA 捕获崩溃），这是与 PIECEWISE（TP2 可用）的主要差异。
