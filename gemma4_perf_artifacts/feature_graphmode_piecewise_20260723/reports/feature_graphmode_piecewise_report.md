# Gemma4 特性 × PIECEWISE 图模式 叠加验证报告

- 日期：2026-07-23
- 模型：Gemma4 31B (dense, bf16) / 26B-A4B (MoE, bf16)，TP=2，910B3
- 代码：`/home/vllm-ascend_tp/vllm-ascend_tp`（vllm 0.23.0 + vllm-ascend @ a83b055e），未改代码
- 目录：`/home/xty/gemma4_feature_graphmode_20260723_062949/`

## 背景

之前 5 特性验证用的是 `--enforce-eager`（因 `FULL_DECODE_ONLY` 在 gemma4 512-dim 全局头 PA 捕获时 ACL 107033 崩溃）。后发现**显式 `cudagraph_mode:PIECEWISE`** 能绕开 `full_graph_pa` 路径、捕获成功（见 dense/MoE W8A8 基线）。本报告补充验证：5 特性能否**叠加 PIECEWISE 图模式**。

## 启动模板（与 eager 版唯一区别：`--enforce-eager` → `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`）
```
vllm serve <model> --tensor-parallel-size 2 \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --limit-mm-per-prompt '{"image":2,"audio":1,"video":1}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --max-model-len 8192 --block-size 128 \
  <feature flags> --additional-config '<json>'
```
每个 case PIECEWISE 捕获 35/35 成功（`Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)`，~17-38s）。

## 结论：5 特性 + chunked+prefix 组合 均可与 PIECEWISE 图模式叠加

| 模型 | baseline | chunked | prefix | weight_nz | async | cpu_binding | chunked+prefix |
|---|---|---|---|---|---|---|---|
| 31B | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 26B MoE | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

14 case 全部：启动成功 + PIECEWISE 捕获成功 + 12/12 功能请求成功 + 配置生效证据齐全。

## 叠加生效证据（每项均确认 PIECEWISE + 特性同时 active）

- **PIECEWISE 模式**：每 case `cudagraph_mode: <CUDAGraphMode.PIECEWISE: 1>` + `Graph capturing finished`。
- **chunked prefill**：`enable_chunked_prefill: True` + `max_num_batched_tokens: 2048` + `Chunked prefill is enabled`。
- **prefix cache**：`enable_prefix_caching: True` + `/metrics` prefix_cache_queries/hits。
- **weight_nz**：`AscendConfig.weight_nz_mode is set ... with value 2`（26B MoE 亦生效）。
- **async scheduling**：`async_scheduling: True` + `Asynchronous scheduling is enabled.`（PIECEWISE+async 兼容）。
- **CPU binding**：`enable_cpu_binding: True` + `[cpu_bind_mode] mode=topo_affinity`（PIEWISE+绑核兼容）。
- **chunked+prefix 组合**：两个 flag 同时 True，PIECEWISE 捕获成功。

## 性能对比（eager vs PIECEWISE，12 请求 mean 延迟）

| 模型 | eager baseline (前次) | PIECEWISE baseline (本次) | 加速 |
|---|---|---|---|
| 31B | 2.737s | 0.711s | ~3.9x |
| 26B | 3.116s | 0.466s | ~6.7x |

图模式显著降延迟；特性叠加不破坏图捕获。

## case 汇总（func 12/12 全 OK）

| case | status | lat_mean_s | case | status | lat_mean_s |
|---|---|---|---|---|---|
| 31B_baseline | OK | 0.711 | 26B_baseline | OK | 0.466 |
| 31B_chunked | OK | 0.725 | 26B_chunked | OK | 0.495 |
| 31B_prefix | OK | 0.722 | 26B_prefix | OK | 0.452 |
| 31B_weightnz | OK | 0.747 | 26B_weightnz | OK | 0.458 |
| 31B_async | OK | 0.702 | 26B_async | OK | 0.442 |
| 31B_cpubind | OK | 0.708 | 26B_cpubind | OK | 0.413 |
| 31B_chunked_prefix | OK | 0.728 | 26B_chunked_prefix | OK | 0.448 |

## 原始日志
- 每个 case：`cases/<case>/`（command.sh, server.log, runner.log, func_verify.json, config_evidence.txt, metrics.prom, status.txt）
- 调度：`logs/run_all.log`；汇总：`reports/case_summary.tsv`

## 结论
gemma4 (31B/26B-MoE) 的 5 大特性 + chunked+prefix 组合，**全部可与 PIECEWISE 图模式叠加**，启动/捕获/请求均成功，未改任何代码。PIECEWISE 是当前 0.23.0 上 gemma4 可用的图模式（FULL/FULL_DECODE_ONLY 因 512-dim PA 捕获 107033 不可用）。
