# Gemma4 特性 × 图模式 叠加验证 — 汇总报告

- 日期：2026-07-23
- 模型：Gemma4 31B (dense, bf16) / 26B-A4B (MoE, bf16)，910B3
- 代码：`/home/vllm-ascend_tp/vllm-ascend_tp`（vllm 0.23.0 + vllm-ascend @ a83b055e），全程未改代码

## 目标

之前的 5 特性验证用 `--enforce-eager`（因 FULL_DECODE_ONLY 在 TP2 崩）。本汇总补充验证 5 特性能否叠加**图模式**，覆盖 eager / PIECEWISE / FULL_DECODE_ONLY 三种。

## 5 特性 × 3 图模式 × 2 模型 — 叠加结果

特性：chunked prefill、prefix cache、weight_nz(mode 2)、async scheduling、CPU binding，外加 chunked+prefix 组合。

| 图模式 | TP | 31B | 26B MoE | 备注 |
|---|---|---|---|---|
| eager (`--enforce-eager`) | 2 | 7/7 ✅ | 7/7 ✅ | 无图；前次验证（feature_verify_20260711） |
| PIECEWISE（显式） | 2 | 7/7 ✅ | 7/7 ✅ | 按 attention 边界切分，避开 full_graph_pa |
| FULL_DECODE_ONLY | 4 | 7/7 ✅ | 7/7 ✅ | 需 TP4 拆 head_dim 到 128；TP1/TP2 不可用 |

**共 42 case，全部启动+捕获+12/12 请求成功，特性 flag 全部生效。** 即 5 特性 + chunked+prefix 组合在三种图模式下（FULL_DECODE_ONLY 限 TP4）均可叠加。

## 关键技术点

1. **gemma4 512-dim 全局头**：`using_paged_attention()` 在 `head_size==512` 强制 PA（`full_graph_pa` → `graph_task_group`），当前 ACL 图捕获崩（107033）。FIA-TND 直接处理 512-dim 又报 561002（不支持）。
2. **eager**：跳过捕获，always 可用，最慢。
3. **PIECEWISE（显式 `cudagraph_mode:PIECEWISE`）**：按 attention 边界切分捕获，不触发 `full_graph_pa`，**TP2 即可用**。注意 `platform.py:61` 的 `VLLM_USE_BREAKABLE_CUDAGRAPH=False` 只禁用 breakable 自动启用，不阻止显式 PIECEWISE。
4. **FULL_DECODE_ONLY**：捕获 decode 全图，命中 `full_graph_pa` → TP1/TP2 崩；**TP4 把 head_dim 拆到 128**（512/4），不触发 PA 回退，改走 FIA-TND，捕获成功。

## 性能（12 请求 mean 延迟，对比图模式收益）

| 模型 | eager | PIECEWISE(TP2) | FULL_DECODE_ONLY(TP4) |
|---|---|---|---|
| 31B baseline | 2.737s | 0.711s | 0.630s |
| 26B baseline | 3.116s | 0.466s | 0.486s |

图模式比 eager 快 ~4-7x；PIECEWISE(TP2) 与 FULL_DECODE_ONLY(TP4) 量级相当（TP4 单卡负载更低但通信开销略增）。

## 复现（每种模式一个启动模板，特性 flag 叠加在 `<feature flags>`）

```bash
# eager (TP2)
vllm serve <model> --tensor-parallel-size 2 ... --enforce-eager --max-model-len 8192 --block-size 128 <feature flags>

# PIECEWISE (TP2)
vllm serve <model> --tensor-parallel-size 2 ... \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' --max-model-len 8192 --block-size 128 <feature flags>

# FULL_DECODE_ONLY (TP4)
vllm serve <model> --tensor-parallel-size 4 ... \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --max-model-len 8192 --block-size 128 <feature flags>
```

## 子目录与报告
- `feature_verify_20260711/` — eager 版（前次），报告 `reports/gemma4_vllm_ascend_feature_report.md`
- `feature_graphmode_20260723_062949/` — PIECEWISE 版，报告 `reports/feature_graphmode_piecewise_report.md`
- `feature_fdo_20260723_071510/` — FULL_DECODE_ONLY(TP4) 版，报告 `reports/feature_graphmode_fdo_report.md` + 本汇总

## 结论
gemma4 (31B/26B-MoE) 的 5 大特性 + chunked+prefix 组合，在 **eager / PIECEWISE / FULL_DECODE_ONLY** 三种图模式下均可叠加（FULL_DECODE_ONLY 需 TP4）。推荐生产用 **PIECEWISE(TP2)**（单卡/双卡可用，无需 TP4）或 **FULL_DECODE_ONLY(TP4)**（4 卡，decode 全图）。未改任何 vllm-ascend 代码。
