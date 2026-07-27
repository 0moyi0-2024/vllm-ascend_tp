# Gemma4 EAGLE3 极致性能测算报告

- 日期：2026-07-27
- 配置：EAGLE3 + PIECEWISE 图模式（Target TP4 / Draft TP1，BF16），`/home/xty/gemma4/31B` + `/home/xty/eagle3/gemma-4-31b-it-eagle3`
- 对比：spec-on（EAGLE3）vs spec-off（Target-only），同 PIECEWISE 图模式、同 TP4、同 max_num_seqs

## 1. 各 position 接受率（num_speculative_tokens=3）

稳态（自然输出，128 token）：

| 指标 | 值 |
|---|---|
| pos0 | ~0.71 |
| pos1 | ~0.48 |
| pos2 | ~0.30 |
| mean acceptance length | ~2.5 / 3 |
| Avg Draft acceptance rate | ~50% |

接受率健康、典型 EAGLE3 衰减 pattern → draft 质量正常，spec decoding 功能正确。

## 2. 吞吐 speedup 曲线（自然输出 128 tok，n=8-48）

| parallel | spec-on (tok/s) | spec-off (tok/s) | speedup |
|---|---|---|---|
| 1 | 28.1 | 24.6 | **+14.2%** ← 峰值 |
| 2 | 51.0 | 48.3 | +5.6% |
| 4 | 91.6 | 96.0 | -4.6%（噪声内） |
| 8 | 188.1 | 186.0 | +1.1% |
| 16 | 177.6 | 360.5 | **-50.8%** |
| 32 | 264.3 | 519.9 | **-49.2%** |

- 峰值收益 **+14%（parallel 1）**，远低于 EAGLE3 论文 2-3x。
- crossover ≈ p4-p8；p16+ 吞吐回归 ~50%。
- spec-off 随并发线性扩展（24.6→48.3→96→186→360→520），spec-on 不扩展（28→51→92→188→178→264）。

## 3. 已尝试的优化（均未达论文水平）

| 优化 | 结果 |
|---|---|
| `--async-scheduling`（async spec 模式） | 无改善（p8/16/32 与非 async 几乎相同）——draft 仍未与 target 重叠 |
| `num_speculative_tokens=2` | 更差（p1 24.1 vs ns3 28.1；接受率 61% 但每步接受 token 少） |
| 高并发 max_num_seqs 64 | 正确性 OK，但吞吐回归 |
| 低并发 max_num_seqs 8 | 正确性 OK，峰值 +14% |

num_spec=3 是当前最优 spec token 数。

## 4. 根因分析

理论 speedup = mean_acceptance × T_decode / (T_spec_verify + T_draft)。
- acceptance 2.5，理论上限 ~2.5x；实际 p1 仅 1.14x → T_spec_verify + T_draft ≈ 2.2×T_decode。
- **draft forward 未与 target 重叠**：vllm-ascend EAGLE3 的 draft（TP1）forward 串行跑在 target（TP4）之后，async scheduling 未实现真重叠（draft 不在独立 stream/核上）。GPU 上 draft 极轻且与 target 重叠，NPU 上 draft TP1 串行 + 全 LLM forward，开销大。
- **高并发 target 饱和**：spec-verify 是多 token forward（3+1 token，~4x decode 工作）。低并发下 target 未饱和、多 token forward 被摊销（小收益）；高并发下 target 已饱和，4x 工作被 acceptance 2.5 抵消后仍净亏 → -50%。
- **低并发 draft 开销主导**：即使 target 未饱和，串行 draft forward 仍吃掉大部分红利 → 仅 +14%。

## 5. 对齐论文的瓶颈与路径

当前 vllm-ascend EAGLE3 框架达不到论文 2-3x，瓶颈是 **draft forward 不与 target 重叠**。对齐路径（框架级，非本次范围）：
1. **draft-target 真重叠**：draft forward 跑在独立 NPU stream，与 target verify 并发（隐藏 draft 时间）。当前 `update_stream` 仅用于 graph update，未用于 draft 重叠。
2. **draft 模型轻量化/专属核**：降低 T_draft。
3. **sweet-spot 调度**：在 target 未饱和的并发区跑 spec，饱和区退回非 spec（动态切换）。

## 6. 结论

- **功能**：EAGLE3 + PIECEWISE 图模式正确（greedy 一致、接受率 ~50%、稳定）。
- **性能**：当前实现峰值 +14%（parallel 1），高并发 -50%。未达论文水平。
- **瓶颈**：draft forward 未重叠 + 高并发 target 饱和。
- **下一步**：draft-target stream 重叠是达论文收益的关键框架改动。

## 7. 原始数据
- spec-on 曲线：`perf_opt/server_specon_clean.log`、`server_async.log`、`server_ns2.log`
- spec-off 曲线：`perf_opt/server_specoff_clean.log`、`server_specoff_hi.log`
- 测评脚本：`perf_opt/tn2.py`、`tn3.py`、`throughput_natural.py`
- 低并发初步对比：`reports/eagle3_perf_analysis.md`
