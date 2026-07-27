# Gemma4 EAGLE3 极致性能优化 — 最终报告

- 日期：2026-07-27
- 配置：EAGLE3 + PIECEWISE 图模式，Target TP4 / Draft TP1（默认），BF16
- 目标：对齐 EAGLE3 论文 2-3x

## 1. 结论：当前 NPU + vllm-ascend 下 EAGLE3 收益天花板 ~1.14x，未达论文水平

| parallel | spec-on | spec-off | speedup |
|---|---|---|---|
| 1 | 28.1 tok/s | 24.6 | **+14.2%（峰值）** |
| 2 | 51.0 | 48.3 | +5.6% |
| 4 | 91.6 | 96.0 | ~flat |
| 8 | 188.1 | 186.0 | ~flat |
| 16 | 177.6 | 360.5 | **-50.8%** |
| 32 | 264.3 | 519.9 | **-49.2%** |

接受率健康（pos0 0.71 / pos1 0.48 / pos2 0.30，mean 2.5），功能正确。

## 2. 每步成本拆解（parallel 1，反推）

- spec-off：40.7 ms / target-1token-step。
- spec-on：124.5 ms / spec-step（3.5 token/step）= target-verify(4 token, ~40.7ms，p1 未饱和摊销) + 3×draft-step。
- → **draft-step ≈ 27.9 ms**，即 draft 单步 ≈ 0.69× target-1token-step。
- 论文假设 draft-step << target-step（GPU 上 ~0.1×，且重叠）。NPU 上 draft 单步达 0.69×，且 EAGLE3 的 K 步串行不可重叠 → 收益被吃掉。

## 3. 已尝试的优化（均未达论文水平）

| 优化 | 结果 |
|---|---|
| `--async-scheduling`（async spec） | 无改善（draft 仍串行，未真重叠） |
| draft `draft_tensor_parallel_size=4` | 无改善（p1 27.2 vs TP1 28.1；小 draft，TP4 通信开销抵消） |
| `num_speculative_tokens=2` | 更差（p1 24.1；每步接受 token 少） |
| `parallel_drafting` | 仅 dflash/dspark 可用，eagle3 不支持 |
| 高/低并发 | 低并发峰值 +14%，高并发回归 -50% |

draft 已是 graph-captured（`use_cuda_graph=True` under PIECEWISE），非 eager。

## 4. 根因

EAGLE3 的 K=3 draft 步串行（每步需上一步输出），无法与 target 重叠。NPU 上每个 draft 步的固定开销（kernel launch / graph replay / hidden-state combine[layers 2,30,57] / sampling / sync）≈ 28ms，远超小 draft 模型本身的算力成本，导致 draft-step ≈ 0.69× target-step。GPU 上 draft 极轻 + 重叠，NPU 上 draft 重 + 串行。

高并发（p16+）额外问题：target 已饱和，spec-verify 多 token forward（~4x 工作）被 acceptance 2.5 抵消后净亏 → -50%。

## 5. 对齐论文的路径（框架/内核级，非本次可完成）

1. **降低 draft 单步固定开销**：把 K 步 draft 的 launch/sync/combine 开销降到接近 0（如 draft 全图单次 replay 产出 K token、hidden-state combine 融合、减少 host-sync）。当前 28ms/步 → 目标 < 5ms/步。
2. **draft 与 target 真重叠**：EAGLE3 的 K 步理论上可与下一轮 target verify 部分流水化（需改造 proposer 调度，eagle3 当前不支持 parallel_drafting）。
3. **sweet-spot 调度**：target 未饱和区开 spec、饱和区退回非 spec（动态切换，避免高并发回归）。

## 6. 当前可交付的最佳配置

```bash
vllm serve /home/xty/gemma4/31B --tensor-parallel-size 4 --max-num-seqs 8 \
  --max-model-len 8192 --block-size 128 \
  --speculative-config '{"model":"/home/xty/eagle3/gemma-4-31b-it-eagle3","method":"eagle3","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
```
- 低并发（parallel ≤ 4）：+6~14% decode 收益。
- 高并发：建议关 spec（回归）。
- 唯一代码改动：Mod 1（+5 行，对齐上游 image_token_index）。

## 7. 原始数据
- 见 `reports/eagle3_perf_extreme_analysis.md` 及 `perf_opt/` 下各 server_*.log / tn*.py。
