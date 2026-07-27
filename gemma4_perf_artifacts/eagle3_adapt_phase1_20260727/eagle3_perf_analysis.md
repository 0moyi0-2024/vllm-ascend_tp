# Gemma4 EAGLE3 性能分析（PIECEWISE 图模式）

- 日期：2026-07-27
- 对比：EAGLE3+PIECEWISE（spec-on）vs Target-only+PIECEWISE（spec-off），同图模式公平对比
- Workload：evalscope perf，parallel=1，number=20，input 512 → output 256，greedy（temperature=0, ignore_eos）
- TP4 Target / TP1 Draft，BF16，max_num_seqs=8

## 1. 各 position 接受率（pos0/pos1/pos2，稳态）

num_speculative_tokens=3，故 3 个 draft position。稳态（eval 中后段）：

| 区间 | pos0 | pos1 | pos2 | 平均接受率 | mean acceptance length |
|---|---|---|---|---|---|
| 1 | 0.704 | 0.444 | 0.296 | 48.1% | 2.44 |
| 2 | 0.787 | 0.463 | 0.343 | 53.1% | 2.59 |
| 3 | 0.737 | 0.618 | 0.461 | 60.5% | 2.82 |

**结论**：
- pos0 接受率最高（~0.70-0.79），pos1 中等（~0.44-0.62），pos2 最低（~0.30-0.46）——典型 EAGLE3 衰减 pattern。
- 平均接受率 ~48-60%，mean acceptance length ~2.4-2.8（3 spec token 中平均接受 ~2.5 个）。
- 接受率健康，说明 draft 模型质量正常、spec decoding 功能正确。

## 2. 性能对比（spec-on vs spec-off）

| 指标 | spec-on (EAGLE3) | spec-off (Target-only) | 收益 |
|---|---|---|---|
| TPOT avg | 37.8ms | 40.8ms | **-7.4%**（每 token 快 3ms） |
| TPOT p50 | 35.5ms | 41.0ms | -13.4% |
| TPOT p99 | 56.0ms | 41.4ms | **+35%**（spec 尾延迟变差） |
| Decode tok/s | 26.46 | 24.50 | **+8.0%** |
| Avg Output Rate | 25.76 | 24.10 | +6.9% |
| TTFT | 297.9ms | 214.4ms | **+39%**（spec 首 token 更慢） |

## 3. 分析

**收益**：decode 吞吐 +8%、TPOT -7.4%。mean acceptance length ~2.5 理论上限 ~2.5x，实际仅 ~1.08x。

**为什么低并发下收益小**：
- parallel=1（低并发）下，draft forward 不被批量摊销，每步串行多一次 draft forward。
- spec 的 target forward 是多 token 验证（3+1 token），比 1-token decode forward 更重。
- 反推：benefit = 2.5×T_decode/(T_spec_verify+T_draft) ≈ 1.08 → T_spec_verify+T_draft ≈ 2.3×T_decode。即每步 spec 的 target+draft 开销约为普通 decode 的 2.3 倍，吃掉了大部分接受率红利。
- TTFT 变差 39%：首 token 需先跑 draft propose + target verify，比直接 decode 多一轮。

**p99 变差**：spec decoding 引入尾延迟方差（接受率波动 → 步数波动）。

## 4. 结论与下一步

- **功能正确**：接受率正常（pos0 ~75%，mean length 2.5），greedy token 与 spec-off 完全一致。
- **低并发收益有限**：parallel=1 下 decode 吞吐仅 +8%、且 TTFT/p99 变差。这是 EAGLE3 在低并发下的典型表现，不是适配 bug。
- **预期高并发收益更高**：parallel↑ 时 draft forward 被批量摊销，T_draft/seq 下降，benefit 趋近 mean acceptance length。下一阶段（性能优化）应：
  1. 提高并发（max_num_seqs 64，parallel 8/16）重测吞吐收益。
  2. 评估 draft forward 开销（T_draft 偏高，可能 draft TP1 未与 target 重叠 / draft 模型可优化）。
  3. 调 num_speculative_tokens（acceptance pos2 已 ~0.35，可试 2 或 4）。

## 5. 原始数据
- spec-on summary：`token_compare/perf_specon_summary.txt`；接受率：`logs/eagle3_piecewise_perf.log`（SpecDecoding metrics）
- spec-off summary：`token_compare/perf_specoff_summary.txt`
- eval 日志：`logs/eval_specon.log`、`logs/eval_specoff.log`
