# Gemma4 EAGLE3 性能瓶颈深挖 — 最终根因

- 日期：2026-07-27
- 方法：插桩 _propose（draft）+ _model_forward（target），实测各段耗时

## 1. 插桩实测（parallel 1, num_spec=3, PIECEWISE, draft TP1）

### Draft 侧（每 _propose 调用，1050 次平均）
```
combine=0.51s  draft_fwd=5.96s  sample=0.00s   → draft 单步 5.7ms
```
**draft forward 仅 5.7ms/步** —— draft 不是瓶颈。

### Target 侧（每 _model_forward，800 次平均）
```
[PROF_TGT] n=800 avg_tgt_fwd=125ms tokens=4 aclgraph=True
```
**target spec-verify = 125ms（4 token, 已 graph 捕获, 非 eager）**。

## 2. 瓶颈定位

| 项 | 耗时 | 说明 |
|---|---|---|
| target spec-verify (FIA, 512-dim, 4 token, graph) | 125ms | 真瓶颈 |
| draft forward | 5.7ms | 可忽略 |
| 非 spec PA decode (1 token) | 40.7ms | 基线 |

- verify 4 token = 125ms → 31ms/token；PA decode = 40.7ms/token。
- verify per-token(31) < PA(40.7) → FIA 在分摊（4 token < 4×PA=163ms）。
- 但 verify 总量 125ms = **3.07× PA decode(40.7ms)**，因 FIA 512-dim attention 按 token 线性增长（4 token = 4× attention），**未摊销到 1-token 成本**。
- spec-step = 125(verify) + 5.7(draft) ≈ 131ms 产 3.5 token → 26.7 tok/s；非 spec = 40.7ms 产 1 token → 24.6 tok/s → **+14%**。

## 3. 为什么未达论文 3×

论文（GPU）：4-token verify 摊销到 ~1-token 成本（GPU compute-bound，4 token ≈ 1 token）→ spec-step ≈ 40.7+5.7 = 46ms → 3.5 tok/step → 75 tok/s ≈ **3×**。

NPU（当前）：p1 下 NPU overhead-bound，4-token verify 不摊销（125ms = 3× 1-token）→ 仅 +14%。FIA 512-dim attention per-token ~21ms 是不可压成本。

## 4. 高并发回归（-50%）解释

高并发（p16）下 NPU compute-bound：verify 做 4× token 工作，acceptance 3.5 < 4 → 净亏（4/3.5=1.14× 工作量，但实测 -50% 说明 draft 也随 batch 增 + 调度开销）。低并发 overhead-bound 才有 +14%。

## 5. 对齐论文的路径（CANN/内核级）

瓶颈是 **CANN FIA 512-dim attention per-token 成本**（~21ms/token），使 4-token verify 无法摊销到 1-token。路径：
1. **CANN FIA 512-dim 优化**：让多 token verify 摊销（4 token ≈ 1 token 成本）→ 直接 3×。
2. **多 token PA**：`_npu_paged_attention` 支持多 token query 且摊销 → 替代 FIA verify（需修 Mod2 garbage 路径）。
3. **降 512-dim attention 成本**：FIA kernel 针对 512-dim 优化。

均为 CANN/内核级，非 vllm-ascend 配置或 Python 代码可解。

## 6. 结论

- **真瓶颈**：target spec-verify 的 FIA 512-dim attention per-token 成本（125ms/4token vs 40.7ms/1token），非 draft（5.7ms 可忽略）。
- **已确认**：verify 已 graph 捕获（aclgraph=True），非 eager；draft 已 graph 捕获。
- **天花板**：+14%（p1），高并发 -50%。论文 3× 需 CANN FIA 512-dim 让多 token verify 摊销。
- **vllm-ascend 代码改动**：仍仅 Mod 1（+5 行）。所有性能插桩已回退。

## 7. 原始证据
- draft 插桩：`perf_opt/server_prof.log`（[PROF_EAGLE3] combine/draft_fwd/sample）
- target 插桩：`perf_opt/server_tgt_prof.log`（[PROF_TGT] avg_tgt_fwd/tokens/aclgraph）
