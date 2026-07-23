# Gemma4 31B / 26B-A4B MoE — vLLM-Ascend 特性支持验证报告

- 验证时间：2026-07-11 08:44 ~ 10:28
- 报告生成：2026-07-11
- 原始日志根目录：`/home/xty/gemma4_vllm_ascend_feature_verify_20260711_084432`
- 后台任务启动方式：`setsid nohup bash scripts/run_all.sh`（脱离 SSH/Claude 会话，断连继续运行）
- 状态检查命令：`bash /home/xty/gemma4_vllm_ascend_feature_verify_20260711_084432/scripts/check_status.sh`

---

## 1. 环境摘要

| 项 | 值 |
|---|---|
| 机器 | Ascend 910B3 服务器，8 × 910B3（每卡 64 GiB HBM），共 8 卡全部可用 |
| OS / 内核 | Linux 5.15.0-119-generic |
| NPU 驱动 | npu-smi 25.5.1 |
| Python | 3.11.15 (`/usr/local/python3.11.15/bin/python`) |
| vLLM | 0.23.0 (`/usr/local/python3.11.15/lib/python3.11/site-packages/vllm`) |
| vLLM-Ascend | 0.1.dev3825+gf298192fe，**editable 安装自当前路径** `/home/vllm-ascend_tp/vllm-ascend_tp/vllm_ascend` |
| torch / torch-npu | 2.10.0 / 2.10.0 |
| transformers | 5.5.4 |
| vLLM-Ascend 源码路径 | `/home/vllm-ascend_tp/vllm-ascend_tp`（**vllm 启动路径必须为该目录**，editable install） |
| 31B 模型 | `/home/xty/gemma4/31B`，served `gemma-4-31B-it`，dense（非 MoE），bf16，32.7B params / 62.5 GiB 权重 |
| 26B MoE 模型 | `/home/xty/gemma4/26B`，served `gemma-4-26B-A4B-it`，MoE（128 experts，top-k 8），bf16，51.6 GiB 权重 |
| 使用 NPU | 0,1 / 2,3 / 4,5 / 6,7 四组，每组 TP=2 独占两张卡 |
| 并行任务 | 4 个 slot 并行，14 个 case 轮转分配到 4 slot |

环境证据文件：`env/env_full.txt`、`env/source_locations.txt`、`env/source_facts.md`、`env/source_grep_relevant.txt`、`env/boot_findings.md`。

---

## 2. 总结表

支持结论口径：**支持并生效** / **支持但无明显收益** / **支持但性能退化** / **启动支持但运行异常** / **不支持** / **未能验证**。
性能收益非本次重点（按用户要求未使用 `vllm bench`，改用无网络的 12 请求功能验证 + 日志/metrics 证据）；下表“性能收益”列为轻量观测，仅供参考。

| 模型 | chunked prefill | prefix cache | weight_nz | async scheduling | CPU binding | chunked+prefix 组合 |
|---|---|---|---|---|---|---|
| 31B (dense) | 支持并生效 | 支持并生效（命中率 86.5%） | 支持并生效（mode 2） | 支持并生效 | 支持并生效（topo_affinity） | 支持并生效 |
| 26B MoE | 支持并生效 | 支持并生效（命中率 86.4%） | 支持并生效（mode 2，作用于 MoE） | 支持并生效 | 支持并生效（topo_affinity） | 支持并生效 |

> 说明：以上 5 个特性 + chunked+prefix 组合，**在两个模型上均“启动成功 + 配置生效 + 12/12 请求成功 + 日志/metrics 证据充分”**。详见第 4 节。

### 重要前提（gemma4 启动三处坑，已绕过）

参考启动命令（`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`、默认 `max_model_len`、默认 `block_size`）在当前 vllm-ascend 构建上**无法直接启动 gemma4**，需三处调整（详见 `env/boot_findings.md`）：

1. `--max-model-len 8192`：gemma4 `max_position_embeddings=262144`，默认 256k 会在 KV cache profiling 阶段 OOM（单卡 56.63/60.96 GiB）。
2. `--block-size 128`：Ascend 注意力后端仅支持 `[128]`，默认 `block_size=16` → `ValueError: No common block size for 16`。
3. `--enforce-eager`：`FULL_DECODE_ONLY` cudagraph 捕获失败（`AclmdlRICaptureEnd error code 107033`），改用 eager。**代价**：`using_paged_attention()` 仅在 `FULL_DECODE_ONLY` 下返回 True，故 eager 下 gemma4 走 FIA-TND 注意力路径，而非 mixed PA/FIA 路径（该混合路径正是 `fix-gemma4-mixed-pa-fia-param-utils` 分支尚未修好的部分）。

所有 5 特性验证均在 `--enforce-eager` 下进行（特性本身与 cudagraph 模式正交）。

---

## 3. 各特性原理（简述）

1. **chunked prefill**：长 prompt 的 prefill 是 compute-heavy，将其切成多个调度步（受 `max_num_batched_tokens` 限制），让 decode / 短请求可插入，改善混合负载 TTFT/p95/p99 与吞吐稳定性。**注意**：`--max-num-partial-prefills>1`（Concurrent Partial Prefill）在 Ascend **不支持**（`NotImplementedError`），故仅验证 `--enable-chunked-prefill --max-num-batched-tokens 2048`。
2. **prefix cache / APC**：缓存已处理请求的 KV cache block，共享 token prefix 的新请求可复用 KV、跳过共享部分 prefill，主要降低 TTFT、提升共享前缀负载吞吐。
3. **weight_nz**：Ascend NZ 是面向 NPU 矩阵计算友好的权重布局。`weight_nz_mode`：`0`=关闭、`1`=仅 quant 场景启用（默认）、`2`=BF16/FP16 也启用。两模型均为 bf16 非量化，故默认(mode 1)==关闭；强制开启用 mode 2。
4. **async scheduling**：CPU 侧调度与设备执行重叠，减少 NPU 利用率空隙，改善吞吐/延迟。
5. **CPU binding / 绑核**：控制 worker 进程、关键 runtime 线程、内存页及部分 NPU IRQ 的 CPU/NUMA 亲和性，降低跨 NUMA 访存与线程抢占，主要改善尾延迟与稳定性，不改数值。

---

## 4. 每个模型逐项验证详情

每个 case 目录：`cases/<case>/`，含 `command.sh`、`server.log`、`runner.log`、`smoke_response.json`、`func_verify.json`、`config_evidence.txt`、`metrics.prom`、`evidence_grep.txt`，prefix case 另含 `bench_shared_prefix.log`、`prefix_cache_metrics.txt`，cpubind case 另含 `taskset_main.txt`、`thread_affinity.txt` 等。

通用启动命令模板（31B；26B 增加 `--enable-expert-parallel`）：
```
cd /home/vllm-ascend_tp/vllm-ascend_tp && \
ASCEND_RT_VISIBLE_DEVICES=<devs> HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=256 \
vllm serve <model_path> --served-model-name <name> --tensor-parallel-size 2 \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --limit-mm-per-prompt '{"image":2,"audio":1,"video":1}' \
  --enforce-eager --max-model-len 8192 --block-size 128 \
  <feature flags> --additional-config '<json>'
```
功能验证：12 条不同短问题，`max_tokens=32, temperature=0`，无网络（urllib→localhost）。所有 case 均 **12/12 成功**，输出正常。

### 4.1 Baseline（两模型）

- 参数差异：`--no-enable-prefix-caching --no-enable-chunked-prefill --no-async-scheduling --additional-config '{"enable_cpu_binding": false, "weight_nz_mode": 0}'`
- 证据：`config_evidence.txt` 显示 `enable_prefix_caching: False, enable_chunked_prefill: False, async_scheduling: False, enable_cpu_binding: False, weight_nz_mode: 0`。
- func：31B 12/12（mean 2.74s）、26B 12/12（mean 3.12s）。smoke 回答正确（“vLLM 是一个高性能的开源 LLM 推理和部署库……”）。
- 日志：`cases/31B_baseline/`、`cases/26B_baseline/`。

### 4.2 Chunked Prefill（两模型）

- 参数差异：`--enable-chunked-prefill --max-num-batched-tokens 2048`（关闭 prefix cache 隔离变量）。
- 生效证据：日志 `Chunked prefill is enabled with max_num_batched_tokens=2048.`；config `enable_chunked_prefill: True, max_num_batched_tokens: 2048`。
- func：31B 12/12（2.95s）、26B 12/12（3.16s）。
- 日志：`cases/31B_chunked/server.log`、`cases/26B_chunked/server.log`。
- 结论：**支持并生效**。

### 4.3 Prefix Cache / APC（两模型）

- 参数差异：`--enable-prefix-caching`（关闭 chunked prefill 隔离变量）。
- 生效证据：config `enable_prefix_caching: True`；`/metrics` 出现 `vllm:prefix_cache_queries_total` 与 `vllm:prefix_cache_hits_total`。
- 命中率（shared-prefix 40 请求 workload 后）：
  - 31B：queries 140270 / hits 121344 → **86.5% 命中**；共享前缀 TTFT 1.96s vs 随机 2.26s（**-13.2%**）。
  - 26B：queries 140092 / hits 121088 → **86.4% 命中**；共享前缀 TTFT 2.13s vs 随机 2.38s（**-10.4%**）。
- func：31B 12/12、26B 12/12。
- 日志：`cases/31B_prefix/{bench_shared_prefix.log,prefix_cache_metrics.txt}`、`cases/26B_prefix/...`。
- 结论：**支持并生效**（命中率与 TTFT 收益均被观测到）。

### 4.4 Weight NZ（两模型）

- 参数差异：`--additional-config '{"weight_nz_mode": 2}'`（其余关闭）。
- 生效证据：日志 `AscendConfig.weight_nz_mode is set from additional_config with value 2.`（31B 与 26B 均出现）；config `weight_nz_mode: 2`。
- 26B（MoE）：mode 2 作用于 MoE 权重，服务正常启动并 12/12 请求成功（即 weight_nz + MoE 组合可用）。
- func：31B 12/12（2.64s）、26B 12/12（3.03s）。
- 日志：`cases/31B_weightnz/`、`cases/26B_weightnz/`。
- 结论：**支持并生效**（含 MoE 场景）。性能增益非本次重点，未做 A/B 性能对比。

### 4.5 Async Scheduling（两模型）

- 参数差异：`--async-scheduling`（其余关闭）。
- 生效证据：日志 `Asynchronous scheduling is enabled.`（vllm.py:999）；config `async_scheduling: True`。
- func：31B 12/12（2.97s）、26B 12/12（3.21s），无 hang。
- 日志：`cases/31B_async/`、`cases/26B_async/`。
- 结论：**支持并生效**。

### 4.6 CPU Binding / 绑核（两模型）

- 参数差异：`--additional-config '{"enable_cpu_binding": true}'`（其余关闭；baseline 为 false 作为对照）。
- 生效证据：日志 `[cpu_bind_mode] mode=topo_affinity rank=0 visible_npus=[...]` 与 `rank=1`（worker 实际执行绑核）；config `enable_cpu_binding: True`。
- 亲和性采集：`taskset_main.txt`、`cpus_allowed_main.txt`、`thread_affinity.txt`、`threads_psr.txt` 已保存。
- func：31B 12/12（2.70s）、26B 12/12（2.88s）。
- 日志：`cases/31B_cpubind/`、`cases/26B_cpubind/`。
- 结论：**支持并生效**（绑核动作确实执行，topo_affinity 模式）。

### 4.7 组合：chunked prefill + prefix cache（两模型）

- 参数：`--enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 2048`。
- 生效证据：config 同时 `enable_prefix_caching: True` 且 `enable_chunked_prefill: True`；日志同时出现 chunked prefill 启用与 APC metrics。
- func：31B 12/12（3.16s）、26B 12/12（3.28s）。
- 日志：`cases/31B_chunked_prefix/`、`cases/26B_chunked_prefix/`。
- 结论：**支持并生效**，无冲突报错。

---

## 5. 组合特性结论

| 组合 | 31B | 26B MoE | 说明 |
|---|---|---|---|
| chunked prefill + prefix cache | 支持 | 支持 | 同时启用，无报错，请求成功 |
| async scheduling + chunked prefill | 未单独跑 | 未单独跑 | 两者均单独验证可用；未发现互斥报错（async case 已关闭 chunked 以隔离变量） |
| weight_nz + MoE | — | 支持 | 26B weight_nz_mode=2 + EP 128 experts 正常启动并 12/12 成功 |
| CPU binding + TP=2/EP | 支持 | 支持 | TP=2（31B）/ TP=2+EP（26B）下 topo_affinity 绑核生效 |
| **Concurrent Partial Prefill**（`--max-num-partial-prefills>1`） | **不支持** | **不支持** | `NotImplementedError: Concurrent Partial Prefill is not supported`（见 `cases/31B_chunked/server.log` 首次失败记录），已改用 `--enable-chunked-prefill` 验证 |

---

## 6. 并行与后台运行记录

- 启动：`setsid nohup bash scripts/run_all.sh > logs/run_all.nohup.log 2>&1 &`，`run_all.pid` 记录 PID。
- 4 slot × 2 卡 + 端口：slot0=0,1/9000；slot1=2,3/9002；slot2=4,5/9004；slot3=6,7/9006。14 case 轮转分配。
- 完整调度日志见 `logs/run_all.log`。关键时间线：
  - 10:05:47 START（14 case）
  - 10:10:47 ~ 10:27:40 各 case 陆续 END，全部 rc=0
  - 10:28:02 run_all COMPLETE
- 每个 case：`cases/<case>/runner.log` 记录 START/ready/smoke/func/END 时间与 duration；`status.txt`=OK，`exitcode`=0。
- 每个 case 启动命令见 `cases/<case>/command.sh`。
- 进程清理：因 vllm EngineCore/Worker_TP 会在父进程退出后逃逸进程组，`run_one_case.sh` 采用“优雅 TERM → 捕获后代 PID 强杀 → 按 NPU 卡清理 orphan worker”三段式 shutdown，并在每个 case 启动前 pre-flight 清卡。运行结束后 8 卡全部空闲、无残留 vllm 进程。

---

## 7. 附录

### 7.1 原始日志目录树
```
/home/xty/gemma4_vllm_ascend_feature_verify_20260711_084432/
├── README.txt
├── env/            env_full.txt, source_locations.txt, source_facts.md,
│                   source_grep_relevant.txt, boot_findings.md, vllm_serve_help.txt, npu_smi_info.txt ...
├── scripts/        run_all.sh, run_one_case.sh, check_status.sh, collect_results.sh, shared_prefix_bench.py
├── logs/           run_all.log, run_all.nohup.log, slotN.log, slotN_<case>.log
├── pids/           run_all.pid, slotN.pid
├── cases/          14 个 case 目录（每个含 command.sh, server.log, runner.log,
│                   smoke_response.json, func_verify.json, config_evidence.txt,
│                   metrics.prom, evidence_grep.txt, status.txt, exitcode.txt; prefix/cpubind 额外文件）
├── bench/          各 case bench 日志副本（功能性 func_verify 为主）
├── metrics/        各 case metrics.prom 副本
├── status/         各 case exitcode
└── reports/        case_summary.tsv, gemma4_vllm_ascend_feature_report.md（本文件）
```

### 7.2 case 汇总（case_summary.tsv）

| case | status | func_ok/total | lat_mean_s | lat_max_s |
|---|---|---|---|---|
| 31B_baseline | OK | 12/12 | 2.74 | 5.63 |
| 31B_chunked | OK | 12/12 | 2.95 | 6.46 |
| 31B_prefix | OK | 12/12 | 2.96 | 6.50 |
| 31B_weightnz | OK | 12/12 | 2.64 | 5.70 |
| 31B_async | OK | 12/12 | 2.97 | 6.21 |
| 31B_cpubind | OK | 12/12 | 2.70 | 5.58 |
| 31B_chunked_prefix | OK | 12/12 | 3.16 | 6.68 |
| 26B_baseline | OK | 12/12 | 3.12 | 5.94 |
| 26B_chunked | OK | 12/12 | 3.16 | 5.99 |
| 26B_prefix | OK | 12/12 | 3.11 | 5.84 |
| 26B_weightnz | OK | 12/12 | 3.03 | 5.99 |
| 26B_async | OK | 12/12 | 3.21 | 6.08 |
| 26B_cpubind | OK | 12/12 | 2.88 | 5.45 |
| 26B_chunked_prefix | OK | 12/12 | 3.28 | 6.28 |

### 7.3 启动命令修改说明（相对参考命令）

| 修改 | 原始 | 修改后 | 原因 |
|---|---|---|---|
| max_model_len | 默认 256k | `--max-model-len 8192` | 默认 256k 在 profiling 阶段 OOM |
| block_size | 默认 16 | `--block-size 128` | Ascend 后端仅支持 [128]，否则 "No common block size for 16" |
| cudagraph | `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` | `--enforce-eager` | FULL_DECODE_ONLY 捕获失败（ACL 107033） |
| 31B expert-parallel | 参考命令未含（用户确认 31B 非 MoE） | 不加 `--enable-expert-parallel` | 31B dense；26B 保留 |
| chunked prefill | 参考 `--max-num-partial-prefills 2` 等 | 仅 `--enable-chunked-prefill --max-num-batched-tokens 2048` | Concurrent Partial Prefill 不支持 |

### 7.4 失败 case traceback
- 早期 `31B_chunked`（首次，含 `--max-num-partial-prefills 2`）：`NotImplementedError: Concurrent Partial Prefill is not supported`，已记录于 `cases/31B_chunked/server.log`（首次失败快照后被正确重跑覆盖为 OK）。
- 早期 OOM / block-size / cudagraph 三类失败证据见 `env/boot_findings.md` 与 `tests/boot_test_31b.log`。

---

## 8. 最终结论

| 模型 | chunked prefill | prefix cache | weight_nz | async scheduling | CPU binding |
|---|---|---|---|---|---|
| 31B | 支持并生效 | 支持并生效（86.5% 命中） | 支持并生效（mode 2） | 支持并生效 | 支持并生效（topo_affinity） |
| 26B MoE | 支持并生效 | 支持并生效（86.4% 命中） | 支持并生效（mode 2，含 MoE） | 支持并生效 | 支持并生效（topo_affinity） |

- **未验证成功项**：无。14 case 全部 OK。
- **已知限制**：(1) Concurrent Partial Prefill 不支持；(2) FULL_DECODE_ONLY cudagraph 在当前构建对 gemma4 捕获失败（故用 eager），mixed PA/FIA 路径暂不可用——这正是 `fix-gemma4-mixed-pa-fia-param-utils` 分支待修内容。
- **原始日志完整性**：全部保存于 `/home/xty/gemma4_vllm_ascend_feature_verify_20260711_084432/`，含每个 case 的 server.log / metrics / 证据 grep / 功能验证结果 / 启动命令。

---

## 9. 附加：flashcomm1（flash_comm_v1 / SP 通信融合）验证

按用户要求追加验证 flashcomm1。详见独立报告 `reports/flashcomm1_verification_report.md`。

- **配置门**：原被 `is_moe_model` 误判阻断（gemma4 dense 的 `text_config` 含 `num_experts=None` 等占位 key，`_is_contain_expert` 仅按 key 名判定 → 误判 dense 为 MoE → 触发“MoE 需 EP”断言）。**已修复** `vllm_ascend/utils.py:_is_contain_expert`（expert key 仅当值非 None 才计为 MoE 信号）；验证 31B dense→False、26B MoE→True，无回归，31B baseline 启动+服务正常。
- **运行时**：配置门通过后，eager / 编译+SP-pass / +EP 三种路径均在 profiling 阶段报 `tensor a (1024) must match tensor b (2048)` 形状不匹配 —— gemma4 的 SP 层改写 / `matmul_allreduce` 融合与 gemma4 层结构不匹配，属分支待完成的 gemma4 SP 适配，且与 cudagraph 捕获失败（ACL 107033）耦合，非局部补丁可解。
- **结论**：flashcomm1 在两个 gemma4 模型上**配置可开、运行时暂不支持**；已交付 is_moe 误判修复（保留于工作区 `vllm_ascend/utils.py`），`platform.py` 实验性改动已回滚。
- **保留的代码改动**：`git diff --stat` → `vllm_ascend/utils.py | 7 ++++++-`（仅此一处）。

