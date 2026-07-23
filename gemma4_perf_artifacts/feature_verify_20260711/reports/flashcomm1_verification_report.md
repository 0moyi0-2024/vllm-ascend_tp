# flashcomm1 (flash_comm_v1 / Sequence Parallelism 通信融合) 验证报告

- 验证时间：2026-07-11 10:35 ~ 10:55
- 模型：Gemma4 31B (dense) / 26B-A4B (MoE)，TP=2，910B3
- 原始日志：`tests/flashcomm1_*/server.log`

## 1. flashcomm1 是什么

`flash_comm_v1`（flashcomm1）是 vllm-ascend 的**序列并行（Sequence Parallelism, SP）通信融合**特性：把 `RowParallelLinear`（down_proj）的 matmul 与 all-reduce（以及 rmsnorm）融合成 `matmul_allreduce` 类算子，减少通信开销，主要在长序列/高并发 prefill 时收益。MoE 场景还融合 dispatch/combine 通信。

- 开关：`--additional-config '{"enable_flashcomm1": true}'` 或环境变量 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`（默认 0=关）。
- 运行时激活（`vllm_ascend/ascend_forward_context.py:117-137`）：
  - dense 非drafter模型：`enable_sp() and num_tokens > 1000`（1000 为经验并发/token 阈值）；
  - MoE context 模型：`enable_sp() and num_tokens`（无 1000 阈值）。
- 编译期：需 `enable_sp_by_pass=True`（`ascend_config.py:284`），条件为 `not enforce_eager and pass_config.enable_sp`，由 SP 层重写 pass 把线性层改写为 SP 形态。

## 2. 验证过程与发现

### 2.1 发现并修复 is_moe_model 误判（配置门阻断）—— 已修复

首次在 31B(dense) 启用 flashcomm1 立即报：
```
AssertionError: Flash Comm v1 requires enable_expert_parallel=True for MoE models.
```
（`vllm_ascend/platform.py:762`）

根因：`vllm_ascend/utils.py:_is_contain_expert` 仅按 **key 名是否含 "expert"** 判定 MoE，而 Gemma4 **dense** 的 `text_config` 带有占位字段 `expert_intermediate_size=None / num_experts=None / top_k_experts=None` → 误判为 MoE → 触发“MoE 必须 EP”断言。31B dense 又无法开启 `--enable-expert-parallel`（报 `Number of experts in the model must be greater than 0`），死锁。

**修复**（`vllm_ascend/utils.py` `_is_contain_expert`）：expert key 仅当值非 None 时才作为 MoE 信号。验证：
- 31B(dense)：修复前 True → 修复后 **False**（正确）
- 26B(MoE)：修复前/后均为 **True**（num_experts=128，正确，无回归）

修复后 31B flashcomm1 配置门通过（`AscendConfig.enable_flashcomm1 is set from additional_config with value True`），26B 同样通过。

### 2.2 运行时阻断：SP/matmul_allreduce 形状不匹配 —— 未支持（gemma4 SP 支持缺口）

配置门通过后，无论 eager 还是编译模式，均在 EngineCore 初始化（profiling dummy run）失败：
```
RuntimeError: Worker failed with error 'The size of tensor a (1024) must match the size of tensor b (2048) at non-singleton dimension 0'
```
（1024 = 2048/tp_size）

复现的三种配置均失败：
| 配置 | 结果 |
|---|---|
| `--enforce-eager` + flashcomm1（无 SP pass） | 配置通过 → 运行时 1024/2048 形状不匹配 |
| `--enforce-eager` + flashcomm1 + `--enable-expert-parallel`（dense） | 配置阶段即拒：`Number of experts must be > 0` |
| `cudagraph_mode=NONE` + `pass_config.enable_sp=true` + flashcomm1（编译+SP pass，不捕获图） | 配置通过 → 仍 1024/2048 形状不匹配 |

根因分析：flashcomm1 运行时融合要求线性层已被 SP pass 正确改写（输入按 tp 切分）。当前构建中 gemma4 的 SP 层改写/`matmul_allreduce` 路径与 gemma4 层结构不匹配，导致 dummy profiling（num_tokens=2048>1000 触发 `flash_comm_v1_enabled=True`）即形状错位。这正是 `fix-gemma4-mixed-pa-fia-param-utils` 分支尚未完成的 gemma4 注意力/线性 SP 支持范畴。

附带发现：gemma4 的 `FULL_DECODE_ONLY` cudagraph 捕获失败（ACL 107033），而 SP pass 又依赖 cudagraph capture sizes（`platform.py:520-538`），二者进一步耦合，使 flashcomm1 在 gemma4 上短期内难以闭环。

## 3. 尝试过的“想办法支持”

1. **is_moe 误判修复**（已合入工作区，`utils.py`）：解除 31B dense 的配置门，是正确且无回归的修复（26B MoE 检测不变）。
2. **`--enable-expert-parallel` 绕过**（dense）：不可行，dense 0 experts 被拒。
3. **`cudagraph_mode=NONE` + `pass_config.enable_sp=true`**（编译但不捕获图，规避 ACL 107033）：尝试对 `platform.py` 的 SP-aclgraph sizes 断言做条件放宽实验，配置门通过，但运行时仍 1024/2048 形状不匹配 —— 说明问题不在断言，而在 gemma4 SP 层改写本身。**该 platform.py 实验性改动已回滚**，未保留。

## 4. 结论

| 模型 | flashcomm1 配置门 | flashcomm1 运行时 | 总结 |
|---|---|---|---|
| 31B (dense) | 通过（已修 is_moe 误判） | **不支持**（SP/matmul_allreduce 1024/2048 形状不匹配） | 配置可开，运行时需 gemma4 SP 支持（分支 WIP） |
| 26B (MoE) | 通过 | **不支持**（同样 1024/2048 形状不匹配） | 配置可开，运行时需 gemma4 SP 支持（分支 WIP） |

- **已交付的改进**：`vllm_ascend/utils.py:_is_contain_expert` 修复 is_moe_model 对 gemma4 dense 的误判（key-presence → 值非 None）。该修复正确、无回归，并顺带修正了 `ascend_forward_context` 对 31B 的 `is_context_moe_model` 误判。
- **未支持根因**：gemma4 的 SP 层改写 / `matmul_allreduce` 融合与 gemma4 层结构不匹配（1024 vs 2048），属于 `fix-gemma4-mixed-pa-fia-param-utils` 分支待完成的 gemma4 注意力/线性 SP 适配；且与 gemma4 cudagraph 捕获失败（ACL 107033）耦合。彻底支持需在该分支上完成 gemma4 SP 适配，非局部补丁可解。

## 5. 复现命令

```bash
cd /home/vllm-ascend_tp/vllm-ascend_tp
ASCEND_RT_VISIBLE_DEVICES=0,1 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=256 \
vllm serve /home/xty/gemma4/31B --served-model-name gemma-4-31B-it --tensor-parallel-size 2 \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --limit-mm-per-prompt '{"image":2,"audio":1,"video":1}' \
  --enforce-eager --max-model-len 8192 --block-size 128 \
  --no-enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 2048 \
  --additional-config '{"enable_cpu_binding": false, "weight_nz_mode": 0, "enable_flashcomm1": true}' \
  --host 0.0.0.0 --port 9000
# → AscendConfig.enable_flashcomm1=True，但 EngineCore init 报 1024 vs 2048 形状不匹配
```
