# 根因定位：Gemma4 FULL_DECODE_ONLY cudagraph 捕获失败（ACL 107033）

- 复现：`tests/fdo_rootcause/server.log`（FULL_DECODE_ONLY + block-size 128 + max-model-len 8192 + debug 日志）
- 复现命令见 `tests/fdo_rootcause/boot.sh`

## 1. 失败栈（精确定位到函数）

```
vllm/model_executor/models/gemma4_mm.py:1590  forward
  vllm/model_executor/models/gemma4.py:1278   forward
    vllm/.../attention/attention.py:751       unified_attention_with_output
      vllm_ascend/attention/attention_v1.py:1564  AscendAttentionBackend.forward
        attention_v1.py:1496  forward_impl
          attention_v1.py:1391 forward_paged_attention
            attention_v1.py:1191 full_graph_pa          ← 捕获在此失败
              → torch.npu.graph_task_group_end(stream)
                → ACL rtStreamEndCapture failed, reason=task group status error
                → AclmdlRICaptureEnd error code 107033
                → 后续 NPUCachingAllocator captures_underway.empty() INTERNAL ASSERT（捕获状态被破坏的连带错误）
```

## 2. 根因

**Gemma4 的 512-dim 全局注意力头走 PagedAttention（PA）回退，而该 PA 路径在当前 torch_npu/ACL 构建下无法被捕获进 NPU 图。**

证据链：
1. Gemma4 `text_config`：`head_dim=256`、`global_head_dim=512`、`num_attention_heads=32`、`num_hidden_layers=60` —— **异构头**（部分层 256-dim，全局层 512-dim）。
2. `vllm_ascend/attention/utils.py:146-162 using_paged_attention()`：
   - `FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE = 512`（`device/utils.py:23`）；
   - `if head_size == 512: return True` → 512-dim 全局头强制走 PA（`_npu_paged_attention`），因为 A2/A3 的 FIA-TND 还不支持 512-dim（见 `utils.py:151` TODO："Remove this fallback when A2/A3 FIA TND supports Gemma4's 512-dim global attention heads. Decode can use PA directly."）。
3. FULL_DECODE_ONLY 模式下捕获 decode 图。PA 路径 `full_graph_pa`（`attention_v1.py:1129-1193`）用 `torch.npu.graph_task_group_begin/end` 把 `torch_npu._npu_paged_attention` 包起来再捕获：
   ```
   graph_task_group_begin(stream)
   torch_npu._npu_paged_attention(query, key_cache, value_cache, ..., workspace)
   handle = graph_task_group_end(stream)
   ```
4. 捕获结束 `AclmdlRICaptureEnd` → `rtStreamEndCapture` 失败：`reason=task group status error`，错误码 107033。即 **`_npu_paged_attention` 的 task group 无法被 NPU 图捕获**。连带的 `captures_underway.empty()` 是捕获状态被破坏后的二次报错。

## 3. 结论

- 失败位置：`vllm_ascend/attention/attention_v1.py:full_graph_pa` 中的 `torch_npu._npu_paged_attention` 图捕获。
- 触发条件：Gemma4 异构头中的 **512-dim 全局头**触发 PA 回退（`using_paged_attention` 因 `head_size==512` 返回 True），而该 PA kernel 的 `graph_task_group` 在当前 ACL/torch_npu 下不可捕获。
- 与分支的关系：这正是 `fix-gemma4-mixed-pa-fia-param-utils` 分支的待修点。源码 TODO 明示：等 A2/A3 的 FIA-TND 支持 512-dim 全局头后，移除 PA 回退，decode 直接走 FIA，图捕获即可闭环（FIA 路径可捕获，不走 `graph_task_group`+`_npu_paged_attention`）。
- 因此“Gemma4 不能用图模式”是**当前构建**的状态，不是模型本身限制：原生/上游设计上 FULL_DECODE_ONLY 是支持的，待分支把 512-dim FIA-TND（或 PA 图捕获）修好即可恢复。

## 4. 对 flashcomm1 的影响

flashcomm1 的 SP pass 依赖 cudagraph capture sizes（`platform.py:520-538`）。一旦本根因修好（FULL_DECODE_ONLY 恢复），flashcomm1 拆解中的 **#2（SP 依赖 cudagraph 捕获）自动解除**，剩余主要是 **#3（gemma4 SP 层改写 + matmul_allreduce 形状对齐）**。即 flashcomm1 的估时里有一块是“图模式未修好”的债，图模式本就该修好。

## 5. 修复方向（供参考，未实施）

- 路线 A（推荐，与分支主线一致）：让 A2/A3 FIA-TND 支持 512-dim 全局头，移除 `using_paged_attention` 的 512 回退，decode 走 FIA（可捕获）。
- 路线 B：修复 `_npu_paged_attention` 的 `graph_task_group` 图捕获（torch_npu/ACL 侧，107033 task group status error）。
- 路线 C（临时绕过，已用）：`--enforce-eager` 跳过捕获，代价是性能与 PA 路径（eager 下 `using_paged_attention` 在非 FULL_DECODE_ONLY 也返回 False，实际走 FIA-TND，但无图加速）。

## 6. patch 尝试结果（2026-07-13）：当前构建无法在 vllm-ascend 层 patch 通

尝试“路线 A”：给 `using_paged_attention` 加 env 开关 `VLLM_ASCEND_DISABLE_PA_512_FALLBACK=1`，让 512-dim 全局头改走 FIA-TND、绕开不可捕获的 PA。结果（`tests/fdo_no_pa_fallback/server.log`）：

```
RuntimeError: ... FusedInferAttentionScoreKernelNpuOpApi.cpp:615
NPU function error: call aclnnFusedInferAttentionScoreV3 failed, error code is 561002
```

即 **FIA-TND（aclnnFusedInferAttentionScoreV3）在当前 CANN/torch_npu 构建下同样不支持 512-dim**（561002），与源码 TODO 一致。于是 512-dim 全局头两条路都被卡在 CANN/torch_npu 层：

| 路径 | 512-dim 全局头结果 |
|---|---|
| PA（`_npu_paged_attention` + graph_task_group） | 图捕获失败 ACL 107033（task group status error） |
| FIA-TND（`aclnnFusedInferAttentionScoreV3`） | kernel 拒绝 ACL 561002（不支持 512-dim） |

结论：**当前构建下，没有 vllm-ascend 层的 patch 能让 gemma4 FULL_DECODE_ONLY 图捕获（进而 flashcomm1）生效**。必须等 CANN/torch_npu 升级——要么 FIA-TND 支持 512-dim（移除 PA 回退），要么 PA 的 `graph_task_group` 可被图捕获。实验性 patch（`using_paged_attention` env 开关）已回滚，工作区仅保留 `utils.py` 的 is_moe 误判修复。
