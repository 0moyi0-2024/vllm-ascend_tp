# Gemma4 on A2 错误分析：rope UB 溢出 + cudagraph attention 参数 unpack 不匹配

本文记录在 vLLM-Ascend 上跑通 Gemma4（含 W8A8 量化版）时遇到的两个**非量化、非环境**的代码问题：rope triton kernel UB 溢出（编译期）、cudagraph 全解码图 attention 参数 unpack 不匹配（推理期）。给出完整原始报错、根因分析、修改方案与验证结论。

> 硬件：4 × Ascend 910B3（A2），TP=4
> 模型：Gemma4 31B（浮点 `/home/xty/gemma4/31B`；W8A8 量化 `/home/wangminghua/gemma4_w8a8_rollback`）
> 启动配置：`--tensor-parallel-size 4 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`

## 背景：Gemma4 异构 head_dim

Gemma4 文本部分是异构 head_dim 的 GQA：
- `sliding_attention`：`head_dim=256`，`num_key_value_heads=16`，rope `partial_rotary_factor=1.0`
- `full_attention`：`global_head_dim=512`，`num_global_key_value_heads=4`，`attention_k_eq_v=True`，rope `partial_rotary_factor=0.25`

vLLM 的 `Gemma4ForConditionalGeneration` 对所有层统一建 `QKVParallelLinear`，full_attention（k_eq_v）层在加载时把 `k_proj` 复制到 V 槽位。A2 上 FIA TND 暂不支持 512-dim global attention head，故 512-dim full-attention 层在 decode 时回退到 PA（paged attention），sliding 层用 FIA——这导致一个 decode graph 里**混合**了 PA 与 FIA 两种 attention 参数格式。

---

## 问题一：rope triton kernel UB 溢出（编译期）

### 原始报错

triton 环境修好后（见附录），`profile run` 编译阶段 `vllm_ascend/ops/triton/rope.py` 的 `_triton_rope` kernel 在 BiShengIR 编译失败：

```
loc("/home/vllm-ascend_tp/vllm-ascend_tp/vllm_ascend/ops/triton/rope.py":24:0): error: ub overflow,
requires 1581056 bits while 1572864 bits available!
(possible reason: tiling basic block is too large or block number is more
than what user expect due to multi-buffer feature is enabled and some ops
need extra local buffer.)
loc("/home/vllm-ascend_tp/vllm-ascend_tp/vllm_ascend/ops/triton/rope.py":24:0): error: Failed to run BiShengHIR pipeline
[ERROR] Failed to run BiShengIR pipeline
[INFO]: The compiled kernel cache is in /root/.triton/cache/...

subprocess.CalledProcessError: Command
'[/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/bishengir/bin/bishengir-compile,
 /tmp/tmpeemnbish/kernel.ttadapter.mlir,
 --target=Ascend910B3,
 --enable-auto-multi-buffer=True,
 --enable-auto-bind-sub-block=True,
 --enable-hfusion-compile=true,
 --enable-hivm-compile=true,
 --enable-triton-kernel-compile=true,
 -o, /tmp/tmpeemnbish/kernel]'
returned non-zero exit status 1.
triton.compiler.errors.MLIRCompilationError
```

调用链（profile run → model forward → rope）：
```
vllm_ascend/ops/rotary_embedding.py:168  rope_forward_oot
  → vllm_ascend/ops/triton/rope.py:285  rope_forward_triton
    → _triton_rope[(n_row,)](... BLOCK_SIZE_HEAD=BLOCK_SIZE_HEAD ...)
      → triton JIT compile → bishengir-compile → UB overflow
```

### 根因分析

`vllm_ascend/ops/triton/rope.py` 的 `_triton_rope` kernel 按 `BLOCK_SIZE_HEAD` 分块 q/k head（代码注释自称 *"q/k head dimensions are tiled with BLOCK_SIZE_HEAD to avoid UB overflow"*）。当前取值：

```python
# rope_forward_triton
if is_neox_style:
    BLOCK_SIZE_HEAD = 64      # Gemma4 全层 is_neox_style=True，走这里
else:
    BLOCK_SIZE_HEAD = 32
```

Gemma4 全层 `is_neox_style=True`，故 `BLOCK_SIZE_HEAD=64`。neox 路径的 tile 形状为 `[BLOCK_SIZE_HEAD, pad_rope_dim//2]`（q/k 各两块）：
- `sliding_attention`：`head_dim=256`，`rope_dim=256`，`pad_rope_dim=256` → tile `[64, 128]`
- `full_attention`：`head_dim=512`，`rope_dim=128`，`pad_rope_dim=128` → tile `[64, 64]`

在 A2（910B3）上 UB（统一缓冲）上限 1572864 bits（≈192 KB），该 kernel 实际需求 1581056 bits，**溢出 8192 bits（1024 字节）**。即现有的 `BLOCK_SIZE_HEAD=64` 分块对 Gemma4 大 head_dim 仍不足以避开 UB 溢出。

### 为什么没有环境/配置解法

1. **不能关闭 triton rope 走 native 回退**：`rope_forward_oot` 用 `if HAS_TRITON:` 选择 triton/native。`HAS_TRITON`（`vllm.triton_utils.importing`）由 triton ascend driver 是否 active 决定，**无 env 强制**。且 `HAS_TRITON=False` 时 `vllm/v1/worker/block_table` 的 `_compute_slot_mapping_kernel`（`from vllm.triton_utils import triton` → placeholder）会变 no-op，推理报 `'function' object is not subscriptable`。rope 与 block_table 被 `HAS_TRITON` 死耦合。
2. **不能禁 multi-buffer**：报错提示 multi-buffer 占额外 local buffer，但 `--enable-auto-multi-buffer` 由 triton-ascend `NPUOptions.multibuffer` 控制，`NPUOptions` 不从 env 读取，无法用环境变量关闭。
3. **不能换 triton-ascend 版本**：osinfra 索引上只有 3.2.1（vllm_ascend pin 的版本），无更新版可能换 tiling。
4. `BLOCK_SIZE_HEAD` 在 kernel 调用处硬编码，无配置项。

结论：**rope UB 溢出是代码 tiling 问题**（kernel 对 Gemma4 大 head_dim 分块不足），无环境/配置解法，需改 `rope.py`。

### 修改方案

`rope_forward_triton` 里对大 head_dim 自适应缩小 `BLOCK_SIZE_HEAD`（已提交到 `fix-gemma4-mixed-pa-fia-param-utils` 分支，commit `82222df1`）：

```python
# vllm_ascend/ops/triton/rope.py  rope_forward_triton
if is_neox_style:
    BLOCK_SIZE_HEAD = 64
else:
    BLOCK_SIZE_HEAD = 32
# gemma4 large head_dim (256 sliding / 512 full-attention) overflows the A2
# (Ascend910B3) unified buffer at the default head tile. Shrink the head tile
# for large head_dim.
if head_dim > 128:
    BLOCK_SIZE_HEAD = min(BLOCK_SIZE_HEAD, 16)
```

`BLOCK_SIZE_HEAD=16` 让 UB tile 减半以上（远超 1024 字节溢出量），编译通过。只影响大 head_dim 场景，对其它模型无影响。

---

## 问题二：cudagraph 全解码图 attention 参数 unpack 不匹配（推理期）

rope 修复、模型加载/编译/cudagraph 捕获均成功后，真实 decode 推理在 `update_graph_params` 崩溃。该问题有两个阶段的报错（对应修复的两步）。

### 原始报错 A：expected 21, got 9（早期，PA 参数未包装时）

```
File "/home/vllm-ascend_tp/vllm-ascend_tp/vllm_ascend/compilation/acl_graph.py", line 297, in update_full_graph_params
    impl_cls.update_graph_params(
File "/home/vllm-ascend_tp/vllm-ascend_tp/vllm_ascend/attention/attention_v1.py", line 713, in update_graph_params
    (
ValueError: not enough values to unpack (expected 21, got 9)
RuntimeError: Worker failed with error 'not enough values to unpack (expected 21, got 9)',
please check the stack trace above for the root cause
```

### 原始报错 B：cannot unpack non-iterable PagedAttentionGraphParam（引入包装后）

引入 `PagedAttentionGraphParam` 包装 PA 参数后，错误变为：

```
File "/home/vllm-ascend_tp/vllm-ascend_tp/vllm_ascend/attention/attention_v1.py", line 772, in update_graph_params
    (
TypeError: cannot unpack non-iterable PagedAttentionGraphParam object
RuntimeError: Worker failed with error 'cannot unpack non-iterable PagedAttentionGraphParam object'
```

调用链（decode 推理 → update full graph params）：
```
vllm_ascend/worker/model_runner_v1.py  _update_full_graph_params_if_needed
  → vllm_ascend/compilation/acl_graph.py:297  update_full_graph_params
    → vllm_ascend/attention/attention_v1.py  update_graph_params  (line 462/572/737 三分支)
      → 对 captured_attn_params 逐个 unpack → 崩
```

### 根因分析

attention 后端有两条 capture/update 路径，元组格式不同：

| 路径 | capture 方法 | capture 元组 | update 分支 | update unpack |
|---|---|---|---|---|
| FIA（`full_graph_fia`） | `attention_v1.py` `full_graph_fia` | **21** 元 | 非 paged 分支（`update_graph_params` else） | 21 |
| PA（`full_graph_pa`） | `attention_v1.py` `full_graph_pa` | **9** 元 | paged 分支（`update_graph_params` if） | 9 |

- PA capture 元组（9 元）：`query, key_cache, value_cache, num_kv_heads, num_heads, scale, block_table, seq_lens, output`
- FIA capture 元组（21 元）：上述基础上多 `attn_mask, block_size, query_start_loc, softmax_lse, sparse_mode, pre_tokens, next_tokens, c8_k_aq_scale, c8_k_aq_offset, c8_v_aq_scale, c8_v_aq_offset, layer_name` 等

Gemma4 异构 head_dim 导致一个 decode graph 里**混合**了 PA（512-dim full-attn，9 元）和 FIA（256-dim sliding，21 元）参数，存进同一个 `graph_params.attn_params[num_tokens]` 列表。

而 `update_graph_params` 用**单一**分支决策（`if using_paged_attention(...)` / `elif _EXTRA_CTX.sinks` / `else`）去 unpack 整个列表，必然有一种格式的参数被按另一种 unpack → 崩。

- **报错 A（21 vs 9）**：update 走非 paged 分支（unpack 21），但列表里混入了 PA 的 9 元参数 → `expected 21, got 9`。根因：capture 侧 `forward_impl` 调 `using_paged_attention(num_tokens, vllm_config, head_size)` **传了 head_size**（512 → PA 回退 → True），而 update 侧 `update_graph_params` 调 `using_paged_attention(num_tokens, vllm_config)` **没传 head_size**，512 回退被跳过，且 decode shape 不在 `pa_shape_list`（默认空）→ 非 paged 分支。capture/update 的 PA 判定不一致。
- **报错 B（cannot unpack non-iterable）**：修复方案用 `PagedAttentionGraphParam` dataclass 包装 PA 参数（`params` + `layer_name`），在 462（PA）和 sinks 分支用 `isinstance(param, PagedAttentionGraphParam)` 分发。但 **else（FIA）分支漏了同样的 isinstance 分发**，导致混入的 PA 参数（`PagedAttentionGraphParam`，不可迭代）被按 21 元 FIA 元组 unpack → `cannot unpack non-iterable`。

### 修改方案

最终修复（`fix-gemma4-mixed-pa-fia-param-utils` 分支，commit `ff43467e` "Fix mixed PA graph param replay in FIA update"）：

1. **PA 参数包装**：`full_graph_pa` 捕获时把 9 元元组包进 `PagedAttentionGraphParam(params=..., layer_name=...)`（dataclass，定义在 `vllm_ascend/attention/utils.py`），与 FIA 参数混在同一个 `attn_params` 列表里。`layer_name` 用 `self._graph_metadata_layer_name()`（layer-aware replay 按 layer_name 解析层）。

2. **三分支统一 isinstance 分发**：`update_graph_params` 的三个分支（PA:462 / sinks:572 / FIA else:737）都加：
   ```python
   if isinstance(param, PagedAttentionGraphParam):
       update_paged_attention_graph_param(update_stream, handle, event, param, block_table, seq_lens)
       continue
   ```
   PA 参数走 `update_paged_attention_graph_param`（`attention/utils.py`，调 `_npu_paged_attention`），FIA 参数 fall through 到原 21 元 unpack。`ff43467e` 补的就是 else（FIA）分支漏掉的这一段。

这样混合列表里 PA/FIA 参数各按自己的路径 update，不再 unpack 不匹配。`PagedAttentionGraphParam` 是显式类型标记（非元组长度、非哨兵），对 PA 参数扩展鲁棒。

### 验证结论

`fix-gemma4-mixed-pa-fia-param-utils`（ff43467e）+ rope.py UB 修复 + modelslim k_eq_v 量化适配 → 量化 gemma4 W8A8 在 FULL_DECODE_ONLY cudagraph 模式下端到端推理正确：
- `25 * 4` → `100` ✓
- `用一句话介绍杭州` → `"杭州是一座将自然山水之美与现代科技之速完美融合，被誉为'人间天堂'的数字化名城。"` ✓
- 日志无 21/9、无 cannot unpack、无 UB 溢出、无推理错误。

---

## 附录：triton 环境问题（已环境修复，非代码）

启动期推理报 `'function' object is not subscriptable`（`block_table._compute_slot_mapping_kernel[(...)]`）。根因：`triton` 3.5.0 base 重装时覆盖了 `triton-ascend` 3.2.1 的 ascend 版 `libtriton.so`（126MB，含 `triton._C.libtriton.ascend` 绑定）为普通版（412MB，无 ascend 绑定）→ `import triton` 失败 → `HAS_TRITON=False` → `@triton.jit` 变 placeholder → kernel 不可下标。

修复（纯环境）：从 osinfra 索引重装 triton-ascend 3.2.1 恢复 ascend 版 `libtriton.so`：
```bash
pip install --force-reinstall --no-deps \
  --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple \
  --trusted-host triton-ascend.osinfra.cn triton-ascend==3.2.1
```
（`--no-deps` 关键，避免 triton 3.5.0 被重拉再次覆盖）

---

## 汇总

| 问题 | 类型 | 根因 | 修复 | 分支/commit |
|---|---|---|---|---|
| rope UB 溢出 | 代码（tiling） | `_triton_rope` 的 `BLOCK_SIZE_HEAD=64` 对 Gemma4 大 head_dim 分块不足 | `head_dim>128` 时 `BLOCK_SIZE_HEAD=min(.,16)` | `fix-gemma4-mixed-pa-fia-param-utils` `82222df1` |
| cudagraph unpack | 代码（PA/FIA 混合） | PA/FIA 参数混在一个列表，update 单分支 unpack 不匹配；包装后 else 分支漏 isinstance 分发 | `PagedAttentionGraphParam` 包装 + 三分支统一 isinstance 分发 | `fix-gemma4-mixed-pa-fia-param-utils` `ff43467e` |
| triton 不可下标 | 环境 | triton 3.5.0 覆盖 triton-ascend 的 ascend libtriton.so | 重装 triton-ascend 3.2.1 | （环境，非代码） |
