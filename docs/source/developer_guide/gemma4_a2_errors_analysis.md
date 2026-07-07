# Gemma4 on A2 (Ascend 910B3) 错误分析

本文记录在 vLLM-Ascend（分支 `A2SupportGemma4`）上跑通 Gemma4（含 W8A8 量化版）时遇到的两个**非量化、非环境**的代码问题：rope triton kernel UB 溢出、cudagraph 全解码图 attention 参数更新 unpack 不匹配。给出原始报错与根因分析。

> 硬件：4 × Ascend 910B3（A2），TP=4
> 模型：Gemma4 31B（浮点 `/home/xty/gemma4/31B`；W8A8 量化 `/home/wangminghua/gemma4_w8a8_rollback`）
> 配置：`--tensor-parallel-size 4 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`

## 背景

Gemma4 文本部分是异构 head_dim GQA：
- `sliding_attention`：`head_dim=256`，`num_key_value_heads=16`，rope `partial_rotary_factor=1.0`
- `full_attention`：`global_head_dim=512`，`num_global_key_value_heads=4`，`attention_k_eq_v=True`，rope `partial_rotary_factor=0.25`

vLLM 的 `Gemma4ForConditionalGeneration` 对所有层统一建 `QKVParallelLinear`，full_attention（k_eq_v）层在加载时把 `k_proj` 复制到 V 槽位。W8A8 量化适配（`modelslim_config.py` 处理 k_eq_v 层缺失 `v_proj` shard 的 KeyError）已另行提交，与本节两个问题无关。

---

## 问题一：rope triton kernel UB 溢出（编译期）

### 原始报错

triton 修复后（见附录），`profile run` 编译阶段 `vllm_ascend/ops/triton/rope.py` 的 `_triton_rope` kernel 在 BiShengIR 编译失败：

```
loc("vllm_ascend/ops/triton/rope.py":24:0): error: ub overflow,
requires 1581056 bits while 1572864 bits available!
(possible reason: tiling basic block is too large or block number is more
than what user expect due to multi-buffer feature is enabled and some ops
need extra local buffer.)
[ERROR] Failed to run BiShengIR pipeline

subprocess.CalledProcessError: Command '['.../triton/backends/ascend/bishengir/bin/bishengir-compile',
'/tmp/.../kernel.ttadapter.mlir', '--target=Ascend910B3',
'--enable-auto-multi-buffer=True', '--enable-auto-bind-sub-block=True',
'--enable-hfusion-compile=true', '--enable-hivm-compile=true',
'--enable-triton-kernel-compile=true', '-o', '/tmp/.../kernel']'
returned non-zero exit status 1.
```

调用链（profile run → model forward → rope）：
```
vllm_ascend/ops/rotary_embedding.py:168 rope_forward_oot
  → vllm_ascend/ops/triton/rope.py:285 rope_forward_triton
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

结论：**rope UB 溢出是代码 tiling 问题**（kernel 对 Gemma4 大 head_dim 分块不足），无环境/配置解法，需改 `rope.py`（如对大 head_dim 自适应缩小 `BLOCK_SIZE_HEAD`）。该问题与量化无关，浮点 Gemma4 在 A2 上同样触发。

---

## 问题二：cudagraph 全解码图 attention 参数更新 unpack 不匹配（推理期）

### 原始报错

rope 修复、模型加载/编译/cudagraph 捕获均成功后，真实 decode 推理在 `update_graph_params` 崩溃：

```
File ".../vllm_ascend/compilation/acl_graph.py", line 297, in update_full_graph_params
    impl_cls.update_graph_params(
File ".../vllm_ascend/attention/attention_v1.py", line 713, in update_graph_params
    (
ValueError: not enough values to unpack (expected 21, got 9)
RuntimeError: Worker failed with error 'not enough values to unpack (expected 21, got 9)'
```

### 根因分析

attention 后端有两条 capture/update 路径，元组格式不同：

| 路径 | capture 方法 | capture 元组 | update 分支 | update unpack |
|---|---|---|---|---|
| FIA（`full_graph_fia`） | `attention_v1.py:922` | **21** 元 | 非 paged 分支（`:713`） | 21 |
| PA（`full_graph_pa`） | `attention_v1.py:1139` | **9** 元 | paged 分支（`:479`） | 9 |

- PA capture 元组（9 元）：`query, key_cache, value_cache, num_kv_heads, num_heads, scale, block_table, seq_lens, output`
- FIA capture 元组（21 元）：上述基础上多 `attn_mask, block_size, query_start_loc, softmax_lse, sparse_mode, pre_tokens, next_tokens, c8_k_aq_scale, c8_k_aq_offset, c8_v_aq_scale, c8_v_aq_offset, layer_name` 等

关键：**capture 和 update 用不同的 `using_paged_attention` 判定**，导致 full_attention(512) 层 capture 走 PA（9 元）、update 走非 paged（期望 21 元）→ `expected 21, got 9`。

- **capture 侧**（`forward_impl`，`attention_v1.py:1469`）**传了 head_size**：
  ```python
  if (attn_state == DecodeOnly
      and self.sliding_window is None
      and using_paged_attention(num_tokens, self.vllm_config, self.head_size)):  # 传 head_size
      output = self.forward_paged_attention(...)   # → full_graph_pa，9 元
  else:
      output = self.forward_fused_infer_attention(...)  # → full_graph_fia，21 元
  ```
- **update 侧**（`update_graph_params`，`attention_v1.py:462`）**没传 head_size**：
  ```python
  if using_paged_attention(num_tokens, vllm_config):   # 没传 head_size！
      # paged 分支，unpack 9
  else:
      # 非 paged 分支，unpack 21
  ```

而 `using_paged_attention`（`vllm_ascend/attention/utils.py:87`）对 Gemma4 512 head 有专门回退：
```python
# TODO: Remove this fallback when A2/A3 FIA TND supports Gemma4's
#  512-dim global attention heads. Decode can use PA directly; prefill is
#  handled by the device adaptor.
if head_size == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE:   # = 512
    return True
```

所以对 full_attention（head_size=512）层：
- **capture**：`using_paged_attention(..., 512)` → 512 回退 → True → PA → 存 **9** 元组
- **update**：`using_paged_attention(num_tokens, vllm_config)`（head_size 缺省 None）→ 跳过 512 回退 → 落到 `cudagraph_mode==FULL_DECODE_ONLY and runtime_shape in pa_shape_list`；`pa_shape_list` 默认空 → False → 非 paged 分支 → unpack **21** → `expected 21, got 9`

即 **`update_graph_params` 漏传 `head_size`，导致 512 head 层的 PA/FIA 路径在 capture/update 间不一致**。这是代码 bug，与量化无关，浮点 Gemma4 同样触发。

### 验证：`pa_shape_list` 配置可影响 update 分支但无法根治

`pa_shape_list` 来自 `additional_config`（vLLM `--additional-config` JSON），可配置。尝试用非代码手段让 update 也走 PA 分支：

```bash
--additional-config '{"pa_shape_list":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]}'
```
（即 decode cudagraph 捕获尺寸；`update_full_graph_params` 传入的是 `num_tokens_padded`，是这些捕获尺寸之一）

结果：错误**反转**为 `too many values to unpack (expected 9)`：
```
ValueError: too many values to unpack (expected 9)
RuntimeError: Worker failed with error 'too many values to unpack (expected 9)'
```

说明 `pa_shape_list` 确实把 update 切到了 PA 分支（期望 9），但 capture 侧仍有 21 元组（FIA）——即 `sliding_attention`（head_size=256）层的 capture 没随 `pa_shape_list` 切到 PA。`forward_impl` 路由 PA 还需 `self.sliding_window is None`；Gemma4 sliding 层 `sliding_window` 非空，不满足该条件，故 sliding 层 capture 仍走 FIA（21 元），而 update 走 PA（期望 9）→ `expected 9, got 21`。

> 注：`forward_impl` 的 PA 路由条件是 `attn_state == DecodeOnly and self.sliding_window is None and using_paged_attention(...)`。Gemma4 sliding_attention 层带 sliding_window，被排除在 PA 之外，强制走 FIA。因此 `pa_shape_list` 无法让 sliding 层 capture 切 PA，capture/update 仍不一致。

结论：**`pa_shape_list` 配置无法根治**，因为 capture 侧 PA 路由还受 `sliding_window is None` 限制，与 update 侧判定不对齐。根本修复需改代码：让 `update_graph_params` 与 `forward_impl` 用一致的 PA 判定（传 head_size + 同样的 sliding_window 等条件），或让 capture/update 的元组格式统一。

### 临时绕过

`--enforce-eager`（关闭 cudagraph）不触发 `update_full_graph_params`，可绕过该问题完成推理验证（量化输出正确）。

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

## 问题三：`PagedAttentionGraphParam` unpack 崩溃（fix 分支推理期）

### 背景

为修问题二，在 `fix-gemma4-mixed-pa-fia-param-utils` 分支（commit `d9664f0d`，把 mixed PA replay helper 移到 `attention/utils.py`）引入 `PagedAttentionGraphParam` dataclass 包装 PA 参数，并在 `update_graph_params` 用 `isinstance` 分发：

```python
@dataclass
class PagedAttentionGraphParam:
    params: tuple
    layer_name: str | None
```

`full_graph_pa` 捕获时把 PA 9 元组包成 `PagedAttentionGraphParam(params, layer_name)`。

### 原始报错

加载/编译/cudagraph 捕获/服务启动均成功，**真实 decode 推理**时崩溃：

```
Traceback (most recent call last):
  ...
  File ".../vllm_ascend/worker/model_runner_v1.py", line 2882, in _model_forward
    self._update_full_graph_params_if_needed(
  File ".../vllm_ascend/worker/model_runner_v1.py", line 2842, in _update_full_graph_params_if_needed
    update_full_graph_params(
  File ".../vllm_ascend/compilation/acl_graph.py", line 297, in update_full_graph_params
    impl_cls.update_graph_params(
  File ".../vllm_ascend/attention/attention_v1.py", line 730, in update_graph_params
    (
TypeError: cannot unpack non-iterable PagedAttentionGraphParam object
RuntimeError: Worker failed with error 'cannot unpack non-iterable PagedAttentionGraphParam object'
```

### 根因分析

`update_graph_params` 有三个分支，`isinstance(param, PagedAttentionGraphParam)` 分发**只加了两处，漏了 else（FIA）分支**：

| 分支 | 行 | `isinstance(PagedAttentionGraphParam)` 分发 | unpack 元数 |
|---|---|---|---|
| `if using_paged_attention(...)`（PA） | 464 → 480 | ✅ 有（`param = param.params` 解包） | 9 |
| `elif _EXTRA_CTX.sinks:`（FIA-v2） | 521 → 572 | ✅ 有（PA update + `continue`） | 15 |
| `else:`（FIA） | 636 → 730 | ❌ **无** | 21 |

Gemma4 在 `FULL_DECODE_ONLY` 下：`using_paged_attention(num_tokens, vllm_config)`（不传 head_size）→ False（跳过 464 分支），`_EXTRA_CTX.sinks` → False（跳过 521 分支），落入 `else`（636）FIA 分支。该分支循环里 730 行直接对 `param` 做 21 元 unpack，而 full-attention（512）层的 `param` 是 `PagedAttentionGraphParam`（不可迭代）→ `cannot unpack non-iterable`。

即：**fix 把 PA 参数包成了不可迭代的 `PagedAttentionGraphParam`，但只在 PA 分支和 sinks 分支处理了它，else（FIA）分支没处理**——而 Gemma4 恰好走 else 分支，于是 512-head 层的 PA 参数在 FIA 21 元 unpack 处崩。

### 修复方向（未应用，待确认）

在 `else`（FIA）分支循环体开头补上与 `sinks` 分支一致的 `isinstance(param, PagedAttentionGraphParam)` 分发：命中则调 `update_paged_attention_graph_param(...)` 做 PA update 并 `continue`，否则走 21 元 FIA unpack。462 与 sinks 分支已有此分发，else 分支对齐即可。

### 复现

```bash
# 分支 fix-gemma4-mixed-pa-fia-param-utils（含 d9664f0d）+ 本地 rope.py / modelslim_config.py(k_eq_v) 补丁
vllm serve /home/wangminghua/gemma4_w8a8_rollback --tensor-parallel-size 4 \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --enable-prefix-caching --gpu-memory-utilization 0.92 --port 1025 \
  --limit-mm-per-prompt '{"image": 1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
# 启动成功；首条推理请求 500：
curl --noproxy '*' http://127.0.0.1:1025/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma4_w8a8_rollback","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
```

---

## 汇总

| 问题 | 类型 | 根因 | 是否有非代码解法 |
|---|---|---|---|
| rope UB 溢出 | 代码（tiling） | `_triton_rope` 的 `BLOCK_SIZE_HEAD=64` 对 Gemma4 大 head_dim 分块不足 | 否（HAS_TRITON 耦合 rope/block_table；multi-buffer 无 env；只有 3.2.1） |
| cudagraph 21 vs 9 | 代码（PA 判定不一致） | `update_graph_params` 漏传 `head_size`，且 `forward_impl` PA 路由还要求 `sliding_window is None` | 否（`pa_shape_list` 仅反转错误，sliding 层 capture 仍 FIA） |
| PagedAttentionGraphParam unpack 崩溃 | 代码（fix 分支分发遗漏） | `update_graph_params` 的 else（FIA）分支未对 `PagedAttentionGraphParam` 做 isinstance 分发，Gemma4 走 else → 21 元 unpack 崩 | 否（需在 else 分支补 isinstance 分发） |
| triton 不可下标 | 环境 | triton 3.5.0 覆盖 triton-ascend 的 ascend libtriton.so | 是（重装 triton-ascend 3.2.1） |
