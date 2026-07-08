# Gemma4 W8A8 量化适配：31B Dense + 26B MoE

本文记录让 ModelSlim W8A8_DYNAMIC 量化版 Gemma4（31B dense + 26B MoE）在 vLLM-Ascend（A2 / Ascend 910B3）上加载并推理所做的量化适配：两份完整原始报错、根因定位、修改方案与验证结果。

> 硬件：4 × Ascend 910B3（A2），TP=4
> 31B dense：`/home/wangminghua/gemma4_w8a8_rollback`
> 26B MoE（128 experts）：`/home/wangminghua/gemma4_moe_w8a8_only_experts`（仅 expert 量化，attn/mlp 为 FLOAT）
> 启动配置：`--tensor-parallel-size 4 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`

---

## 背景：Gemma4 异构 head_dim + k_eq_v + MoE

Gemma4 文本部分关键结构：
- `sliding_attention`：`head_dim=256`，`num_key_value_heads=16`
- `full_attention`：`global_head_dim=512`，`num_global_key_value_heads=4`，`attention_k_eq_v=True`

vLLM 的 `Gemma4ForConditionalGeneration` 对所有层统一建 `QKVParallelLinear`（`qkv_proj` 融合），full_attention（k_eq_v）层在加载时把 `k_proj` 复制到 V 槽位（safetensors 里这些层**没有 v_proj 权重**）。MoE 版所有层都是 MoE（128 experts），expert 权重（gate_proj/up_proj/down_proj）为 W8A8_DYNAMIC，attention/mlp 为 FLOAT。

---

## 问题一：31B Dense——full_attention（k_eq_v）层 KeyError

### 原始报错

启动 W8A8 量化版 31B dense 模型，在 **worker 加载阶段**（构建量化层时）崩溃：

```
Traceback (most recent call last):
  File ".../vllm/v1/executor/multiproc_executor.py", line 855, in worker_main
    worker = WorkerProc(*args, **kwargs)
  ...
  File ".../vllm_ascend/worker/worker.py", line 699, in load_model
    self.model_runner.load_model()
  ...
  File ".../vllm_ascend/quantization/modelslim_config.py", line 624, in get_quant_method
    if self.is_layer_skipped_ascend(prefix, self.packed_modules_mapping):
  File ".../vllm_ascend/quantization/modelslim_config.py", line 674, in is_layer_skipped_ascend
    is_shard_skipped = self.quant_description[shard_prefix + ".weight"] == "FLOAT"
                       ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyError: 'language_model.model.layers.5.self_attn.v_proj.weight' [58980,58980][multiproc_executor.py:888,worker_main]
[UC][E] WorkerProc failed to start.
RuntimeError: Engine core initialization failed. See root cause above.
```

4 个 worker 都报同样的 `KeyError`，整个模型加载失败。报错发生在 layer 5（`full_attention`，k_eq_v）的 `qkv_proj`（`QKVParallelLinear`），查 `v_proj.weight` 这个 quant 描述键时找不到。

### 根因

检查 `quant_model_description.json` 与 safetensors：

- **sliding_attention 层（如 layer 0）**：有 `q_proj` / `k_proj` / `v_proj`（各带 weight_scale/weight_offset）。
- **full_attention（k_eq_v）层（如 layer 5）**：只有 `q_proj` / `k_proj`，**没有 v_proj**（V 由 K 在加载时复制，safetensors 里也没有）。

`Gemma4ForCausalLM.packed_modules_mapping = {"qkv_proj": ["q_proj","k_proj","v_proj"], ...}` 被赋给 `quant_config.packed_modules_mapping`。

`AscendModelSlimConfig.is_layer_skipped_ascend` / `get_linear_quant_type` 在迭代 packed shard 时**直接下标取 `quant_description[shard_prefix + ".weight"]`**，迭代到 k_eq_v 层缺失的 `v_proj` 时 → `KeyError`。

### 修改

文件：`vllm_ascend/quantization/modelslim_config.py`

在两处 packed-shard 迭代里**跳过 quant 描述中不存在的 shard**：

**1. `get_linear_quant_type`**
```python
for shard_prefix in shard_prefixes:
    shard_key = shard_prefix + ".weight"
    # Skip shards that have no dedicated quant entry — k_eq_v layers
    # (e.g. Gemma4 full-attention) where v_proj is replicated from k_proj
    # at load time, so only q_proj/k_proj exist in the quant description.
    if shard_key not in quant_description:
        continue
    shard_quant_type = quant_description[shard_key]
    ...
```

**2. `is_layer_skipped_ascend`**
```python
for shard_prefix in shard_prefixes:
    shard_key = shard_prefix + ".weight"
    if shard_key not in self.quant_description:
        continue
    is_shard_skipped = self.quant_description[shard_key] == "FLOAT"
    ...
```

只影响"缺失 shard"场景（k_eq_v 复制），对其它模型无影响。

### 验证

修复后 31B dense 模型加载/编译/推理全通过（FULL_DECODE_ONLY cudagraph，配合 rope UB 修复 + PA/FIA unpack 修复）。推理测试：`25 * 4` → `100` ✓；GPQA-Diamond 10 题 90% 准确率。

---

## 问题二：26B MoE——expert 量化 KeyError + `moe.` 前缀不匹配

### 原始报错

启动 W8A8 量化版 26B MoE 模型（仅 expert 量化），在 **worker 加载阶段**崩溃：

```
Traceback (most recent call last):
  File ".../vllm/v1/executor/multiproc_executor.py", line 855, in worker_main
    worker = WorkerProc(*args, **kwargs)
  ...
  File ".../vllm_ascend/worker/worker.py", line 754, in load_model
    self.model_runner.load_model()
  File ".../vllm/model_executor/model_loader/__init__.py", line 143, in get_model
    ...
  File ".../vllm_ascend/quantization/modelslim_config.py", line 389, in get_linear_quant_type
    quant_type = quant_description[prefix + ".weight"]
                 ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
KeyError: 'language_model.model.layers.0.moe.experts.weight' [383723,383723][multiproc_executor.py:888,worker_main]
[UC][E] WorkerProc failed to start.
RuntimeError: Engine core initialization failed. See root cause above.
```

4 个 worker 都报同样的 `KeyError`，加载失败。

### 根因（三个子问题）

MoE 模型有三层叠加的 prefix/struct 不匹配：

**子问题 1：gemma4 不在 `packed_modules_model_mapping`**

gemma4（dense + MoE）不在 `packed_modules_model_mapping` 中，`get_quant_method` 不会为 gemma4 设置 `self.packed_modules_mapping`。FusedMoE 模块（prefix `language_model.model.layers.0.moe.experts`）的 `experts` 不在 mapping 中 → `get_linear_quant_type` 走 `else` 分支直接查 `quant_description["language_model.model.layers.0.moe.experts.weight"]` → `KeyError`（quant 描述里是 per-expert 键，没有 fused `experts.weight`）。

**子问题 2：`moe.` 前缀不匹配**

vLLM 的 gemma4 模型在 weight loading 时用 `re.sub(r"(?<!\.moe)\.experts\.(\d+)\.", r".moe.experts.\1.", name)`（gemma4.py:89）把 checkpoint 的 `.experts.0.` 重映射为模型路径的 `.moe.experts.0.`。但 quant 描述（来自 checkpoint）保持 `.experts.0.`（无 `moe.`）。所以模型 prefix `...moe.experts.0.gate_proj` 与 quant 描述 `...experts.0.gate_proj` 不匹配——即使 packed mapping 加了 `experts`，shard 查找 `...moe.experts.0.gate_proj.weight` 也找不到。

**子问题 3：model_type 解析为 `gemma4_text`**

多模态模型的 `hf_config.model_type` 可能解析为 text_config 的 `gemma4_text` 而非 `gemma4`，需在映射表中两个都加。

### 修改

文件：`vllm_ascend/quantization/modelslim_config.py`（3 处，同 k_eq_v 修复一起）

**1. `packed_modules_model_mapping` 加 `gemma4` + `gemma4_text`**

```python
"gemma4": {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
},
"gemma4_text": { ... 同上 ... },
```

让 gemma4 dense/MoE 的 `qkv_proj`/`gate_up_proj`/`experts` 融合模块按 packed shard 查找。`qkv_proj` 和 `gate_up_proj` 兼容 dense 模型，`experts` 兼容 MoE 模型（各自未使用的条目无副作用）。

**2. `QUANT_MODEL_SUBSTR_MAPPINGS` 加 `gemma4`/`gemma4_text` `.moe.` → `.` 映射**

```python
"gemma4": {".moe.experts": ".experts"},
"gemma4_text": {".moe.experts": ".experts"},
```

`quant_prefix_mapper` 里把模型 prefix 的 `.moe.experts` 还原成 quant 描述的 `.experts`，让 packed shard 查找匹配。

**3. `_apply_extra_quant_adaptations` 加 `moe.` 前缀适配**

```python
moe_extra = {}
for k in self.quant_description:
    if ".experts." in k and ".moe.experts." not in k:
        new_k = re.sub(r"(?<!\.moe)\.experts\.(\d+)\.", r".moe.experts.\1.", k)
        if new_k != k:
            moe_extra[new_k] = self.quant_description[k]
self.quant_description.update(moe_extra)
```

镜像 gemma4.py:89 的 regex，给 quant 描述里的 expert 键加 `moe.`，与模型路径 `...moe.experts.0...` 一致。

> 以上 3 处 MoE 修改与 k_eq_v 修复（问题一）一起，构成了完整的 gemma4 量化适配。所有修改仅限 `modelslim_config.py` 一个文件。

### 验证

修复后 26B MoE 模型加载/编译/推理全通过（FULL_DECODE_ONLY cudigraph）。推理测试：`25 * 4` → `100` ✓。无 KeyError、无 UB 溢出、无 OOM。

---

## 总结：修改文件与提交位置

| 修改 | 文件 | 分支 / commit | 说明 |
|---|---|---|---|
| k_eq_v 容错（问题一） | `modelslim_config.py` | `msmodel_gemma4` / `b50823ac` | 跳过缺失 v_proj shard |
| MoE expert 量化适配（问题二） | `modelslim_config.py` | `msmodel_gemma4` / `a4cc1ece` | packed mapping + substr mapping + `moe.` 前缀 |
| 本两次修改合并 | `modelslim_config.py` | `msmodel_gemma4` 分支 | k_eq_v + MoE expert 完整适配 |

所有量化适配修改仅限 `vllm_ascend/quantization/modelslim_config.py` 一个文件。rope UB 溢出修复和 PA/FIA unpack 修复在 `fix-gemma4-mixed-pa-fia-param-utils` 分支（独立修复），详见 [gemma4_a2_ub_unpack_errors.md](gemma4_a2_ub_unpack_errors.md)。

## 复现步骤

### 31B dense
```bash
git checkout msmodel_gemma4  # 含 quant 适配
git checkout fix-gemma4-mixed-pa-fia-param-utils -- vllm_ascend/ops/triton/rope.py vllm_ascend/attention/attention_v1.py vllm_ascend/attention/utils.py  # rope UB + PA/FIA 修复
vllm serve /home/wangminghua/gemma4_w8a8_rollback --tensor-parallel-size 4 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --port 1025 ...
```

### 26B MoE
```bash
git checkout msmodel_gemma4
git checkout fix-gemma4-mixed-pa-fia-param-utils -- vllm_ascend/ops/triton/rope.py vllm_ascend/attention/attention_v1.py vllm_ascend/attention/utils.py
vllm serve /home/wangminghua/gemma4_moe_w8a8_only_experts --tensor-parallel-size 4 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --port 1025 ...
```

> 该环境 `http_proxy=127.0.0.1:1080`，curl 访问 localhost 需 `--noproxy '*'`。
