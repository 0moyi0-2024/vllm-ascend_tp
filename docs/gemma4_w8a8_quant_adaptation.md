# Gemma4 W8A8 量化适配过程

本文记录让 ModelSlim W8A8_DYNAMIC 量化版 Gemma4 在 vLLM-Ascend（A2 / Ascend 910B3）上加载并推理所做的一次最小量化适配：原始报错、根因定位、修改方案与验证。

> 硬件：4 × Ascend 910B3（A2），TP=4
> 量化权重：`/home/wangminghua/gemma4_w8a8_rollback`（ModelSlim W8A8_DYNAMIC，`quant_model_description.json`，`model_quant_type=W8A8_DYNAMIC`）
> 启动配置：`--tensor-parallel-size 4 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` + gemma4 tool/reasoning parser

---

## 1. 背景与目标

Gemma4 文本部分是异构 head_dim 的 GQA：
- `sliding_attention`：`head_dim=256`，`num_key_value_heads=16`，rope `partial_rotary_factor=1.0`
- `full_attention`：`global_head_dim=512`，`num_global_key_value_heads=4`，`attention_k_eq_v=True`，rope `partial_rotary_factor=0.25`

vLLM 的 `Gemma4ForConditionalGeneration` 对**所有层**统一建 `QKVParallelLinear`（`qkv_proj` 融合），full_attention（k_eq_v）层在加载时把 `k_proj` 复制到 V 槽位（safetensors 里这些层**没有 v_proj 权重**，只有 q_proj/k_proj）。

目标：让 W8A8 量化版权重在该环境上加载并推理，适配尽量小，能上库。

---

## 2. 原始报错

启动量化版 gemma4，在**加载阶段**（worker 初始化、构建量化层时）崩溃：

```
Traceback (most recent call last):
  File ".../vllm/v1/executor/multiproc_executor.py", line 855, in worker_main
    worker = WorkerProc(*args, **kwargs)
  ...
  File ".../vllm_ascend/worker/worker.py", line 699, in load_model
    self.model_runner.load_model()
  File ".../vllm_ascend/worker/model_runner_v1.py", line 3713, in load_model
    self.model: nn.Module = get_model(vllm_config=self.vllm_config)
  ...
  File ".../vllm_ascend/ops/linear.py", line 126, in __init__
    self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
  File ".../vllm_ascend/quantization/modelslim_config.py", line 624, in get_quant_method
    if self.is_layer_skipped_ascend(prefix, self.packed_modules_mapping):
  File ".../vllm_ascend/quantization/modelslim_config.py", line 674, in is_layer_skipped_ascend
    is_shard_skipped = self.quant_description[shard_prefix + ".weight"] == "FLOAT"
                       ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyError: 'language_model.model.layers.5.self_attn.v_proj.weight' [58980,58980][multiproc_executor.py:888,worker_main]
[2026-07-06 13:32:08.035464][UC][E] WorkerProc failed to start.
```

4 个 worker 都报同样的 `KeyError`，整个量化模型加载失败。

报错发生在 layer 5（`full_attention`，k_eq_v）的 `qkv_proj`（`QKVParallelLinear`），查 `v_proj.weight` 这个 quant 描述键时找不到。

---

## 3. 根因定位

### 3.1 quant 描述与 safetensors 的实际键

检查 `quant_model_description.json` 与 `quant_model_weights.safetensors.index.json`：

- **sliding_attention 层（如 layer 0）**：有 `q_proj` / `k_proj` / `v_proj`（各带 weight / weight_scale / weight_offset）。
- **full_attention（k_eq_v）层（如 layer 5）**：只有 `q_proj` / `k_proj`，**没有 v_proj**（V 由 K 在加载时复制，safetensors 里也没有 v_proj 权重）。

```
# layer 0 (sliding)
model.language_model.layers.0.self_attn.q_proj.weight   = W8A8_DYNAMIC
model.language_model.layers.0.self_attn.k_proj.weight   = W8A8_DYNAMIC
model.language_model.layers.0.self_attn.v_proj.weight   = W8A8_DYNAMIC   # 有

# layer 5 (full/k_eq_v)
model.language_model.layers.5.self_attn.q_proj.weight   = W8A8_DYNAMIC
model.language_model.layers.5.self_attn.k_proj.weight   = W8A8_DYNAMIC
# 没有 v_proj（V 由 K 复制）
```

### 3.2 packed shard 迭代直接下标取值

`Gemma4ForCausalLM.packed_modules_mapping = {"qkv_proj": ["q_proj","k_proj","v_proj"], ...}` 被赋给 `quant_config.packed_modules_mapping`。

`AscendModelSlimConfig.is_layer_skipped_ascend` / `get_linear_quant_type` 在迭代 packed shard 时**直接下标取 `quant_description[shard_prefix + ".weight"]`**：

```python
# get_linear_quant_type (modelslim_config.py:367)
for shard_prefix in shard_prefixes:
    shard_quant_type = quant_description[shard_prefix + ".weight"]   # ← KeyError
    ...

# is_layer_skipped_ascend (modelslim_config.py:674)
for shard_prefix in shard_prefixes:
    is_shard_skipped = self.quant_description[shard_prefix + ".weight"] == "FLOAT"   # ← KeyError
    ...
```

对 full_attention（k_eq_v）层，`qkv_proj` 的 shards = `[q_proj, k_proj, v_proj]`，迭代到 `v_proj` 时查 `language_model.model.layers.5.self_attn.v_proj.weight` → 该层无此键 → `KeyError`，整个量化模型加载失败。

### 3.3 前缀映射本身没问题

`Gemma4ForConditionalGeneration.hf_to_vllm_mapper`（`model.language_model.` → `language_model.model.`）已把 quant 描述键正确重映射到 vLLM 命名（`apply_vllm_mapper` 在模型 `__new__` 时调用）。报错时 q_proj/k_proj 都能命中（循环到第 3 个 shard v_proj 才报错），证明前缀映射正确，问题纯粹是 k_eq_v 层缺 v_proj shard。

---

## 4. 修改方案（最小适配）

文件：`vllm_ascend/quantization/modelslim_config.py`

在两处 packed-shard 迭代里**跳过 quant 描述中不存在的 shard**（k_eq_v 层的 v_proj 由 k_proj 复制、无独立条目，其量化属性跟随已存在的 shard）：

### 4.1 `get_linear_quant_type`

```python
for shard_prefix in shard_prefixes:
    shard_key = shard_prefix + ".weight"
    # Skip shards that have no dedicated quant entry. This happens for
    # k_eq_v layers (e.g. Gemma4 full-attention) where v_proj is
    # replicated from k_proj at load time, so only q_proj/k_proj exist in
    # the quant description. Their quant type follows the present shards.
    if shard_key not in quant_description:
        continue
    shard_quant_type = quant_description[shard_key]
    ...
```

### 4.2 `is_layer_skipped_ascend`

```python
for shard_prefix in shard_prefixes:
    shard_key = shard_prefix + ".weight"
    # Skip shards absent from the quant description (e.g. v_proj of
    # k_eq_v layers like Gemma4 full-attention, where V is replicated
    # from K and has no dedicated quant entry). Their skip status
    # follows the present shards.
    if shard_key not in self.quant_description:
        continue
    is_shard_skipped = self.quant_description[shard_key] == "FLOAT"
    ...
```

只影响"缺失 shard"的场景（如 k_eq_v 复制），对其它模型无影响；上游 vLLM 的 `is_layer_skipped` 本就用列表匹配、不会 `KeyError`，ascend 版只是补齐这个容错。

### 4.3 加载侧

`QKVParallelLinear` 对 k_eq_v 层在加载时把 `k_proj` 复制到 V 槽位（`_weight_iterator` remapping），量化 scale/offset 同样复制——这部分 vLLM 已支持，无需改。`get_quant_method` 返回正确的 `AscendLinearMethod(W8A8_DYNAMIC)` 后，权重加载自动走通。

---

## 5. 验证

修复后量化模型加载/编译/服务全通过：

- ✅ `KeyError` 消失，日志出现 `Applied hf_to_vllm_mapper to quant_description keys`。
- ✅ 权重加载成功（每卡 ~10GB），W8A8_DYNAMIC 量化方法正确选型（q/k/v/o/gate/up 均 W8A8_DYNAMIC，k_eq_v 层正确处理）。
- ✅ `--enforce-eager` 下推理正确；`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`（配合 rope UB 修复 + PA/FIA unpack 修复，见 [gemma4_a2_ub_unpack_errors.md](gemma4_a2_ub_unpack_errors.md)）下推理也正确。
- 推理测试：
  - `25 * 4` → `100` ✓
  - `用一句话介绍杭州` → `"杭州是一座将自然山水之美与现代科技之速完美融合，被誉为'人间天堂'的数字化名城。"` ✓

量化适配（`modelslim_config.py` k_eq_v 修复）已提交到 `msmodel_gemma4` 分支（commit `5be6ec95`）。

---

## 6. 复现步骤

1. 确认 triton 可用：`python -c "from triton._C.libtriton.ascend import ir"`。若失败，按 [gemma4_a2_ub_unpack_errors.md](gemma4_a2_ub_unpack_errors.md) 附录重装 triton-ascend 3.2.1。
2. 应用量化适配（`msmodel_gemma4` 分支，commit `5be6ec95`）。
3. （cudagraph 模式）应用 rope UB 修复 + PA/FIA unpack 修复（`fix-gemma4-mixed-pa-fia-param-utils` 分支 `82222df1` / `ff43467e`）。
4. 启动（eager 绕过 cudagraph 问题即可验证量化本身）：
   ```bash
   vllm serve /home/wangminghua/gemma4_w8a8_rollback --served-model-name gemma4_w8a8_rollback \
     --tensor-parallel-size 4 --enable-auto-tool-choice --tool-call-parser gemma4 \
     --reasoning-parser gemma4 --enable-prefix-caching --gpu-memory-utilization 0.92 \
     --port 1025 --limit-mm-per-prompt '{"image": 1}' --enforce-eager
   ```
5. 推理验证：
   ```bash
   curl --noproxy '*' http://127.0.0.1:1025/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"gemma4_w8a8_rollback","messages":[{"role":"user","content":"计算 25 * 4 等于多少？只给数字"}],"max_tokens":40}'
   ```

> 注：该环境 `http_proxy=127.0.0.1:1080`，curl 访问 localhost 需 `--noproxy '*'`。

---

## 7. 文件与提交清单

| 项 | 文件 | 分支 / commit | 说明 |
|---|---|---|---|
| 量化适配 | `vllm_ascend/quantization/modelslim_config.py` | `msmodel_gemma4` / `5be6ec95` | k_eq_v 缺失 v_proj shard 容错（`get_linear_quant_type` + `is_layer_skipped_ascend`） |
| rope UB 修复 | `vllm_ascend/ops/triton/rope.py` | `fix-gemma4-mixed-pa-fia-param-utils` / `82222df1` | 大 head_dim 自适应 BLOCK_SIZE_HEAD（详见 ub_unpack 文档） |
| PA/FIA unpack 修复 | `vllm_ascend/attention/attention_v1.py` | `fix-gemma4-mixed-pa-fia-param-utils` / `ff43467e` | PagedAttentionGraphParam + 三分支 isinstance 分发（详见 ub_unpack 文档） |
| triton 环境 | 系统 pip 包 | — | 重装 triton-ascend 3.2.1（详见 ub_unpack 文档附录） |
