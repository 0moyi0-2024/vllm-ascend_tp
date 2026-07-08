# Gemma4 ModelSlim Quantization PR 讲稿

本文是 PR 讲稿版本，面向 maintainer 解释本 PR 为什么需要做、原始报错是什么、根因在哪里、每个修改点分别解决什么问题，以及为什么这个方案不会影响其它模型逻辑。

对应 PR：`[feature] Support Gemma4 ModelSlim quantization`

当前 PR 覆盖两个 Gemma4 ModelSlim W8A8_DYNAMIC 量化适配问题：

1. 31B dense 模型的 k_eq_v attention 层缺少 `v_proj` quant entry，导致 `qkv_proj` packed shard 查询时报 `KeyError`。
2. 26B MoE 模型的 expert 量化 key 和 vLLM 模块 prefix 不一致，导致 FusedMoE expert quant method 选择时报 `KeyError`。

原始报错素材来自 `error_gemma4` 分支提交 `efba95183a86a826f0977fca2a1a63d5364e4761` 的 `docs/gemma4_quant_adaptation_dense_moe.md`。

---

## 1. 开场：这个 PR 做了什么

这个 PR 只改 vLLM-Ascend 的 ModelSlim quant config 解析逻辑，用来支持 Gemma4 的两个 W8A8_DYNAMIC 量化形态：

- Gemma4 31B dense W8A8：attention 层使用 `QKVParallelLinear`，其中 full-attention k_eq_v 层没有独立 `v_proj` quant key。
- Gemma4 26B MoE W8A8：expert 权重是 W8A8_DYNAMIC，attention / dense MLP 是 FLOAT，MoE expert prefix 在 vLLM 模块路径和 ModelSlim quant description 之间存在 `moe.` infix 差异。

这个 PR 的设计目标是：

1. 让这两个 Gemma4 量化模型能完成 quant method 选择、权重加载和推理。
2. 只处理 Gemma4 已知的 key 形态差异，不扩大全局容错。
3. 保持其它模型已有的 packed shard 校验行为，尤其是缺失必要 shard 时继续暴露错误。
4. 修改范围尽量小，集中在 `vllm_ascend/quantization/modelslim_config.py`，并用单测锁住边界。

---

## 2. 背景：为什么 Gemma4 会触发 ModelSlim key 适配问题

Gemma4 的文本模型结构里有几个和常见模型不同的点：

- dense 版 attention 层统一使用 `QKVParallelLinear`，也就是 vLLM 侧模块 prefix 是 `qkv_proj`。
- full-attention 层启用 `attention_k_eq_v=True`，checkpoint 和 ModelSlim quant description 中只有 `q_proj` / `k_proj`，没有独立 `v_proj`。
- vLLM 的 Gemma4 loader 会在权重加载阶段把 `k_proj` 复制到 `v_proj` 槽位，所以缺少 `v_proj` 是模型结构预期，不是 checkpoint 损坏。
- MoE 版中 FusedMoE expert 模块在 vLLM 模块树里是 `...moe.experts...`，但 ModelSlim quant description 里保留 on-disk 的 `...experts...` 命名。

ModelSlim config 的职责是在模型模块初始化时，根据当前 module prefix 选择对应的 quant method。这个阶段早于实际权重加载，因此如果 quant config lookup 不能理解 Gemma4 的这些 key 差异，模型会在 worker 初始化阶段直接失败。

---

## 3. 原始报错一：31B dense k_eq_v 缺少 v_proj

31B dense W8A8 模型启动时，worker 在加载模型阶段失败。核心报错是：

```text
File ".../vllm_ascend/quantization/modelslim_config.py", line 624, in get_quant_method
  if self.is_layer_skipped_ascend(prefix, self.packed_modules_mapping):
File ".../vllm_ascend/quantization/modelslim_config.py", line 674, in is_layer_skipped_ascend
  is_shard_skipped = self.quant_description[shard_prefix + ".weight"] == "FLOAT"
                     ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyError: 'language_model.model.layers.5.self_attn.v_proj.weight'
```

这个错误发生在 layer 5 的 `self_attn.qkv_proj`。layer 5 是 full-attention k_eq_v 层，ModelSlim quant description 里有：

```text
language_model.model.layers.5.self_attn.q_proj.weight
language_model.model.layers.5.self_attn.k_proj.weight
```

但没有：

```text
language_model.model.layers.5.self_attn.v_proj.weight
```

因为 full-attention k_eq_v 层的 V 是由 K 复制出来的。

### 3.1 根因

`QKVParallelLinear` 是 packed module，ModelSlim config 会用 packed mapping 把：

```text
qkv_proj -> q_proj / k_proj / v_proj
```

展开，然后逐个 shard 查询 quant type / skip status。

旧逻辑的关键行为是直接下标访问：

```python
quant_description[shard_prefix + ".weight"]
self.quant_description[shard_prefix + ".weight"]
```

这对普通 q/k/v 都存在的模型没有问题，但对 Gemma4 k_eq_v 层来说，`v_proj.weight` 缺失是预期结构，所以这里会在选择 quant method 阶段提前失败，根本走不到后面的权重加载复制逻辑。

### 3.2 方案设计

这里不能简单改成 `dict.get()` 或跳过所有 missing shard，因为那会吞掉真正损坏的 quant config。例如缺少 `q_proj.weight` 或 `k_proj.weight` 应该继续失败。

最终方案是增加一个非常窄的判断：

```python
def _is_missing_k_eq_v_shard(shard_key, quant_description):
    if not shard_key.endswith(".v_proj.weight"):
        return False
    shard_prefix = shard_key[: -len("v_proj.weight")]
    return (
        f"{shard_prefix}q_proj.weight" in quant_description
        and f"{shard_prefix}k_proj.weight" in quant_description
    )
```

也就是说，只有同时满足下面条件才允许跳过：

1. 缺失的是 `v_proj.weight`。
2. 同一层的 `q_proj.weight` 存在。
3. 同一层的 `k_proj.weight` 存在。

这正好对应 Gemma4 k_eq_v 的 checkpoint 形态。

### 3.3 修改点一：`get_linear_quant_type`

修改前，packed shard 查询遇到缺失 key 会直接 `KeyError`：

```python
shard_quant_type = quant_description[shard_prefix + ".weight"]
```

修改后，只对 k_eq_v 的缺失 `v_proj` continue：

```python
shard_key = shard_prefix + ".weight"
if shard_key not in quant_description and _is_missing_k_eq_v_shard(
    shard_key, quant_description
):
    continue
shard_quant_type = quant_description[shard_key]
```

注意这里最后仍然是 `quant_description[shard_key]`，不是 `.get()`。这意味着非 k_eq_v 场景的 missing shard 仍然保持原来的失败行为。

这个修改解决的是：`create_scheme_for_layer -> get_quant_type_for_layer -> get_linear_quant_type` 里选择 linear quant scheme 时，k_eq_v 层可以从 q/k 两个现有 shard 判断出 `W8A8_DYNAMIC`。

### 3.4 修改点二：`is_layer_skipped_ascend`

`get_quant_method` 在创建 quant method 之前会先判断这个 layer 是否应该跳过量化。旧逻辑同样会直接查每个 packed shard：

```python
is_shard_skipped = self.quant_description[shard_prefix + ".weight"] == "FLOAT"
```

修改后也只对 k_eq_v 的缺失 `v_proj` continue：

```python
shard_key = shard_prefix + ".weight"
if shard_key not in self.quant_description and _is_missing_k_eq_v_shard(
    shard_key, self.quant_description
):
    continue
is_shard_skipped = self.quant_description[shard_key] == "FLOAT"
```

这里同样保留字典下标访问，所以坏配置不会被吞掉。

### 3.5 为什么这个方案安全

这个修改不会让所有 missing shard 都变成合法，只允许一种特定模式：

```text
q_proj.weight exists
k_proj.weight exists
v_proj.weight missing
```

如果缺的是 `q_proj.weight`、`k_proj.weight`，或者 q/k 并不完整，仍然会 `KeyError`。因此它不会掩盖普通模型的坏 quant description。

---

## 4. 原始报错二：26B MoE expert quant lookup 失败

26B MoE W8A8 模型启动时，worker 也在加载阶段失败。核心报错是：

```text
File ".../vllm_ascend/quantization/modelslim_config.py", line 389, in get_linear_quant_type
  quant_type = quant_description[prefix + ".weight"]
               ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
KeyError: 'language_model.model.layers.0.moe.experts.weight'
```

这个错误说明 FusedMoE 的 prefix 是：

```text
language_model.model.layers.0.moe.experts
```

但 ModelSlim config 没有把它当成 packed experts 处理，于是直接查：

```text
language_model.model.layers.0.moe.experts.weight
```

而 quant description 实际上是 per-expert shard 形式：

```text
language_model.model.layers.0.experts.0.gate_proj.weight
language_model.model.layers.0.experts.0.up_proj.weight
language_model.model.layers.0.experts.0.down_proj.weight
```

### 4.1 根因 A：Gemma4 不在 packed_modules_model_mapping

已有的 MoE 模型一般会在 `packed_modules_model_mapping` 里注册：

```python
"experts": [
    "experts.0.gate_proj",
    "experts.0.up_proj",
    "experts.0.down_proj",
]
```

这样当 module prefix 是 `...experts` 时，ModelSlim config 会展开到三个 expert shard，并检查它们的 quant type 是否一致。

但 Gemma4 / Gemma4 text 原来不在这个 mapping 里，所以 `experts` 不会走 packed shard 分支，而是直接查 `experts.weight`，导致 KeyError。

### 4.2 根因 B：vLLM 模块路径带 `.moe.experts`，quant description 是 `.experts`

Gemma4 的 vLLM 模型权重加载逻辑会把 checkpoint 里的：

```text
...experts.0.gate_proj...
```

映射到模块树里的：

```text
...moe.experts.0.gate_proj...
```

也就是说：

- 模型 module prefix：`...layers.0.moe.experts`
- quant description key：`...layers.0.experts.0.gate_proj.weight`

如果只加 packed mapping，还会查到：

```text
...layers.0.moe.experts.0.gate_proj.weight
```

仍然和 quant description 对不上。

### 4.3 根因 C：model_type 可能是 gemma4 或 gemma4_text

多模态包装和 text config 路径下，`model_type` 可能出现 `gemma4` 或 `gemma4_text`。如果只加一个，另一种路径仍然不生效。

---

## 5. MoE 修改点级说明

### 5.1 修改点三：给 gemma4 / gemma4_text 注册 packed_modules_model_mapping

新增：

```python
"gemma4": {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "experts": [
        "experts.0.gate_proj",
        "experts.0.up_proj",
        "experts.0.down_proj",
    ],
},
"gemma4_text": {
    ...
},
```

这里三个 mapping 分别对应：

- `qkv_proj`：Gemma4 attention 的 packed QKV。
- `gate_up_proj`：dense MLP 的 gate/up fused projection。
- `experts`：MoE FusedMoE expert 的 gate/up/down shards。

这个修改解决的是：Gemma4 的 `experts` 不再被当成普通 linear 去查 `experts.weight`，而是按 per-expert packed shards 查询。

为什么同时加 `qkv_proj` 和 `gate_up_proj`：

- dense Gemma4 和 MoE Gemma4 共享 Gemma4 模型结构。
- `qkv_proj` 是前面 k_eq_v 修复依赖的 packed mapping。
- `gate_up_proj` 是 Gemma4 MLP 的标准 fused projection。
- 未使用的 mapping 不会主动生效，只有当前 prefix 的最后一段命中时才会走对应 packed shard 逻辑。

### 5.2 修改点四：给 gemma4 / gemma4_text 增加 scoped substr mapping

新增：

```python
"gemma4": {
    ".moe.experts": ".experts",
},
"gemma4_text": {
    ".moe.experts": ".experts",
},
```

这发生在 `quant_prefix_mapper` 阶段，只对当前 `model_type` 的查询 prefix 生效。

例子：

```text
input prefix:
language_model.model.layers.0.moe.experts

mapped prefix:
language_model.model.layers.0.experts
```

然后 packed experts mapping 会展开成：

```text
language_model.model.layers.0.experts.0.gate_proj.weight
language_model.model.layers.0.experts.0.up_proj.weight
language_model.model.layers.0.experts.0.down_proj.weight
```

这就和 ModelSlim quant description 的 on-disk key 对齐了。

### 5.3 为什么没有做全局 quant_description alias

中间方案曾考虑过在 `_apply_extra_quant_adaptations` 里给所有 `.experts.` key 额外生成一份 `.moe.experts.` alias。但最终没有采用这个设计。

原因是它会变成全局行为：

```text
any_model.layers.0.experts.0.gate_proj.weight
-> any_model.layers.0.moe.experts.0.gate_proj.weight
```

这可能改变其它 MoE 模型的 quant_description key space，影响面比 Gemma4 问题本身更大。

最终方案选择只在 `QUANT_MODEL_SUBSTR_MAPPINGS` 里对 `gemma4/gemma4_text` 做 prefix 查询映射。这样：

- 只影响 Gemma4。
- 不复制 quant_description。
- 不改变其它模型 key。
- 不引入全局 alias。

这也是本 PR “能上库”的关键收敛点。

---

## 6. 单测设计

这个 PR 增加的 UT 主要是为了锁住两个边界：Gemma4 能跑通，非 Gemma4 不被误伤。

### 6.1 k_eq_v 测试

新增测试：缺失 `v_proj.weight` 但 q/k 存在时，允许判断 quant type：

```python
q_proj.weight = W8A8_DYNAMIC
k_proj.weight = W8A8_DYNAMIC
v_proj.weight missing
```

预期：

- `get_linear_quant_type(...) == "W8A8_DYNAMIC"`
- `is_layer_skipped_ascend(...) == False`

同时新增反向测试：缺失必要 shard 时继续报 `KeyError`。

### 6.2 Gemma4 MoE prefix 测试

新增测试确认：

```text
language_model.model.layers.0.moe.experts
```

在 `gemma4` / `gemma4_text` 下会映射成：

```text
language_model.model.layers.0.experts
```

并且可以根据：

```text
experts.0.gate_proj.weight
experts.0.up_proj.weight
experts.0.down_proj.weight
```

判断出 `W8A8_DYNAMIC`。

### 6.3 packed mapping 覆盖测试

新增测试确认 `gemma4` / `gemma4_text` 都注册了：

```python
qkv_proj -> q_proj / k_proj / v_proj
gate_up_proj -> gate_proj / up_proj
experts -> experts.0.gate_proj / experts.0.up_proj / experts.0.down_proj
```

### 6.4 skip / mixed shard 边界测试

新增测试确认：

- expert 三个 shard 都是 FLOAT 时，整个 FusedMoE 被视为 skip。
- expert shards 混合 FLOAT 和 W8A8_DYNAMIC 时，继续报 `ValueError`。

这保证 packed experts 的一致性校验仍然存在。

### 6.5 非 Gemma4 不改写测试

新增测试确认非 Gemma4 模型，例如 `qwen3_5_moe`，不会把：

```text
model.layers.0.moe.experts
```

改写掉。

同时确认 `_apply_extra_quant_adaptations` 不会给所有模型额外加 `.moe.experts` alias。

---

## 7. 为什么这个 PR 不影响其它模型

这个 PR 对非 Gemma4 模型的影响面很小：

1. `_is_missing_k_eq_v_shard` 只允许缺失 `v_proj.weight` 且 q/k 都存在的情况。
   其它 missing shard 继续通过原来的 dict 下标访问报错。

2. Gemma4 MoE 的 `.moe.experts -> .experts` 映射只挂在：

   ```python
   QUANT_MODEL_SUBSTR_MAPPINGS["gemma4"]
   QUANT_MODEL_SUBSTR_MAPPINGS["gemma4_text"]
   ```

   非 Gemma4 model_type 不会走这条映射。

3. 没有做全局 quant_description alias。
   这避免了其它 MoE 模型突然多出一套 `.moe.experts` quant key。

4. packed_modules_model_mapping 只新增 `gemma4` / `gemma4_text` 条目。
   其它模型原有 mapping 不变。

---

## 8. 修改点和原始报错的对应关系

| 原始问题 | 报错 key | 修改点 | 作用 |
|---|---|---|---|
| 31B dense k_eq_v 缺少 V | `...self_attn.v_proj.weight` | `_is_missing_k_eq_v_shard` + 两处 packed shard loop | 只允许缺失 k_eq_v 的 `v_proj` |
| 26B MoE experts 没有 packed mapping | `...moe.experts.weight` | `packed_modules_model_mapping["gemma4"]` / `["gemma4_text"]` | 让 FusedMoE experts 展开成 gate/up/down shards |
| 26B MoE prefix 不一致 | model prefix 有 `.moe.experts`，quant key 是 `.experts` | `QUANT_MODEL_SUBSTR_MAPPINGS` | Gemma4 查询时 strip `.moe` |
| 防止影响其它模型 | 潜在全局 alias 风险 | 不在 `_apply_extra_quant_adaptations` 加 `.moe.experts` alias | 保持非 Gemma4 行为不变 |

---

## 9. 可以怎么向 reviewer 总结

可以用下面这段话作为 PR review 里的 summary：

> This PR adapts Ascend ModelSlim quant config lookup for Gemma4 W8A8_DYNAMIC checkpoints. Gemma4 k_eq_v full-attention layers intentionally omit `v_proj` quant entries because V is replicated from K at load time, so the packed qkv lookup now skips only that expected missing `v_proj` shard when q/k entries both exist. For Gemma4 MoE, the FusedMoE module prefix contains `.moe.experts` while ModelSlim keeps on-disk `.experts` keys, so the PR adds Gemma4-scoped packed mappings and prefix mapping for `gemma4` / `gemma4_text`. The changes are scoped to Gemma4 and preserve the previous KeyError/ValueError behavior for malformed packed quant descriptions in other models.

---

## 10. 验证情况

本地轻量验证：

```bash
python3 -m ruff check vllm_ascend/quantization/modelslim_config.py tests/ut/quantization/test_modelslim_config.py
python3 -m ruff format --check vllm_ascend/quantization/modelslim_config.py tests/ut/quantization/test_modelslim_config.py
python3 -m py_compile vllm_ascend/quantization/modelslim_config.py tests/ut/quantization/test_modelslim_config.py
git diff --check
```

当前本地系统 Python 缺 `torch`，所以完整：

```bash
python3 -m pytest tests/ut/quantization/test_modelslim_config.py -q
```

在本机无法执行，会在导入 `tests/ut/conftest.py` 时失败：

```text
ModuleNotFoundError: No module named 'torch'
```

模型侧验证来自适配过程：

- 31B dense W8A8：原始 `v_proj.weight` KeyError 消失，模型可加载并推理。
- 26B MoE W8A8 expert-only：原始 `moe.experts.weight` KeyError 消失，模型可加载并推理。

