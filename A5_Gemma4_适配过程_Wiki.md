# A5 适配 Gemma4 过程 Wiki

## 1. 背景

本次工作目标是让 Gemma4 在 vllm-ascend 的 A5 环境上具备可用的图模式执行能力，并补齐相关 MoE 路径兼容。

Gemma4 相比常规 dense 模型和部分已有 MoE 模型，有几个需要额外关注的点：

1. attention 层配置存在差异，不能简单假设所有层的图模式参数完全同构。
2. 图模式 capture/replay/update 阶段，需要保证 replay 时使用的 metadata 与 capture 时的 attention layer 一一对应。
3. FIA workspace 的需求不只由 `num_tokens` 决定，具体 attention op 的 shape 和参数也会影响 workspace 大小。
4. MoE 路径中，Gemma4 的 routing function、activation name 和 hf config 字段与已有模型存在差异。

因此，这次适配不是单点修复，而是围绕 attention graph metadata、FIA workspace、MoE routing、MoE activation 四个方向完成兼容。

## 2. 目标

本次 A5 Gemma4 适配的目标如下：

1. Gemma4 在 A5 图模式下可以正常启动并执行。
2. attention graph replay 阶段能使用正确层的 metadata。
3. FIA 算子 workspace 不因为跨层复用而出现 size mismatch。
4. Gemma4 MoE 的 custom routing function 可以正常调用。
5. Gemma4 MoE 的 `gelu` / `gelu_tanh` activation 可以按模型语义执行。
6. 尽量控制改动范围，保留旧路径 fallback，降低对其他模型的影响。

## 3. 问题一：FIA workspace size mismatch

### 3.1 现象

在 A5 图模式启动 Gemma4 时，报错出现在 `aclnnFusedInferAttentionScoreV5` 图模式 replay/update 相关路径中。

典型错误是：

```text
Value 55664128 for parameter workspaceLen is invalid.
Reason: The passed workspace size 55664128 does not meet the workspace size 104871936 actually required by the operator.
```

从错误信息可以看出，传入算子的 workspace 小于该次算子实际需要的 workspace。

### 3.2 初步判断

原实现中，FIA graph workspace 存在：

```python
graph_params.workspaces[num_tokens]
```

也就是说，workspace 的缓存粒度只有 `num_tokens`，属于 bucket 级缓存。

但 Gemma4 的 attention layer 可能不是完全同构的。同一个 `num_tokens` bucket 下，不同 attention op 的参数和 shape 可能不同，例如不同层可能需要不同大小的 FIA workspace。

如果第一个 capture 到的 layer workspace 较小，后续另一个 layer 复用了这个 workspace，就可能出现：

```text
passed workspace size < required workspace size
```

### 3.3 根因

FIA workspace 的正确粒度应该是 attention op 级别，而不是单纯的 `num_tokens` bucket 级别。

原先的缓存方式：

```python
workspace = graph_params.workspaces.get(num_tokens)
if workspace is None:
    workspace = get_max_workspace(...)
    update_graph_params_workspaces(num_tokens, workspace)
```

隐含假设是：

同一个 `num_tokens` bucket 下所有 FIA op 都可以共用同一个 workspace。

这个假设对 Gemma4 不成立。

### 3.4 最终方案

将 FIA workspace 从 bucket 级共享缓存改为 per-op 保存。

capture 阶段：

```python
workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(...)

attn_params = attn_params + (
    weak_ref_tensors(workspace),
    self._graph_metadata_layer_name(layer),
)
graph_params.attn_params[num_tokens].append(attn_params)
```

update/replay 阶段：

```python
workspace, layer_name = _split_optional_workspace_and_layer_name(...)
if workspace is None:
    workspace = graph_params.workspaces.get(num_tokens)
```

这样每个 captured attention op 都保存自己的 workspace。旧格式 graph param 仍然可以 fallback 到 `graph_params.workspaces[num_tokens]`。

### 3.5 为什么不继续调用 `update_graph_params_workspaces`

`update_graph_params_workspaces(num_tokens, workspace)` 仍然是 bucket 粒度。继续在 FIA path 使用这个接口，会重新引入跨层共享 workspace 的问题。

本次没有删除 workspace 生命周期管理，而是把 FIA workspace 的保存位置从：

```python
graph_params.workspaces[num_tokens]
```

迁移为：

```python
graph_params.attn_params[num_tokens][op_index].workspace
```

由于 `attn_params` 本身就是 graph params 的一部分，workspace 仍然会跟随 graph params 保持引用。

### 3.6 对 draft model 的影响

这套逻辑同样适用于 draft graph。

代码中会先选择当前 graph params：

```python
if _EXTRA_CTX.is_draft_model:
    if _EXTRA_CTX.is_draft_model_prefill:
        graph_params = get_draft_graph_prefill_params()
    else:
        graph_params = get_draft_graph_params()
else:
    graph_params = get_graph_params()
```

workspace 随后 append 到当前 `graph_params.attn_params[num_tokens]` 中。

因此 target graph、draft graph、draft prefill graph 都会保存自己的 per-op workspace，不会因为删除 `update_draft_graph_params_workspaces` 调用而丢失 workspace。

反而，旧逻辑在 draft prefill 场景下存在一个潜在不一致：代码选择的是 `get_draft_graph_prefill_params()`，但更新 workspace 调的是 `update_draft_graph_params_workspaces()`，不是 prefill 对应的 workspace updater。per-op 保存避免了这种对象不一致。

## 4. 问题二：graph replay metadata 可能错配

### 4.1 现象

图模式中，capture 阶段保存的是一组 attention graph params；replay/update 阶段需要从 `forward_context.attn_metadata` 中取最新 metadata 更新 graph task。

原逻辑主要依赖 `attn_keys` 的顺序和 `graph_params.attn_params[num_tokens]` 顺序一致：

```python
for key, param, handle, event in zip(attn_keys, graph_params.attn_params[num_tokens], ...):
    ...
    seq_lens = attn_metadata[key].seq_lens_list
```

对于结构更复杂的模型或 spec decode 场景，这个顺序假设不一定稳定。

### 4.2 旧方案：`_ATTN_KEYS_BUFFER` + regex sort

旧代码中曾存在一个全局 `_ATTN_KEYS_BUFFER`：

```python
_ATTN_KEYS_BUFFER = None
```

target path 中会从 key 字符串中用正则提取数字，然后排序：

```python
def extract_layer_index(key: str) -> int:
    match = re.search(r"(\d+)", key)
    return int(match.group(1)) if match else 0

attn_keys_tmp.sort(key=extract_layer_index)
_ATTN_KEYS_BUFFER = attn_keys_tmp
```

这个逻辑的目的，是在 DFlash 等场景里尝试恢复 target model 的 layer 顺序。

### 4.3 旧方案的问题

这个方案存在几个限制：

1. 它依赖 key 字符串里的数字能代表 layer index。
2. 如果 key 命名变化，或者 key 中有多个数字，就可能排序错误。
3. `_ATTN_KEYS_BUFFER` 是全局缓存，不区分模型、bucket、graph params 数量、draft/target 场景。
4. 它只能修正顺序，不能表达“这个 graph param 原本属于哪个 layer”。

Gemma4 更需要的是显式的 layer 级绑定，而不是对 key 顺序做启发式排序。

### 4.4 最终方案：capture 时保存 layer name

新增状态：

```python
self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
self._layer_name: str | None = None
```

forward 入口记录当前 layer：

```python
self._layer_name = layer.layer_name
```

新增 helper：

```python
def _graph_metadata_layer_name(self, layer: AttentionLayer | None = None) -> str | None:
    layer_name = layer.layer_name if layer is not None else self._layer_name
    return self.kv_sharing_target_layer_name or layer_name
```

capture graph params 时，将 layer name 追加到 param tuple 尾部。

update 阶段通过 helper 选择 metadata key：

```python
def _metadata_key(attn_metadata: dict, fallback_key: str, layer_name: str | None) -> str:
    if layer_name is not None and layer_name in attn_metadata:
        return layer_name
    return fallback_key
```

### 4.5 为什么保留 fallback

如果旧格式 graph param 没有 layer name，或者某些路径 metadata 中找不到该 layer name，则 fallback 到原来的 key。

这样能保证：

1. 新路径可以按 layer name 精确绑定。
2. 旧路径不因为缺少 layer name 直接失败。
3. 兼容 Paged Attention、FIA、FIA v2 不同 graph param tuple 格式。

### 4.6 为什么要支持 `kv_sharing_target_layer_name`

某些 attention layer 可能共享其他层的 KV。此时 replay/update 应该使用 target layer 的 metadata，而不是当前 layer 的 metadata。

因此 `_graph_metadata_layer_name` 中优先返回：

```python
self.kv_sharing_target_layer_name
```

如果没有 kv sharing，再返回当前 layer name。

## 5. 问题三：attn keys 数量和 graph params 数量不一致

### 5.1 现象

update 逻辑中使用 `zip`：

```python
zip(attn_keys, graph_params.attn_params[num_tokens], handles, events)
```

如果 `attn_keys` 数量小于 graph params 数量，后面的 graph params 不会被 update。

### 5.2 方案

新增：

```python
def _expand_attn_keys_for_graph_params(attn_keys: list[str], graph_param_count: int) -> list[str]:
    if len(attn_keys) == 0:
        return []
    return [attn_keys[index % len(attn_keys)] for index in range(graph_param_count)]
```

这个函数只负责扩展循环次数，不负责确定最终 metadata。

最终 metadata 选择仍然由：

```python
_metadata_key(attn_metadata, key, layer_name)
```

完成。

### 5.3 与旧排序逻辑的区别

旧排序逻辑试图通过 key 字符串排序决定 layer 对应关系。

新逻辑中，扩展 key 只是让循环覆盖所有 graph params；真正的对应关系来自 captured op 里保存的 layer name。

## 6. 问题四：Gemma4 MoE config 字段兼容

### 6.1 现象

A5 MoE 通信方式选择逻辑需要读取每 token 选择多少 expert。

旧代码直接访问：

```python
vllm_config.model_config.hf_text_config.num_experts_per_tok
```

但不同模型配置字段名可能不同。有些模型使用：

```python
top_k_experts
```

### 6.2 方案

改为：

```python
num_experts_per_tok = getattr(
    vllm_config.model_config.hf_text_config,
    "num_experts_per_tok",
    getattr(vllm_config.model_config.hf_text_config, "top_k_experts", 1),
)
```

### 6.3 效果

兼容 `num_experts_per_tok` 和 `top_k_experts` 两种字段命名。

如果两个字段都不存在，fallback 到 1，避免启动阶段属性错误。

## 7. 问题五：custom routing function 参数兼容

### 7.1 现象

MoE native select experts 路径中，原来固定传入：

```python
num_experts=num_experts
```

但部分模型的 custom routing function 不接受 `num_experts` 参数，固定传入会导致 TypeError。

### 7.2 方案

新增 signature inspect：

```python
def _inspect_custom_routing_accepts_num_experts(custom_routing_function: Callable) -> bool:
    try:
        signature = inspect.signature(custom_routing_function)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == "num_experts"
        for parameter in signature.parameters.values()
    )
```

再通过 cache 包装：

```python
@functools.cache
def _custom_routing_accepts_num_experts(custom_routing_function: Callable) -> bool:
    return _inspect_custom_routing_accepts_num_experts(custom_routing_function)
```

调用时动态构造 kwargs：

```python
routing_kwargs = {
    "hidden_states": hidden_states,
    "gating_output": router_logits,
    "topk": top_k,
    "renormalize": renormalize,
}
if _custom_routing_accepts_num_experts(custom_routing_function):
    routing_kwargs["num_experts"] = num_experts
topk_weights, topk_ids = custom_routing_function(**routing_kwargs)
```

### 7.3 为什么需要 cache

`inspect.signature` 是 CPU 侧反射操作。如果每次 forward、每层 MoE 都执行，会带来额外 overhead。

custom routing function 通常在模型生命周期内固定，所以签名检查结果可以缓存。review 中也指出了这一点，因此最终使用 `functools.cache`。

## 8. 问题六：Gemma4 MoE activation 兼容

### 8.1 现象

Gemma4 MoE 可能使用 `gelu` 或 `gelu_tanh` activation。

旧代码中，非 `swigluoai` 的情况会落到：

```python
torch_npu.npu_swiglu(gate_up_out)
```

如果模型配置实际是 GELU gate，这个 fallback 数学语义不正确。

### 8.2 方案

在 `unquant_apply_mlp` 中新增：

```python
elif act_name == "gelu":
    gate, up = gate_up_out.chunk(2, dim=-1)
    gate_up_out = torch.nn.functional.gelu(gate) * up
elif act_name == "gelu_tanh":
    gate, up = gate_up_out.chunk(2, dim=-1)
    gate_up_out = torch.nn.functional.gelu(gate, approximate="tanh") * up
```

### 8.3 语义

MoE gate/up projection 输出被拆成两半：

1. gate 分支做 GELU 或 tanh-approx GELU。
2. up 分支保持原值。
3. 两者相乘得到激活后的中间结果。

这与 `gelu` / `gelu_tanh` 的模型配置语义一致。

## 9. 最终代码改动范围

最终 PR 修改 5 个文件：

1. `tests/ut/attention/a2/test_attention_v1.py`
   - 新增 attention graph helper 单测。

2. `vllm_ascend/ascend_forward_context.py`
   - A5 MoE comm selection 兼容 `top_k_experts` 字段。

3. `vllm_ascend/attention/attention_v1.py`
   - 新增 graph helper。
   - 保存 `kv_sharing_target_layer_name` 和 `_layer_name`。
   - capture graph params 时保存 layer name。
   - update graph params 时按 layer name 选择 metadata。
   - FIA/FIA v2 workspace 改为 per-op 保存。
   - 删除 `_ATTN_KEYS_BUFFER` 的全局 regex sort workaround。
   - 保留 workspace 和 metadata fallback，兼容旧 graph params。

4. `vllm_ascend/ops/fused_moe/experts_selector.py`
   - custom routing function 是否接受 `num_experts` 改为动态判断。
   - 使用 `functools.cache` 缓存 signature 检查结果。

5. `vllm_ascend/ops/fused_moe/moe_mlp.py`
   - `unquant_apply_mlp` 支持 `gelu` 和 `gelu_tanh`。

## 10. 验证情况

本地做过的基础检查：

```bash
python3 -m py_compile \
  vllm_ascend/ops/fused_moe/experts_selector.py \
  vllm_ascend/attention/attention_v1.py \
  tests/ut/attention/a2/test_attention_v1.py

python3 -m ruff check \
  vllm_ascend/ops/fused_moe/experts_selector.py \
  vllm_ascend/attention/attention_v1.py \
  tests/ut/attention/a2/test_attention_v1.py

python3 -m ruff format --check \
  vllm_ascend/ops/fused_moe/experts_selector.py \
  vllm_ascend/attention/attention_v1.py \
  tests/ut/attention/a2/test_attention_v1.py
```

PR 中新增的单测覆盖：

1. layer name 存在时，metadata key 优先使用 layer name。
2. layer name 不存在时，metadata key fallback 到原 key。
3. graph param 尾部 workspace 和 layer name 能正确解析。

## 11. 兼容性设计

本次修改遵循三个兼容原则：

### 11.1 旧 graph param 格式兼容

新增字段放在 tuple 尾部，并通过 `_split_optional_workspace_and_layer_name` 解析。

如果没有新增字段：

```python
workspace = None
layer_name = None
```

后续仍然 fallback 到旧逻辑。

### 11.2 metadata fallback

如果保存的 layer name 不在当前 metadata 中：

```python
return fallback_key
```

这样不会强制所有路径都必须支持 layer name。

### 11.3 workspace fallback

如果 param 中没有 per-op workspace：

```python
workspace = graph_params.workspaces.get(num_tokens)
```

这样旧 graph params 仍然可以使用 bucket 级 workspace。

## 12. 风险和注意事项

### 12.1 tuple 结构继续复杂化

为了控制改动范围，本次没有把 graph param tuple 重构为 dataclass，而是在尾部追加可选字段。

优点是 diff 小、兼容旧格式。

缺点是 tuple 结构仍然不够直观，后续维护时需要注意字段顺序。

后续可以考虑单独 PR 把 graph param 结构化。

### 12.2 per-op workspace 增加 workspace 对象数量

FIA 每个 captured op 保存自己的 workspace，会比 bucket 级共享缓存保存更多 workspace 引用。

这是为了 correctness 做的取舍。考虑到 workspace mismatch 会直接导致图模式启动或 replay 失败，这个取舍是必要的。

### 12.3 `inspect.signature` 缓存依赖 callable 可 hash

`functools.cache` 要求参数可 hash。常规 Python function / bound method 是可 hash 的。

如果未来出现不可 hash 的 custom callable 对象，可能需要进一步做 try/fallback。不过当前模型 routing function 通常是函数或可 hash callable。

## 13. 后续建议

1. 将 attention graph param tuple 结构化，减少通过 tuple position 维护字段的风险。
2. 对 graph params 增加更完整的单测，覆盖 FIA、FIA v2、Paged Attention 三类 update path。
3. 如果后续还有模型出现 layer 级 workspace 差异，可以考虑把 workspace 管理抽象为 op-level workspace manager。
4. 对 custom routing function signature 兼容逻辑补充更多 MoE 模型单测。
5. 对 `gelu` / `gelu_tanh` 路径增加数值对齐测试，确保 fallback path 和模型配置一致。

## 14. 总结

这次 A5 适配 Gemma4 的核心经验是：

图模式下不能只依赖执行顺序和 bucket 级缓存来恢复运行时状态。对于 Gemma4 这种层间配置存在差异的模型，attention graph replay 需要更明确的 layer/op 级绑定。

最终方案通过两点解决：

1. metadata 绑定从 attn key 顺序匹配升级为 layer name 显式匹配。
2. FIA workspace 从 `num_tokens` bucket 级缓存升级为 captured attention op 级保存。

MoE 部分则补齐了模型配置字段、routing function 签名和 activation 语义差异。

整体改动尽量保持最小化，并为旧路径保留 fallback，从而在支持 Gemma4 A5 图模式的同时降低对其他模型和现有执行路径的影响。
