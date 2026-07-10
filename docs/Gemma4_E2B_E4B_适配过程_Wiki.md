# Gemma4 E2B/E4B 适配讲稿

这份讲稿主要用于给 maintainer 或同事解释当前 E2B/E4B 适配 PR 做了什么、为什么必须这么做，以及为什么这个修改不会影响其它模型。

当前 PR：<https://github.com/vllm-project/vllm-ascend/pull/11791>

---

## 1. 开场：这个 PR 一句话在做什么

各位好，这个 PR 是为了补齐 Gemma4 E2B/E4B 在 vLLM Ascend 上的 KV sharing cache 语义。

简单说，Gemma4 E2B/E4B 里面有一类特殊 attention layer，它自己不维护独立的 K/V cache，而是复用前面某个 layer 的 K/V cache。

vLLM 模型层和 worker 层已经知道这件事了：

- 有些 layer 会设置 `kv_sharing_target_layer_name`。
- worker 会跳过这些 layer 的独立 KV cache spec。
- worker 会让这些 layer 复用 target layer 的 KV cache。

但是 attention backend 里还有一个缺口：这些 KV-sharing target layer 进入 `reshape_and_cache` 时，仍然会像普通 layer 一样写 cache。这样就会把 producer layer 写好的 K/V 覆盖掉。

所以这次 PR 的核心修改很小：

> 如果当前 attention layer 是 KV-sharing target layer，就不要再写 KV cache，只保留 cache 引用，让后续 attention 读取共享 cache。

---

## 2. 为什么 E2B/E4B 会遇到这个问题

普通 decoder-only 模型里，每一层 attention 一般都有自己的 KV cache：

```text
layer 0 写 layer 0 的 cache
layer 1 写 layer 1 的 cache
layer 2 写 layer 2 的 cache
```

这种情况下，`reshape_and_cache` 每层都写一次 cache 是合理的。

但 Gemma4 E2B/E4B 的一些 layer 是 KV-sharing target layer。它的语义更像这样：

```text
producer layer 写 cache
target layer 不写 cache
target layer 只读取 producer layer 的 cache
```

vLLM 在构建 Gemma4 attention 时，会为这种 target layer 设置：

```python
kv_sharing_target_layer_name = "xxx.layers.<target_idx>.self_attn.attn"
```

这句话的意思不是“当前 layer 也要写 target 的 cache”，而是：

> 当前 layer 的 K/V cache 来自另一个 layer。

所以 target layer 仍然要算 Q，仍然要执行 attention，但它不应该把自己的 K/V 写进共享 cache。

---

## 3. 原问题：target layer 覆盖了 producer layer 的 cache

原来的流程大概是这样的：

```text
worker 层:
  识别到 target layer 有 kv_sharing_target_layer_name
  不给 target layer 分配独立 cache
  把 target layer 的 cache 映射到 producer layer cache

attention backend:
  reshape_and_cache 仍然正常执行
  target layer 把自己的 K/V 写入传进来的 cache
```

问题就出在最后一步。

因为 target layer 拿到的 cache 实际上就是 producer layer 的 cache。它如果继续写，就不是写自己的 cache，而是在覆盖 producer layer cache。

这会导致后续 attention 读到错误的 K/V，表现上可能是：

- 输出异常。
- 精度下降。
- E2B/E4B 的 KV sharing 语义和 vLLM 原生实现不一致。

这个问题不是 FIA/PA 算子选择问题，也不是 graph workspace 问题。它发生得更早，在 attention 前面的 cache 写入阶段。

---

## 4. 修改点一：普通 KV cache 路径跳过 target layer 写 cache

第一个修改点在：

```text
vllm_ascend/attention/attention_v1.py
```

具体位置是 `AscendAttentionBackendImpl.reshape_and_cache`。

原来这里会先绑定 cache：

```python
self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
```

然后继续调用：

```python
DeviceOperator.reshape_and_cache(...)
```

这一步会真正把当前 layer 的 K/V 写入 cache。

现在的修改是在真正写 cache 前加一个判断：

```python
if self.kv_sharing_target_layer_name is not None:
    if self.is_kv_producer:
        attn_metadata.reshape_cache_event.record()
    return query, key, value, output
```

这段代码的意思是：

1. 如果当前 layer 是 KV-sharing target layer，就不要调用 `DeviceOperator.reshape_and_cache`。
2. 但是前面的 `self.key_cache` 和 `self.value_cache` 仍然会绑定。
3. 后续 attention 仍然可以通过这两个引用读到 producer layer 的 cache。
4. 返回值保持原来的四元组结构，不改变调用链。

这里最关键的是：**跳过的是 cache 写入，不是跳过 attention 计算。**

target layer 后续仍然会执行 attention，只是它使用共享 cache。

---

## 5. 修改点二：C8/NZ cache 路径也做同样处理

第二个修改点还是在：

```text
vllm_ascend/attention/attention_v1.py
```

具体位置是 `AscendC8AttentionBackendImpl._reshape_and_cache`。

这条路径是 C8/NZ cache 写入路径，会调用：

```python
torch_npu.npu_scatter_pa_kv_cache(...)
```

如果只改普通 `reshape_and_cache`，量化或 NZ cache 场景还是可能覆盖共享 cache。所以 C8/NZ 路径也必须做同样的判断：

```python
if self.kv_sharing_target_layer_name is not None:
    if self.is_kv_producer:
        attn_metadata.reshape_cache_event.record()
    return query, key, value, output
```

这保证了两条 cache 写入路径语义一致：

```text
普通 cache 路径：target layer 不写共享 cache
C8/NZ cache 路径：target layer 也不写共享 cache
```

这点对后续量化场景比较重要，因为不能只让浮点模型正确，量化 cache 路径仍然错误。

---

## 6. 修改点三：补充单测保护这个语义

第三个修改点在：

```text
tests/ut/attention/a2/test_attention_v1.py
```

这里新增了两个单测。

### 6.1 普通 cache 路径单测

第一个测试是：

```python
test_kv_sharing_target_skips_cache_write
```

它做的事情是：

1. 构造一个带 `kv_sharing_target_layer_name` 的 `AscendAttentionBackendImpl`。
2. mock 掉 `DeviceOperator.reshape_and_cache`。
3. 调用 `reshape_and_cache`。
4. 验证 `DeviceOperator.reshape_and_cache` 没有被调用。
5. 验证 `key_cache` 和 `value_cache` 仍然绑定到了传入的 `kv_cache`。
6. 验证返回的还是原始 `query, key, value, output`。

这个测试证明普通 cache 路径不会再覆盖 shared cache。

### 6.2 C8/NZ cache 路径单测

第二个测试是：

```python
test_c8_kv_sharing_target_skips_nz_cache_write
```

它做的事情类似：

1. 构造一个带 `kv_sharing_target_layer_name` 的 `AscendC8AttentionBackendImpl`。
2. mock 掉 `torch_npu.npu_scatter_pa_kv_cache`。
3. 调用 `_reshape_and_cache`。
4. 验证 C8/NZ cache 写入算子没有被调用。
5. 验证 cache 引用和返回值都保持预期。

这个测试证明量化/NZ cache 路径也遵守同样的 KV sharing 语义。

---

## 7. 为什么这个改动不会影响其它模型

这个问题 maintainer 可能会重点问，所以这里要讲清楚。

### 7.1 普通模型不受影响

普通模型没有 `kv_sharing_target_layer_name`。

新增逻辑的入口条件是：

```python
self.kv_sharing_target_layer_name is not None
```

所以普通模型不会进入这个分支，仍然走原来的 cache write 逻辑。

### 7.2 普通 Gemma4 layer 不受影响

Gemma4 里也不是所有 layer 都是 KV-sharing target layer。

producer layer 和普通 layer 的 `kv_sharing_target_layer_name` 仍然是 `None`，它们会继续正常写自己的 cache。

只有真正复用别人 cache 的 target layer 会跳过写 cache。

### 7.3 不改变 attention 算子选择

这个 PR 没有修改：

- FIA / PA 的选择逻辑。
- A2/A3 large-head fallback。
- graph replay。
- graph workspace。
- MoE routing。
- activation 算子。

它只改 cache 写入阶段，而且只对 KV-sharing target layer 生效。

所以它不是一个扩大影响面的 graph 改动，也不是一个模型全局特化逻辑。

### 7.4 cache 读取能力没有被删掉

虽然 target layer 不再写 cache，但它仍然会绑定：

```python
self.key_cache
self.value_cache
```

也就是说，后续 attention 仍然能读取 cache。

区别只是：

```text
旧逻辑：target layer 先覆盖共享 cache，再读取 cache
新逻辑：target layer 不覆盖共享 cache，直接读取 producer cache
```

这个新逻辑才符合 vLLM 原生 KV sharing 的语义。

---

## 8. 当前主干已经具备哪些基础能力

这次 PR 不是从零开始支持 E2B/E4B。主干里已经有一些 Gemma4 支持能力：

1. worker 侧已经识别 `kv_sharing_target_layer_name`，并跳过 target layer 的独立 KV cache spec。
2. worker 侧已经把共享 layer 的 cache 映射到 target layer cache。
3. A2/A3 对 Gemma4 512 head full attention 已经有 fallback 逻辑。
4. graph replay 已经有 layer-aware metadata 能力。
5. MoE 路径已经支持 Gemma4 使用的 `gelu_tanh` activation 和自定义 routing function。

所以本 PR 做的是补最后一段：

```text
worker 已经不分配 target layer 独立 cache
attention backend 也不要再让 target layer 写共享 cache
```

---

## 9. 当前 PR 状态和 CI 情况

当前 PR 是：

```text
https://github.com/vllm-project/vllm-ascend/pull/11791
```

当前代码改动很小：

```text
vllm_ascend/attention/attention_v1.py
tests/ut/attention/a2/test_attention_v1.py
```

当前 CI 里 `DCO` 已经通过。

`lint-and-select-tests` 当前失败不是代码 lint 问题，而是 PR 标题不符合仓库要求。CI 日志里提示 PR title 必须包含这些前缀之一：

```text
[BugFix], [Performance], [Test], [CI], [Feature],
[Doc], [Misc], [Community], [Refactor]
```

当前标题是：

```text
Gemma4  E2B and E4B
```

建议改成：

```text
[Feature] Support Gemma4 E2B and E4B
```

改完标题后重新触发 CI 即可继续往下跑。

---

## 10. 需要注意的边界

这里要避免在评审时把没验证的场景说满。

### 10.1 Context Parallel 暂不声明

当前 PR 改的是 `attention_v1.py` 里的普通 attention backend。

`vllm_ascend/attention/context_parallel/attention_cp.py` 有独立的 attention backend 和 cache 逻辑。

如果后续要声明 E2B/E4B 支持 context parallel，需要单独验证 CP 路径是否也会触发 target layer 写 cache。如果存在同类问题，再在 CP 路径补相同语义。

### 10.2 E4B MTP/spec decode 需要单独验证

Gemma4 MTP assistant attention 是 Q-only attention，它也依赖 `kv_sharing_target_layer_name` 读取 target model 的 cache。

从语义上说，本 PR 的改法是正确的：target layer 不写共享 cache。

但 MTP/spec decode 还涉及更多路径，比如 speculative config、graph capture、PA/FIA 选择等，所以建议单独跑 eager 和 graph 验证，不要只凭这个 PR 声明 MTP 全路径都已经覆盖。

### 10.3 多模态完整链路需要单独验证

这个 PR 处理的是 Gemma4 text stack 的 attention/KV cache 语义。

如果要声明完整多模态支持，还需要验证图像、音频、视频 encoder 以及融合后的推理链路。

---

## 11. 建议验证方式

最小验证建议跑：

1. E2B text generation eager。
2. E2B text generation ACLGraph / `FULL_DECODE_ONLY`。
3. E4B text generation eager。
4. E4B text generation ACLGraph / `FULL_DECODE_ONLY`。

如果要扩大支持范围，再补：

1. C8/NZ 或 W8A8 量化启动和短输出。
2. `kv_sharing_fast_prefill=False` 和 `True` 对比。
3. E4B MTP/spec decode eager。
4. E4B MTP/spec decode graph。
5. Context parallel 场景。

验证时重点看三点：

1. KV-sharing target layer 不应触发 cache write。
2. 普通 layer 仍应正常触发 cache write。
3. eager 和 graph 下输出应一致或符合预期波动。

---

## 12. 最后总结

这次 PR 的核心可以总结成一句话：

> Gemma4 E2B/E4B 的 KV-sharing target layer 只应该读 producer layer 的 cache，不应该再写 cache。

当前主干已经在 worker 层跳过了 target layer 的独立 KV cache 分配，但 attention backend 原来还会继续写 cache，导致共享 cache 有被覆盖的风险。

本 PR 在普通 cache 路径和 C8/NZ cache 路径都加了同样的保护：

- target layer 跳过 cache write。
- cache 引用仍然保留。
- attention 计算不被跳过。
- 普通模型和普通 layer 不受影响。

所以这是一个比较收敛的修复，改动点少、影响面小，并且符合 vLLM 原生 KV sharing 语义。
