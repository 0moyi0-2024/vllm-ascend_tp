# Gemma4 ModelSlim 量化适配通俗版讲稿

这版只讲两件事：

1. 原始报错是什么，为什么会报错。
2. 这个 PR 的修改逻辑是什么，为什么这样改。

---

## 1. 这个 PR 一句话在做什么

这个 PR 是为了让 Gemma4 的 ModelSlim W8A8_DYNAMIC 量化模型能在 vLLM-Ascend 上正常加载。

它解决了两个加载阶段的 KeyError：

- 31B dense 模型：attention 的 `v_proj.weight` 找不到。
- 26B MoE 模型：MoE expert 的 `moe.experts.weight` 找不到。

这两个错误都不是推理阶段算错，也不是算子本身的问题，而是在模型初始化时，vLLM-Ascend 根据 ModelSlim 的 `quant_model_description.json` 给每个 layer 选择量化方法时，key 对不上。

---

## 2. 原始报错一：31B dense 找不到 `v_proj.weight`

原始报错核心是：

```text
KeyError: 'language_model.model.layers.5.self_attn.v_proj.weight'
```

报错位置大概在：

```python
is_shard_skipped = self.quant_description[shard_prefix + ".weight"] == "FLOAT"
```

也就是说，代码想从 `quant_description` 里查：

```text
language_model.model.layers.5.self_attn.v_proj.weight
```

但是这个 key 不存在，所以直接 `KeyError`。

### 为什么这个 key 不存在

Gemma4 有一种 attention 层叫 full attention，里面有一个特殊配置：

```text
attention_k_eq_v = True
```

它的意思可以简单理解成：

> 这个层里 V 不单独存一份权重，V 直接复用 K。

所以 checkpoint 里这一层只有：

```text
q_proj.weight
k_proj.weight
```

没有：

```text
v_proj.weight
```

这不是量化文件坏了，而是 Gemma4 这个模型本来就是这样存的。

但是 vLLM 里这个层仍然用统一的 `QKVParallelLinear` 表示，也就是逻辑上会把它看成一个融合的：

```text
qkv_proj
```

而 `qkv_proj` 会被拆成三个 shard：

```text
q_proj
k_proj
v_proj
```

于是问题就来了：

- 模型结构上：`qkv_proj` 看起来有 q/k/v 三段。
- 量化描述里：k_eq_v 层只有 q/k，没有 v。
- 旧代码：不管什么情况都直接查 q/k/v 三个 key。
- 结果：查到 v 的时候报 `KeyError`。

### 修改逻辑

这里不能简单写成“缺什么都跳过”，因为如果普通模型真的缺了 `q_proj` 或 `k_proj`，那就是量化文件有问题，应该报错。

所以 PR 加了一个非常窄的判断：

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

通俗说就是：

> 只有当缺的是 `v_proj.weight`，并且同一层的 `q_proj.weight` 和 `k_proj.weight` 都存在时，才认为这是 Gemma4 k_eq_v 的正常情况，可以跳过。

其它情况继续按原逻辑报错。

### 改了哪些地方

改了两个地方，因为 ModelSlim config 会在两个步骤里遍历 packed shards。

第一个地方是判断量化类型：

```python
get_linear_quant_type(...)
```

它需要知道这个 layer 是 `W8A8_DYNAMIC` 还是 `FLOAT`。

第二个地方是判断这个 layer 是否跳过量化：

```python
is_layer_skipped_ascend(...)
```

它需要知道这个 layer 是不是全部 FLOAT。

这两个地方都加了同样的逻辑：

```python
if shard_key not in quant_description and _is_missing_k_eq_v_shard(...):
    continue
```

然后后面仍然保留原来的字典下标访问：

```python
quant_description[shard_key]
```

这点很重要，因为它保证了：

> 只有 Gemma4 k_eq_v 缺 `v_proj` 会被允许；其它缺 key 的情况还是会报错。

---

## 3. 原始报错二：26B MoE 找不到 `moe.experts.weight`

26B MoE 模型的原始报错核心是：

```text
KeyError: 'language_model.model.layers.0.moe.experts.weight'
```

这个报错说明代码在查：

```text
language_model.model.layers.0.moe.experts.weight
```

但是实际量化描述里没有这个 key。

### 为什么会查到这个不存在的 key

MoE expert 权重不是一个简单的 `experts.weight`。

它实际是多个 expert 的多个 projection，例如：

```text
language_model.model.layers.0.experts.0.gate_proj.weight
language_model.model.layers.0.experts.0.up_proj.weight
language_model.model.layers.0.experts.0.down_proj.weight
```

也就是说，它应该被当成 packed module 处理：

```text
experts
  -> experts.0.gate_proj
  -> experts.0.up_proj
  -> experts.0.down_proj
```

但是旧代码里没有给 `gemma4` / `gemma4_text` 注册这个 mapping。

所以代码看到 prefix：

```text
language_model.model.layers.0.moe.experts
```

时不知道它是 fused MoE experts，就把它当普通 linear 处理，直接查：

```text
language_model.model.layers.0.moe.experts.weight
```

这个 key 当然不存在，于是报错。

### 还有一个 prefix 不一致的问题

即使加了 experts mapping，也还有一层问题。

vLLM 的 Gemma4 模型路径里有：

```text
.moe.experts
```

但是 ModelSlim 的量化描述里是：

```text
.experts
```

简单说：

```text
模型里叫：...layers.0.moe.experts...
量化文件叫：...layers.0.experts...
```

中间多了一个 `.moe`。

所以如果不处理这个差异，代码会去查：

```text
language_model.model.layers.0.moe.experts.0.gate_proj.weight
```

但实际存在的是：

```text
language_model.model.layers.0.experts.0.gate_proj.weight
```

还是对不上。

---

## 4. MoE 的修改逻辑

MoE 部分主要做了两件事。

### 修改一：给 Gemma4 注册 packed mapping

新增了：

```python
"gemma4": {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
},
"gemma4_text": {
    ...
},
```

这告诉 ModelSlim config：

> 如果看到 Gemma4 的 `experts`，不要去查 `experts.weight`，而是展开查 `experts.0.gate_proj.weight`、`experts.0.up_proj.weight`、`experts.0.down_proj.weight`。

这样就解决了第一个 MoE 报错：

```text
KeyError: ...moe.experts.weight
```

为什么同时加 `gemma4` 和 `gemma4_text`？

因为不同加载路径下，model_type 可能是 `gemma4`，也可能是 `gemma4_text`。两个都加，避免只修了一种路径。

### 修改二：只给 Gemma4 做 `.moe.experts` 到 `.experts` 的映射

新增了：

```python
"gemma4": {
    ".moe.experts": ".experts",
},
"gemma4_text": {
    ".moe.experts": ".experts",
},
```

它的作用是，在查 quant key 之前，把模型 prefix：

```text
language_model.model.layers.0.moe.experts
```

改成：

```text
language_model.model.layers.0.experts
```

这样展开后就能查到实际存在的 key：

```text
language_model.model.layers.0.experts.0.gate_proj.weight
language_model.model.layers.0.experts.0.up_proj.weight
language_model.model.layers.0.experts.0.down_proj.weight
```

---

## 5. 为什么不做全局改写

一个看起来更简单的做法是：

> 直接给所有 `.experts.` key 都额外加一份 `.moe.experts.` alias。

比如：

```text
model.layers.0.experts.0.gate_proj.weight
```

自动复制成：

```text
model.layers.0.moe.experts.0.gate_proj.weight
```

但这个方案影响面太大。

因为很多 MoE 模型都有 `.experts.`，如果全局这么改，就会改变所有 MoE 模型的 quant_description key 集合。虽然可能不一定马上出错，但它不是 Gemma4 专属修复，review 时也很难证明不会影响别人。

所以最终方案没有做全局 alias，而是只在：

```python
gemma4
gemma4_text
```

这两个 model_type 下做 prefix 映射。

这样更稳，也更容易上库。

---

## 6. 单测保证了什么

单测主要证明几个边界。

第一，k_eq_v 场景下：

```text
q_proj 存在
k_proj 存在
v_proj 缺失
```

可以正常判断成 `W8A8_DYNAMIC`。

第二，如果缺的是必要 shard，比如 `q_proj`，还是会报 `KeyError`。

第三，Gemma4 / Gemma4_text 的：

```text
.moe.experts
```

会映射到：

```text
.experts
```

并能查到 gate/up/down 三个 expert shard。

第四，非 Gemma4 模型不会被这个映射影响。

第五，没有全局生成 `.moe.experts` alias，避免影响其它 MoE 模型。

---

## 7. 最后可以这样总结

这个 PR 本质上不是改计算逻辑，也不是改权重加载逻辑，而是补齐 ModelSlim quant config 对 Gemma4 特殊 key 形态的理解。

31B dense 的问题是：

> k_eq_v 层没有 `v_proj`，但 packed qkv lookup 以前强制查 v。

修复是：

> 只在 q/k 都存在且缺的是 v 时跳过 v。

26B MoE 的问题是：

> Gemma4 experts 没有注册 packed mapping，并且模型 prefix 多了 `.moe`，和 quant description 对不上。

修复是：

> 给 Gemma4 / Gemma4_text 注册 experts packed mapping，并且只对 Gemma4 做 `.moe.experts -> .experts` 的 prefix 映射。

这个方案的核心原则是：

> Gemma4 的特殊情况特殊处理，其它模型保持原逻辑。

