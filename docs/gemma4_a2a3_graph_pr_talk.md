# Gemma4 A2/A3 Graph Execution PR 讲稿

## 1. 开场：这个 PR 要解决什么

各位好，我这次 PR 主要是为了让 Gemma4 在 Ascend A2/A3 上可以正常走 graph execution，重点解决两个问题：

1. **RoPE Triton kernel 编译期 UB 溢出**。
   这个问题发生在 profile run 编译阶段，模型还没有真正开始推理，Triton RoPE kernel 编译就失败了。

2. **FULL_DECODE_ONLY graph 模式下 attention replay 参数格式不匹配**。
   这个问题发生在 graph capture 成功之后，真实 decode 推理时更新 graph params 失败。

这两个问题都不是量化问题，也不是环境配置能绕过去的问题，核心都是 Gemma4 这个模型结构比较特殊：

- sliding attention 层使用 `head_dim=256`
- full/global attention 层使用 `head_dim=512`
- A2/A3 上 512 global attention decode 需要回退到 PagedAttention
- 其它 attention 层仍然可以走 FIA

所以它同时触发了 **large-head RoPE 编译压力** 和 **同一个 decode graph 中 PA/FIA 参数混合** 这两个问题。

---

## 2. 原始报错一：RoPE UB overflow

第一个报错发生在 profile run 编译阶段，原始报错核心是：

```text
loc("vllm_ascend/ops/triton/rope.py":24:0): error: ub overflow,
requires 1581056 bits while 1572864 bits available!

subprocess.CalledProcessError:
bishengir-compile ... --target=Ascend910B3 ...
returned non-zero exit status 1.
```

调用链大概是：

```text
profile_run
  -> model forward
    -> vllm_ascend/ops/rotary_embedding.py rope_forward_oot
      -> vllm_ascend/ops/triton/rope.py rope_forward_triton
        -> _triton_rope JIT compile
          -> bishengir-compile
            -> UB overflow
```

### 为什么会溢出

`rope.py` 里的 Triton RoPE kernel 是按 `BLOCK_SIZE_HEAD` 在 head 维度上切 tile 的，原来的逻辑是：

```python
if is_neox_style:
    BLOCK_SIZE_HEAD = 64
else:
    BLOCK_SIZE_HEAD = 32
```

Gemma4 的 RoPE 是 neox style，所以默认 `BLOCK_SIZE_HEAD=64`。

Gemma4 有两类 attention layer：

- sliding attention：`head_dim=256`，`rope_dim=256`
- full/global attention：`head_dim=512`，`rope_dim=128`

也就是说 RoPE kernel 的 tile 规模会明显比常见 `head_dim <= 128` 的模型更大。A2 上 UB 大小有限，原来的 64 head tile 对 Gemma4 这种 large-head RoPE 不够保守，最终在编译时超过 UB 上限：

```text
requires 1581056 bits
available 1572864 bits
```

差距只有 8192 bits，也就是 1024 bytes，但已经足够让编译失败。

### 为什么不能靠环境配置解决

这里我确认过几个方向：

- 不能简单关闭 Triton RoPE，因为 vLLM 里还有其它 Triton kernel 依赖 `HAS_TRITON`。
- 不能通过环境变量关闭 multi-buffer，因为这是 triton-ascend 编译选项内部控制的。
- 换 triton-ascend 版本也不是稳定方案，当前可用版本和项目依赖是固定的。

所以这个问题本质上是 **kernel tiling 对 large-head 场景不够保守**，需要在代码里调整 tile。

---

## 3. RoPE 修改点：`vllm_ascend/ops/triton/rope.py`

当前 PR 的修改是：

```python
_LARGE_HEAD_DIM_THRESHOLD = 256
_LARGE_HEAD_BLOCK_SIZE = 16

...

if head_dim >= _LARGE_HEAD_DIM_THRESHOLD:
    BLOCK_SIZE_HEAD = min(BLOCK_SIZE_HEAD, _LARGE_HEAD_BLOCK_SIZE)
```

### 为什么是 `head_dim >= 256`

这里没有写 `512`，原因是 Gemma4 不只有 512 head：

- global/full attention 是 `head_dim=512`
- sliding attention 是 `head_dim=256`

如果阈值写成 `512`，只能覆盖 global attention，但 sliding attention 的 `256` large-head RoPE 仍然会走旧的 `BLOCK_SIZE_HEAD=64`。从报错分析看，风险来自 RoPE tile 的规模，而不是只来自 512 这个值。

这里也没有继续用 `>128`，是为了减少影响面：

- `>128` 会覆盖 `192`
- `>=256` 只覆盖 256 及以上
- Gemma4 当前已知需要覆盖的是 256 和 512

所以 `>=256` 是一个更收敛的 large-head 判断，既覆盖 Gemma4，又尽量不碰常见 head size。

### 为什么 block size 改成 16

原来 neox style 是 64，改到 16 后，相当于 head 维度上每个 program 的 tile 明显变小，UB 使用量也会下降。

这个改法没有改变 RoPE 计算公式，也没有改变输入输出布局，只是把同一段计算拆成更小的 tile 来执行。

可以理解成：原来一次搬太多，A2/A3 的 UB 装不下；现在分小块搬，计算结果还是同一个结果。

### 这个修改会不会影响其它模型

严格来说，它对其它模型的影响分两类：

1. **常见模型不受影响**

   对 `head_dim < 256` 的模型，代码完全走原路径：

   ```python
   if head_dim >= 256:
       ...
   ```

   所以 `head_dim=64/80/96/128/192` 都不会被这个修改影响。

2. **`head_dim >= 256` 的其它模型会使用更小 tile**

   对这类模型，RoPE kernel 会用 `BLOCK_SIZE_HEAD=16`。这不会改变计算语义，不会改变精度逻辑，只是 tile 更小。

   潜在影响主要是性能：tile 变小后循环次数可能增加，RoPE kernel 可能稍慢。但这类 large-head RoPE 本身就是更容易遇到 UB 风险的场景，用更保守的 tile 是合理的。

所以我会这样给 maintainer 总结：

> 这个修改不改变 RoPE 数学计算，只改变 large-head 场景的 tiling。常见 head_dim 小于 256 的模型完全不受影响；head_dim 大于等于 256 的模型会使用更保守 tile，主要是为了避免 UB overflow，可能带来轻微性能影响，但不会改变功能和精度语义。

---

## 4. 原始报错二：graph replay attention 参数 unpack 不匹配

第二个问题发生在 graph capture 成功之后，真实 decode 时更新 graph params 报错。

第一阶段原始报错是：

```text
File "vllm_ascend/compilation/acl_graph.py", line 297, in update_full_graph_params
    impl_cls.update_graph_params(...)

File "vllm_ascend/attention/attention_v1.py", line 713, in update_graph_params
    (

ValueError: not enough values to unpack (expected 21, got 9)
RuntimeError: Worker failed with error 'not enough values to unpack (expected 21, got 9)'
```

后面我引入 `PagedAttentionGraphParam` 包装之后，发现还有一个漏补的分支，报错变成：

```text
TypeError: cannot unpack non-iterable PagedAttentionGraphParam object
RuntimeError: Worker failed with error 'cannot unpack non-iterable PagedAttentionGraphParam object'
```

这两个报错本质上是同一个问题：**graph replay 时 PA 和 FIA 的参数格式混在同一个列表里，但 update 阶段用单一路径去 unpack。**

---

## 5. 为什么 Gemma4 会出现 PA/FIA 混合

attention graph capture/update 当前有两种主要路径：

| 路径 | capture 来源 | 参数格式 | update 方式 |
|---|---|---|---|
| PA | `full_graph_pa` | 9 元组 | `_npu_paged_attention` |
| FIA | `full_graph_fia` | 21 元组 | `npu_fused_infer_attention_score` |

PA 参数大概是 9 项：

```text
query, key_cache, value_cache,
num_kv_heads, num_heads, scale,
block_table, seq_lens, output
```

FIA 参数大概是 21 项，除了基础参数，还包括：

```text
attn_mask, block_size, query_start_loc,
softmax_lse, sparse_mode, pre_tokens, next_tokens,
quant scale/offset, layer_name 等
```

普通模型通常一个 decode graph bucket 里参数格式比较一致，要么都是 PA，要么都是 FIA。

但 Gemma4 在 A2/A3 上比较特殊：

- full/global attention 的 `head_dim=512`
- A2/A3 FIA TND 暂不支持这个 512 global head decode
- 所以 global 层 decode 走 PA
- sliding 层还可以走 FIA

结果就是：同一个 `graph_params.attn_params[num_tokens]` 里，既有 PA 参数，也有 FIA 参数。

旧逻辑的问题在于，`update_graph_params` 外层先判断一次：

```python
if using_paged_attention(...):
    # 按 PA 9 元组 unpack 整个列表
elif _EXTRA_CTX.sinks:
    # 按 FIA-v2 参数 unpack
else:
    # 按 FIA 21 元组 unpack 整个列表
```

这个设计默认一个列表里的参数格式是同一种。但 Gemma4 的列表是混合的，所以必然会出现：

- 走 FIA 分支时，把 PA 的 9 元组当成 21 元组解包，报 `expected 21, got 9`
- 引入 wrapper 后，如果某个 FIA 分支没识别 wrapper，就会把 `PagedAttentionGraphParam` 当普通 21 元组解包，报 `cannot unpack non-iterable`

---

## 6. 修改点一：`vllm_ascend/attention/utils.py`

这里新增了两个东西：

### 6.1 `PagedAttentionGraphParam`

```python
@dataclass
class PagedAttentionGraphParam:
    """Mark PA params when PA and FIA share one graph replay list."""

    params: tuple
    layer_name: str | None
```

这个类的作用很简单：**明确标记这个 captured param 是 PA 参数。**

为什么不用 tuple 长度判断？

因为靠长度判断不够稳：

- 后续 PA/FIA 参数可能扩展
- sinks、draft、spec decode 等场景参数结构也可能变化
- `len(param) == 9` / `len(param) == 21` 这种写法可读性和可维护性都不好

所以这里用显式类型标记：

```python
isinstance(param, PagedAttentionGraphParam)
```

这比魔术数字更容易维护，也更适合上库。

### 6.2 `update_paged_attention_graph_param`

这个 helper 做的事情是：当 update 阶段发现当前 captured param 是 PA 参数时，用 PA 的方式更新 graph task。

它内部主要做三步：

1. 从 `param.params` 解出 PA capture 时保存的 query/key_cache/value_cache 等参数。
2. 使用当前 metadata 里的 `block_table` 和 `seq_lens` 重新计算 PA workspace。
3. 调 `_npu_paged_attention` 做 graph task update。

为什么 helper 放到 `utils.py`？

因为 `update_graph_params` 里有多个分支：

- PA 分支
- sinks/FIA-v2 分支
- 普通 FIA 分支

这三个分支都可能遇到 `PagedAttentionGraphParam`，如果把 PA update 逻辑展开写在每个分支里，`attention_v1.py` 会重复很多代码，也更容易漏分支。

这次之前的 `cannot unpack non-iterable PagedAttentionGraphParam` 就是因为漏了普通 FIA 分支。因此把 PA update 收到 helper 里，三个分支只做分发，可以减少重复，也降低后续维护风险。

---

## 7. 修改点二：`vllm_ascend/attention/attention_v1.py`

这个文件里主要有四类改动。

### 7.1 `full_graph_pa` capture 时包装 PA 参数

原来 PA capture 直接 append 一个 9 元 tuple：

```python
graph_params.attn_params[num_tokens].append(
    (
        query,
        key_cache,
        value_cache,
        ...
    )
)
```

现在改成：

```python
graph_params.attn_params[num_tokens].append(
    PagedAttentionGraphParam(
        (
            query,
            key_cache,
            value_cache,
            ...
        ),
        self._graph_metadata_layer_name() if self._use_layer_aware_fia_graph_replay else None,
    )
)
```

这一步的目的不是改变 PA 参数，而是给 PA 参数加一个类型外壳，让 update 阶段能识别它。

### 7.2 为什么要带 `layer_name`

Gemma4 有 sliding/global 混合层，不同层的 metadata 语义不完全一样。

如果 update 阶段只依赖 `attn_keys` 的迭代顺序，可能会出现 global 层拿到 sliding 层 metadata 的情况。之前 A5 图模式精度问题里已经遇到过类似问题。

所以 wrapper 里带 `layer_name`：

```python
layer_name: str | None
```

update 时优先用 captured layer name 找当前 metadata：

```python
metadata_key = layer_name if layer_name is not None and layer_name in attn_metadata else key
```

这样做的好处是：

- 对 Gemma4 这种 mixed layer 模型，replay metadata 更准确
- 对普通模型，如果没有 layer-aware replay，则 `layer_name=None`，回退旧的 key 逻辑

### 7.3 PA update 分支里 unwrap

PA 分支原来直接解 9 元 tuple。

现在多了：

```python
if isinstance(param, PagedAttentionGraphParam):
    param = param.params
```

这保证纯 PA 路径仍然可以正常解包。

这个改动对旧 tuple 也兼容：如果 param 不是 wrapper，就直接走原来的解包逻辑。

### 7.4 sinks/FIA-v2 和普通 FIA 分支里分发 PA wrapper

在 FIA-v2 和普通 FIA 分支循环开头都加了：

```python
if isinstance(param, PagedAttentionGraphParam):
    ...
    update_paged_attention_graph_param(...)
    continue
```

这就是解决 unpack 报错的核心。

现在同一个 graph params 列表里如果混合了 PA/FIA：

- PA 参数命中 `PagedAttentionGraphParam`，走 PA update
- FIA 参数不是 wrapper，继续走原 21 元 unpack 和 FIA update

也就是说，**不再由外层一个分支决定整个列表怎么解包，而是在循环里对每个 captured op 按真实类型分发。**

---

## 8. 为什么 attention 修改不会影响其它模型

这里可以分场景解释。

### 8.1 eager 模式不受影响

这些修改都在 graph capture/update 相关路径里。不开 graph 的 eager 模式不走这套 `update_graph_params` replay 逻辑。

### 8.2 纯 FIA 模型不受影响

纯 FIA 模型 capture 出来的还是普通 FIA tuple，不是 `PagedAttentionGraphParam`。

所以新增判断：

```python
if isinstance(param, PagedAttentionGraphParam):
    ...
```

不会命中，后面的 21 元 unpack 和 FIA update 完全保持原逻辑。

### 8.3 纯 PA 模型兼容

PA capture 现在会保存成 `PagedAttentionGraphParam`，但 PA update 分支开头会 unwrap：

```python
if isinstance(param, PagedAttentionGraphParam):
    param = param.params
```

unwrap 后调用的还是原来的 `_npu_paged_attention_get_workspace` 和 `_npu_paged_attention`，参数内容没有变。

所以纯 PA 模型只是多了一层轻量包装，不改变 PA 算子和计算。

### 8.4 PA/FIA 不混合的模型不改变路由

这个 PR 没有改 `forward_impl` 里 PA/FIA 的路由条件，也没有让其它模型强制走 PA 或 FIA。

它只是让 capture 后的 replay params 能按真实类型更新。

所以对没有混合 PA/FIA 的模型，路由策略不变；对 Gemma4 这种混合模型，才会发挥作用。

### 8.5 为什么不用 tuple 长度判断，也是不影响其它模型的原因之一

如果用 `len(param)` 判断，很可能误伤未来扩展的参数格式。

现在用 dataclass 显式标记，只处理我们自己包装的 PA 参数，不会把普通 tuple 误判成 PA。这是更安全的兼容方式。

---

## 9. 修改点三：`tests/ut/attention/a2/test_attention_v1.py`

单测里新增了一个覆盖：

```python
test_update_graph_params_handles_captured_paged_attention_params
```

这个测试模拟的是最关键的失败场景：

- 外层 `using_paged_attention=False`
- `_EXTRA_CTX.sinks=False`
- 所以 update 进入普通 FIA 分支
- 但 captured params 里放了一个 `PagedAttentionGraphParam`

旧逻辑会在普通 FIA 分支里直接 unpack，导致：

```text
cannot unpack non-iterable PagedAttentionGraphParam object
```

新逻辑应该识别 wrapper，调用 PA update helper，然后 `continue`，不进入 FIA 21 元 unpack。

测试里断言：

- `_npu_paged_attention_get_workspace` 被调用
- `_npu_paged_attention` 被调用
- 使用的是当前 metadata 里的 `seq_lens`
- graph task update begin/end 被调用

这个测试不是为了验证真实 NPU 数值，而是为了防止这次最核心的 replay 分发逻辑再回退。

---

## 10. 整体影响面总结

这次 PR 的影响可以分成 RoPE 和 attention 两部分。

### 10.1 RoPE 影响面

RoPE 修改生效条件是：

```python
head_dim >= 256
```

所以：

- `head_dim < 256`：完全不受影响
- `head_dim >= 256`：使用更小的 `BLOCK_SIZE_HEAD=16`

这个修改不改变 RoPE 公式，不改变输入输出，只改变 tile size。

对其它 large-head 模型来说，理论上功能和精度语义不变，可能有轻微性能影响，但也能降低 UB overflow 风险。

### 10.2 attention graph replay 影响面

attention 修改只在 graph params replay 阶段生效，而且只对 `PagedAttentionGraphParam` 生效。

- 普通 FIA tuple：不命中，原逻辑不变
- 普通 FIA-v2/sinks tuple：不命中，原逻辑不变
- PA wrapper：走 PA update helper
- eager 模式：不走这段逻辑

这说明它不是一个全局行为变更，而是针对 graph replay 参数混合场景的兼容处理。

---

## 11. Maintainer 可能会问的问题

### Q1：为什么 RoPE 阈值是 `>=256`，不是 `512`？

因为 Gemma4 的问题不只在 512 global layer。

Gemma4 sliding attention 是 `head_dim=256`，并且 `rope_dim=256`。这个路径同样属于 large-head RoPE，也会导致更大的 UB tile。

如果只写 512，会漏掉 256 sliding layer。

### Q2：为什么不是 `>128`？

`>128` 也能覆盖 Gemma4，但影响面更大，会覆盖 192。

目前 Gemma4 已知需要覆盖的是 256 和 512。为了减少对其它模型的影响，`>=256` 更收敛。

### Q3：为什么不判断 A2/A3 device type？

这里主要是为了保持 RoPE helper 简洁，不在 Triton op helper 里引入设备类型依赖。

同时 `head_dim >= 256` 本身已经把影响面限制到 large-head RoPE。对其它设备，使用更小 tile 不改变计算语义，最多是性能上更保守。

如果 maintainer 强烈要求进一步限制影响面，也可以把 device 判断加回来。但当前版本的优点是简单、通用、少依赖。

### Q4：为什么不用 `len(param) == 9` 判断 PA？

因为这种写法比较脆弱，也不利于后续扩展。

如果未来 PA 参数多加一个字段，或者其它 graph param 也刚好是 9 项，就可能误判。

显式的 `PagedAttentionGraphParam` 更清楚：只有 capture 时被我们标记为 PA 的参数，update 时才按 PA 处理。

### Q5：为什么要带 `layer_name`？

Gemma4 有 sliding/global 混合 attention，metadata 语义不同。

在 graph replay 时，如果只依赖 attn key 顺序，可能出现 global 层拿到 sliding 层 metadata 的问题。`layer_name` 可以让 update 阶段按 captured op 对应的真实 layer 找 metadata，避免 metadata 错配。

### Q6：这个 PR 会不会让别的模型也走 PA？

不会。

这个 PR 没有改 PA/FIA 的路由决策，只是给已经 capture 成 PA 的参数加标记，并在 update 时按 PA 更新。

也就是说，它不决定“走不走 PA”，只保证“已经 capture 成 PA 的 graph op 不要被 FIA 分支错误 unpack”。

---

## 12. 最后总结

这次 PR 的核心可以一句话概括：

> Gemma4 在 A2/A3 graph 模式下同时遇到 large-head RoPE UB overflow 和 mixed PA/FIA graph replay 参数不匹配；本 PR 通过收窄的 RoPE large-head tiling 和显式 PA graph param 标记，让 capture 和 replay 阶段按真实 op 类型更新参数，避免编译失败和 decode replay unpack 崩溃。

改动保持了几个原则：

- 不改变 eager 路径
- 不改变 attention 路由策略
- 不用 tuple 长度这种魔术判断
- 常见 `head_dim < 256` 模型 RoPE 不受影响
- 普通 FIA/PA 模型 replay 语义保持不变
- 只对 Gemma4 这类 mixed PA/FIA graph replay 场景补兼容

当前验证：

- Gemma4 W8A8 在 4 卡 Ascend 910B3、TP=4、`FULL_DECODE_ONLY` graph 模式下服务启动和简单推理通过
- 无 RoPE UB overflow
- 无 `expected 21, got 9`
- 无 `cannot unpack non-iterable PagedAttentionGraphParam`
- 本地静态检查通过：ruff format、ruff check、compileall

