# A5 Gemma4 Maintainer 重构意见和方案

## 背景

这次根据 maintainer 意见的重构，核心目标是：保留 Gemma4 A5 图模式必要修复，同时降低改动复杂度、缩小影响面，避免让其它模型和 speculative/draft 图路径承担额外风险。

## 改动思路

### 1. 精简 workspace 逻辑

原方案里工具函数偏多，维护成本和解释成本较高。现在收敛成一个核心函数 `cache_graph_workspace`：

- 普通模型保持原有行为：同一个 `num_tokens` 只缓存第一次 workspace，不重复动态计算和替换。
- Gemma4 单独走 max workspace 逻辑：同一个 graph bucket 内可能混有不同 attention layer shape，所以保留该 bucket 下见过的最大 workspace，避免 FIA replay 时 workspace 不足。

### 2. 缩小 Gemma4 特殊逻辑影响面

`_use_max_workspace_for_fia_graph` 只在检测到 `gemma4/gemma4_text` 时开启。

这样 workspace 最大值复用只对 Gemma4 生效，其它模型不会被引入额外 workspace 查询和替换开销。

### 3. 清理 `attention_v1.py` 里的辅助函数

按 maintainer 建议，把能内联的逻辑内联，避免 `attention_v1.py` 里堆太多小函数。

`workspace` 和 `layer_name` 的处理也拆开表达，不再用一个混合 helper 返回多个语义不同的值，提升可读性。

### 4. 保留必要的 layer metadata 绑定

Gemma4 的精度问题来自 graph replay 阶段不能只依赖 `attn_metadata` 字典顺序。

所以保留 `_layer_name` / `kv_sharing_target_layer_name` 相关逻辑，在 capture 时记录真实 attention layer name，update 时按 layer name 找对应 metadata。

这样可以避免 global layer 拿到 sliding-window layer metadata。

### 5. 恢复 draft/speculative graph 路径的旧行为

后续 CI 暴露出 V2 EAGLE spec decode 会走 draft graph capture/update。

Gemma4 修复需要的是 target/non-draft layer metadata 绑定，不需要影响 draft model。因此最新调整是：

- non-draft graph params 继续带 `layer_name`
- draft graph params 保持旧 tuple 结构，不额外携带 `layer_name`

这样避免影响 EAGLE V2 图捕获路径，也减少其它 speculative 方法的风险。

### 6. `experts_selector.py` 保持兼容

对 custom routing function 的 `num_experts` 兼容逻辑保留，避免 Gemma4 routing function 不接受 `num_experts` 时启动报错，同时不破坏其它模型原先带 `num_experts` 的调用路径。

## 最终方案

- Gemma4 图模式需要的两个关键修复仍保留：workspace 按最大值缓存复用；attention replay 按 layer name 绑定 metadata。
- 普通模型尽量维持旧行为：不开启 max workspace；draft/speculative graph 参数结构不变。
- 代码结构上收敛到少量必要逻辑：`utils.py` 里保留 `cache_graph_workspace`，其它能内联的处理放回调用点，避免 helper 过度抽象。

## 给 maintainer 的简短说明

这次重构主要是把 Gemma4 A5 图模式修复做成最小影响面：workspace 最大值缓存只对 Gemma4 开启，其它模型保持首次缓存行为；layer-name metadata 绑定只用于 target graph replay，draft/speculative graph 路径保持原有参数结构。

这样既解决 Gemma4 混合 attention layer 在同一 graph bucket 下 workspace 和 metadata 错配的问题，也避免对其它模型、EAGLE spec decode 和现有图模式路径引入额外行为变化。
