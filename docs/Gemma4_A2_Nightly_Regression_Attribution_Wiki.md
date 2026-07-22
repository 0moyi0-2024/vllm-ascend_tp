# Gemma4 A2 Nightly 精度回归责任边界与评审沟通说明

## 1. 文档目的

本文用于向 maintainer 说明 Gemma4 A2 nightly MoE 精度回归的技术
归因和责任边界。沟通重点是明确“哪个变更引入了什么不一致”，
而不是针对提交者个人。

详细技术证据见同分支的
`Gemma4_A2_Nightly_MoE_Accuracy_Regression_Wiki.md`。

## 2. 建议的正式结论

本次精度回归的高置信度直接来源是
[PR #12305](https://github.com/vllm-project/vllm-ascend/pull/12305) 的 MoE routing
算子迁移不完整。

PR #12305 将 BaseDeviceAdaptor dispatch 侧的
`npu_moe_init_routing_custom` 替换为官方
`torch_npu.npu_moe_init_routing_v2`，但没有同步清理 combine 侧与旧
custom 算子绑定的 `torch.abs(sorted_indices)` 处理。这导致合入后出现
一个之前不存在的算子组合：

```text
npu_moe_init_routing_v2 + torch.abs(sorted_indices)
```

该组合将官方 init-routing 算子的输出按旧 custom 算子的约定处理，
造成 producer 和 consumer 的索引语义不匹配。

## 3. 完整评审说辞

可以在评审会或 PR 讨论中使用以下表述：

> 本次精度回归是由 PR #12305 的 MoE routing 算子替换引入的。该 PR
> 将 BaseDeviceAdaptor 中的 `npu_moe_init_routing_custom` 替换为官方
> `npu_moe_init_routing_v2`，但只替换了 dispatch 侧，没有同步调整与其配套的
> unpermute 处理。
>
> 旧 custom routing 路径在 A2/A3 combine 前对 `expanded_row_idx` 执行
> `torch.abs`；官方 `npu_moe_init_routing_v2` 路径则应原样传递索引，A5
> 已有实现也使用这一方式。#12305 合入后形成了“官方 init-routing
> 算子 + 旧 custom 算子的 `abs` 后处理”这个不匹配组合。
>
> 在 Expert Parallel 的 `active_expert_range` 场景下，`abs` 可能改变官方
> routing 算子产生的特殊索引语义，导致无效路由在 unpermute 阶段被
> 错误恢复，最终污染 MoE 输出。
>
> Nightly 数据与合入时间一致：#12305 合入前，同一配置连续得到
> 70.71 和 72.22；合入后的完整 nightly 下降到 46.97。服务和全部
> 198 条请求都正常完成，因此这不是环境或稳定性问题，而是数值路径
> 回归。
>
> #12305 的单元测试主要通过 mock 验证新算子被调用，没有覆盖
> init-routing 与 unpermute 的数值闭环，因此没有发现该契约不匹配。
> PR #12530 不回退官方算子，只移除已经失效的 custom-op 后处理，
> 是对 #12305 不完整迁移的最小补全。

## 4. 简短版本

适合用于 PR comment 或会议中快速总结：

> #12305 完成了 routing producer 的替换，但遗漏了 consumer 侧与旧
> custom 算子绑定的 `abs` 后处理，导致官方 routing 算子与旧
> unpermute 语义混用。PR #12530 是对该不完整迁移的最小修复。

## 5. 证据链

### 5.1 时间和精度证据

| 状态 | vLLM Ascend 提交 | GPQA-Diamond 精度 |
| --- | --- | ---: |
| #12305 合入前 | `95ef5af` | 70.71 |
| #12305 合入前 | `95ef5af` | 72.22 |
| #12305 合入后 | `0ac41db` | 46.97 |

失败任务：
[29845430434 / 88688691751](https://github.com/vllm-project/vllm-ascend/actions/runs/29845430434/job/88688691751)。

### 5.2 路径证据

Gemma4-26B-A4B 共有 128 个路由专家。TP4/EP4 下每卡 32 个专家，
不满足 A2 MC2 要求的“每卡专家数不超过 24 且 EP 不小于 16”，
因此实际走 ALLGATHER。

PR #12305 修改的 BaseDeviceAdaptor `npu_moe_init_routing` 正是该 ALLGATHER
路径的 dispatch 入口，不是一个无关代码变更。

### 5.3 代码证据

PR #12305 之前：

```text
A2/A3: custom init-routing + abs(sorted_indices)
A5:    official init-routing v2 + unchanged sorted_indices
```

PR #12305 之后：

```text
A2/A3: official init-routing v2 + abs(sorted_indices)
A5:    official init-routing v2 + unchanged sorted_indices
```

只有 A2/A3 被改成了之前不存在的不匹配组合。

## 6. 可能的追问与回答

### 6.1 为什么不认为是评测随机波动？

回归前同一提交的两次结果是 70.71 和 72.22，波动只有 1.51 分。
回归后下降至 46.97，差距远大于正常波动。

### 6.2 为什么 Dense 没有这个问题？

Dense 模型不经过 MoE init-routing、专家 dispatch 和 unpermute 路径，因此不会
命中 #12305 改动的关键逻辑。

### 6.3 为什么其他 MoE 模型不一定马上失败？

其他模型可能选择 MC2 或 ALLTOALL，也可能使用不同的 EP 规模、专家数和
top-k。只有命中 BaseDeviceAdaptor ALLGATHER 且产生特殊路由索引的场景才会
直接暴露该问题。

### 6.4 为什么 PR #12305 的 CI 没有拦截？

该 PR 的单测将 NPU 算子 mock 掉，主要检查接口调用，没有检查真实
`expanded_row_idx` 和 unpermute 后的数值。这是测试覆盖的缺口，不能证明
两个算子的组合契约正确。

### 6.5 是否已经百分之百定位？

更严谨的说法是：回归区间、实际执行路径和算子契约三方面证据都
指向 #12305，属于高置信度直接来源。最终由 PR #12530 修复后的 A2
nightly A/B 结果完成闭环确认。

## 7. 建议避免的说法

不建议使用以下表述：

- “#12305 的作者导致了所有 MoE 精度问题。”
- “已经百分之百确定，不需要再做验证。”
- “只要回退 #12305 就可以了。”

原因是这些说法把技术责任扩大到个人或所有 MoE 场景，也忽略了必要的
硬件 A/B 闭环。建议始终将责任定义为：

> PR #12305 的算子迁移遗漏了 consumer 侧契约调整，引入了可复现的
> A2 Gemma4 MoE ALLGATHER 精度回归。

## 8. 当前修复

最小修复已提交至
[PR #12530](https://github.com/vllm-project/vllm-ascend/pull/12530)。该 PR：

- 保留官方 `npu_moe_init_routing_v2`；
- 只移除 BaseDeviceAdaptor 中的旧 `torch.abs` 变换；
- 增加负索引原样传递的单元测试；
- 不修改 attention、MoE 通信选择或 A5 路径。

该方案不需要回退 #12305 删除的大量 custom-op 代码，属于补齐官方算子
迁移契约的最小修复。
