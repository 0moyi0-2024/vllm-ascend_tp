# Gemma4 A2 Nightly MoE 精度回归分析

## 1. 问题背景

2026 年 7 月 21 日的 A2 nightly 测试中，`google/gemma-4-26B-A4B-it`
在 GPQA-Diamond 数据集上出现明显精度回归。模型可以正常启动，198 条请求也全部
执行成功，但准确率从约 72 分下降到 46.97 分。

本文记录本次问题的测试证据、回归范围、实际执行路径、高置信度根因，以及建议的
验证和修复方法。

## 2. 测试配置

前后两组测试使用相同的主要配置：

```text
模型：google/gemma-4-26B-A4B-it
设备：Atlas A2
Tensor Parallel：4
Expert Parallel：开启
图模式：FULL_DECODE_ONLY
图捕获规格：[1, 2, 4, 8]
Max Model Length：32768
Max Batched Tokens：8192
Max Sequences：8
数据集：GPQA-Diamond，共 198 题
Batch Size：8
Temperature：1.0
Top-p：0.95
Top-k：64
Baseline：72
Threshold：5
```

失败测试没有 API 请求失败、服务启动失败或数据集加载失败。因此，本问题属于推理
精度回归，而不是环境、超时或数据集问题。

## 3. 测试证据

### 3.1 回归前结果

7 月 20 日的 nightly 使用 vLLM Ascend 提交
`95ef5af6474c974666eba8cc2631a26418f68a47`。Gemma4 MoE 连续两次通过：

| Workflow | Gemma4 Job | GPQA-Diamond 准确率 |
| --- | --- | ---: |
| [29756579967](https://github.com/vllm-project/vllm-ascend/actions/runs/29756579967) | [88405257729](https://github.com/vllm-project/vllm-ascend/actions/runs/29756579967/job/88405257729) | 70.71 |
| [29772234438](https://github.com/vllm-project/vllm-ascend/actions/runs/29772234438) | [88455346795](https://github.com/vllm-project/vllm-ascend/actions/runs/29772234438/job/88455346795) | 72.22 |

两次结果的差值只有 1.51 分，说明正常评测波动远小于后续约 24 分的下降。

### 3.2 回归后结果

7 月 21 日失败的 nightly 镜像中，日志记录的 vLLM Ascend 版本为
`0ac41db46bc73e0338bce2052d5a22d441b36d9a`：

| Workflow | Gemma4 Job | GPQA-Diamond 准确率 |
| --- | --- | ---: |
| [29845430434](https://github.com/vllm-project/vllm-ascend/actions/runs/29845430434) | [88688691751](https://github.com/vllm-project/vllm-ascend/actions/runs/29845430434/job/88688691751) | 46.97 |

Workflow head 是 `b92b3a65d9b1693349625ec2eeeb99246688a6c3`，运行时日志中的
`VLLM_ASCEND_VERSION` 是 `0ac41db`。两者之间的 `b92b3a65` 只修改 CI 配置，
因此分析运行时代码时以 `0ac41db` 为边界。

## 4. 回归范围

问题由 `95ef5af..0ac41db` 之间的提交引入。该范围内大部分提交只修改文档或 CI，
与 Gemma4 推理相关的主要运行时变更有两个：

1. [PR #12305](https://github.com/vllm-project/vllm-ascend/pull/12305)：
   将自定义 MoE init-routing 算子替换为
   `torch_npu.npu_moe_init_routing_v2`。
2. [PR #11266](https://github.com/vllm-project/vllm-ascend/pull/11266)：
   修改 FIA decode 阶段的 attention mask 处理。

结合实际命中的 MoE 通信路径和下文的算子契约不匹配，PR #12305 是本次回归的
高置信度首要来源。

## 5. 为什么命中 A2 ALLGATHER 路径

Gemma4-26B-A4B 包含 128 个路由专家。在 TP4、EP4 配置下，每张卡负责 32 个
专家。

A2 当前只有同时满足以下条件时才选择 MC2：

```text
每卡专家数 <= 24
EP world size >= 16
```

当前配置每卡 32 个专家、EP world size 为 4，因此不会走 MC2，而是选择
ALLGATHER。对应调用链为：

```text
Gemma4 MoE layer
  -> AllGatherCommImpl
  -> TokenDispatcherWithAllGather.token_dispatch
  -> DeviceOperator.npu_moe_init_routing
  -> 专家计算
  -> TokenDispatcherWithAllGather.token_combine
  -> DeviceOperator.npu_moe_token_unpermute
```

Dense 模型不会进入该路径。使用 MC2 或 ALLTOALL 的 MoE 场景也不会使用完全相同的
init-routing 和 unpermute 组合。

## 6. 高置信度根因

### 6.1 PR #12305 前的算子组合

修改前，A2/A3 使用以下配套实现：

```python
# Dispatch
torch.ops._C_ascend.npu_moe_init_routing_custom(...)

# Combine
torch_npu.npu_moe_token_unpermute(
    permuted_tokens=permuted_tokens,
    sorted_indices=torch.abs(sorted_indices),
    probs=probs,
)
```

A5 已经使用官方 routing 算子，并且在 unpermute 时明确不做 `abs`：

```python
# Dispatch
torch_npu.npu_moe_init_routing_v2(...)

# Combine
torch_npu.npu_moe_token_unpermute(
    permuted_tokens=permuted_tokens,
    sorted_indices=sorted_indices,
    probs=probs,
)
```

这个设备差异说明，自定义算子和官方算子输出的 `expanded_row_idx` 并不适合共用
同一种后处理语义。

### 6.2 PR #12305 后出现不匹配组合

PR #12305 将 BaseDeviceAdaptor 的 dispatch 算子替换成官方算子，但保留了原来
BaseDeviceAdaptor 中的 `abs` 处理，形成了新的组合：

```text
官方 npu_moe_init_routing_v2
              +
torch.abs(sorted_indices)
```

这个组合既不是原来的 A2/A3 实现，也不是已经使用官方算子的 A5 实现。

开启 Expert Parallel 后，`active_expert_range` 会筛除不属于当前 rank 的专家。
routing 算子需要在 `expanded_row_idx` 中表示这些条目。如果对官方算子产生的负值
或哨兵值继续执行 `abs`，就可能改变这些值的语义，使无效路由在 unpermute 阶段被
当成有效行，从而将错误专家结果混入最终输出。

这种错误不会必然触发异常或 NaN，因此与现象一致：

- 服务正常启动；
- 198 条请求全部完成；
- 问题集中在 MoE ALLGATHER 路径；
- 输出可以生成，但模型精度严重下降；
- Dense 模型不经过该 routing 路径。

仍建议通过 A/B 测试或单算子数值对比确认设备侧哨兵值的具体语义。但从回归区间、
实际执行路径和不匹配的算子契约来看，PR #12305 是高置信度根因。

## 7. 为什么原有测试没有发现

PR #12305 的单元测试通过 mock `npu_moe_init_routing_v2` 验证函数是否被调用。
这种测试只能验证 Python API 接线，不能验证以下内容：

- 路由后的 hidden states 是否正确；
- `expanded_row_idx` 的实际取值；
- `active_expert_range` 下非本卡专家的处理；
- init-routing 与 unpermute 组合后的最终数值；
- EP 模型在图模式下的端到端精度。

此前 PR CI 也没有运行完整的 GPQA-Diamond 精度评测。Nightly 是首次同时覆盖受影响
ALLGATHER 路径和端到端模型精度的测试。

## 8. 最小验证和修复建议

保留 PR #12305 引入的官方 `npu_moe_init_routing_v2`，但将
BaseDeviceAdaptor 的 unpermute 行为与官方算子的契约对齐：

```python
@staticmethod
def npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs):
    return torch_npu.npu_moe_token_unpermute(
        permuted_tokens=permuted_tokens,
        sorted_indices=sorted_indices,
        probs=probs,
    )
```

修改后使用完全相同的 Gemma4 A2 nightly 配置验证。验收条件如下：

1. 服务正常启动并完成全部 198 条 GPQA-Diamond 请求。
2. 精度恢复到此前约 70-72 分的范围，并满足 baseline 和 threshold。
3. 现有 A2 MoE ALLGATHER 用例不出现 NaN。
4. 增加单算子数值测试：当 `active_expert_range` 排除部分专家时，dispatch 后再
   unpermute 的输出应与 PyTorch 参考实现一致。

如果去掉 `abs` 后精度仍未恢复，再单独回退 PR #11266 的 FIA decode mask 修改，
作为第二阶段 A/B 验证。PR #11266 仍位于回归区间内，但它不能解释当前已经存在的
MoE ALLGATHER 算子契约不匹配。

## 9. 长期测试建议

建议增加 A2 ALLGATHER 完整数值闭环测试，而不是只 mock API 调用。测试至少覆盖：

- 多 token、多 top-k 专家；
- 同时包含本卡专家和 `active_expert_range` 外的专家；
- routing 与 unpermute 后的结果对比 PyTorch 参考实现；
- `expanded_row_idx` 的特殊值处理；
- 至少一个 Gemma4 MoE 图模式冒烟用例。

这样可以直接保护 init-routing 与 unpermute 之间的算子契约，避免单独验证某一个
算子调用时漏掉组合路径上的精度问题。
