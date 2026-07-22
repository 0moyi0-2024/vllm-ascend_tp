# Gemma4 26B (MoE) 重复退化（repetition loop）分析报告

- 日期：2026-07-22
- 基线代码：`vllm-ascend` upstream `main` @ `44fc51ffb` + cherry-pick PR [#12530](https://github.com/vllm-project/vllm-ascend/pull/12530)（`[Bugfix][MoE] Preserve routing indices during unpermute`，本地 commit `3d4bc8171`）
- 硬件：Ascend 910B3 ×4（devices 4,5,6,7），CANN 9.0.0
- vllm：源码构建自 `.github/vllm-main-verified.commit` = `54503ecec0`（`VLLM_TARGET_DEVICE=empty`，保留 torch 2.10.0 + torch_npu 2.10.0）
- 评测框架：evalscope 1.8.0，数据集 GPQA Diamond（modelscope），0-shot

## 1. 结论（TL;DR）

Gemma4 26B（MoE，A4B）在 `thinking 关闭 + temp 0.95 + 无 repetition_penalty` 的生成配置下，对 GPQA Diamond 有 **30.8%（61/198）的题陷入退化性重复 loop**——模型卡在一个短句上无限重复直至 `max_tokens` 上限，无法给出答案字母，全部判错。这把全量准确率压到 **0.5859**。

作为对照，**31B dense 在完全相同的生成配置下零重复 loop**（0/198），全量准确率 **0.7677**。差异指向 26B MoE 特有的问题。

## 2. 现象

### 2.1 触顶样本是重复 loop，不是长推理

对 61 个触达 `max_tokens` 上限的样本检查其输出尾部，全部是退化性重复，例如：

| 样本 | 重复片段 | 重复次数 |
|---|---|---|
| #1（基因关系题） | `ANSWER: B` | 1764 |
| #2（SARS-CoV-2 ORF3a 题） | `Looking at.` | 506 |
| #3（恒星硅原子比题） | `Wait, 0.3.` | 454 |

样本头部通常是正常的前置推理，进入某个状态后突然开始重复并无法跳出。

### 2.2 输出长度呈双峰分布

198 题的输出 token 数分布（`max_tokens=8192`）：

| 区间 | 样本数 |
|---|---|
| < 1000 | 111 |
| 1000–2000 | 26 |
| 2000–4000 | 0 |
| 4000–8000 | 0 |
| 8000（触顶） | 61 |

正常作答的题集中在 < 1000 token（中位数 917），loop 的题全部顶到 8192 上限，中间地带几乎为空——即"要么正常短答，要么卡死 loop"，没有"合理长推理"这一档。一旦进入 loop 就无法自行收敛。

## 3. 与 31B dense 的对照

两模型使用**完全相同**的生成配置与评测脚本，仅模型不同：

| 指标 | 26B MoE（+PR12530） | 31B dense |
|---|---|---|
| GPQA Diamond 全量 mean_acc | **0.5859** | **0.7677** |
| 样本数 | 198 | 198 |
| `max_tokens` 设置 | 8192 | 32768 |
| 触顶（达 max_tokens）样本 | 61（30.8%） | 0（0%） |
| 检测到重短语重复的样本 | 61（30.8%） | 0（0%） |
| 输出 token 中位数 | 917 | 709 |
| 输出 token 最大值 | 8192（=上限） | 9557（未触 32768 上限） |
| 平均输出 token | 3079.9 | 854.5 |
| TPOT | 22.58 ms | 31.35 ms |

31B dense 在 `max_tokens=32768` 下都没有任何样本触顶，也没有重短语重复。26B MoE 的重复退化是模型/配置特异的，不是评测脚本或服务端问题（两模型服务端在评测期间均零报错）。

## 4. PR #12530 的影响

PR #12530（`Preserve MoE routing indices during unpermute`）修复了 unpermute 阶段路由索引丢失的问题。cherry-pick 后的前 10 题小样本结果：

| 配置 | 前 10 题 mean_acc | loop 题数 |
|---|---|---|
| 26B 无修复（`max_tokens=32768`） | 0.8 | 2/10 |
| 26B + PR12530（`max_tokens=32768`） | 0.9 | 1/10 |
| 26B + PR12530（`max_tokens=8192`，全量 198） | 0.5859 | 61/198 |

前 10 题样本量过小，无法定论 PR12530 对 loop 率的改善幅度。全量 198 题显示修复后仍有 30.8% 的 loop 率，说明 **PR12530 未能消除该重复退化问题**——要么根因不在 unpermute 路由索引，要么还有其它 MoE 路由/expert 选择路径存在类似问题。

## 5. 根因推测

按可能性排序：

1. **MoE 路由 / expert 选择相关**——31B dense 在同配置下零 loop，26B MoE 31% loop，差异直接指向 MoE 机制。PR12530 修了 unpermute 的索引丢失，但 moe_block 内可能仍有路由/permute 路径在特定 token 模式下产生退化输出。建议进一步排查 `vllm_ascend/device/device_op.py` 及 MoE forward 的 unpermute/merge 路径。

2. **26B（A4B，4B 激活）active 容量较小**——相比 31B dense，高 temp 下更易出现重复退化，属模型固有的小模型倾向，叠加 MoE 放大。

3. **生成配置缺 `repetition_penalty`**——一旦模型开始重复，没有惩罚机制打破 loop，任由其顶到 `max_tokens`。

## 6. 缓解与后续验证

### 6.1 生成侧缓解（不改代码）
- 增加 `repetition_penalty=1.1~1.2`，预期可显著压低 loop 率、提升准确率；
- 或开启 thinking 模式（`thinking` 不再 disabled），让模型在推理通道内收敛；
- 或降低 `temperature`（如 0.7）。

### 6.2 代码侧验证（定位是否 routing 相关）
- 对比 cherry-pick PR12530 前后的**全量 198 题** loop 率（当前仅有"修复后"全量数据，缺"修复前"全量对照）；
- 在 loop 样本的生成步打开 `ASCEND_LAUNCH_BLOCKING=1` 抓取 MoE 路由/expert 命中分布，观察 loop token 的路由是否异常固定。

## 7. 复现

### 7.1 环境
```bash
# vllm 从源码构建（保留 torch 2.10），verified commit 见 .github/vllm-main-verified.commit
VLLM_TARGET_DEVICE=empty pip install .
pip uninstall -y triton
pip install --force-reinstall --no-deps triton-ascend==3.2.1 \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
# 注意：不要设 VLLM_VERSION=0.25.1（verified commit 是 pre-0.25.1 API）
```

### 7.2 服务（26B MoE TP4 图模式 + PR12530）
```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=256 \
vllm serve /home/xty/gemma4/26B \
  --served-model-name gemma-4-26B-A4B-it \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":2,"audio":1,"video":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

### 7.3 评测
```python
# evalscope TaskConfig 关键项
datasets=["gpqa_diamond"], dataset_hub="modelscope", limit=None  # 全量 198
generation_config={
    "temperature": 0.95, "top_p": 0.95, "top_k": 64,
    "max_tokens": 8192, "stream": True,
    "extra_body": {"thinking": {"type": "disabled"}},
}
eval_batch_size=8, seed=42
```

> 注：环境存在 HTTP 代理（`http_proxy=127.0.0.1:1080`）且 `no_proxy` 为空，evalscope 客户端会把 localhost 请求走代理导致 502；评测脚本需强制 `os.environ["no_proxy"]="127.0.0.1,localhost"`（不能用 `setdefault`）。

## 8. 数据产物

- 26B 全量评测报告：`outputs/gpqa_26b_tp4_full/20260722_043518/reports/gemma-4-26B-A4B-it/gpqa_diamond.json`
- 26B 前 10 题（PR12530）：`outputs/gpqa_26b_tp4_10/20260722_034438/`
- 31B 全量对照：`outputs/gpqa_31b_tp4/20260721_075711/reports/gemma-4-31B-it/gpqa_diamond.json`
- 服务日志：`outputs/serve26b_tp4_cp.log`、`outputs/serve31b_tp4.log`
