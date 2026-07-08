# Gemma4 on A5 (Ascend 950PR) W8A8 量化错误分析

本文记录在 vLLM-Ascend（分支 `A5quant`）上启动 Gemma4 W8A8 量化模型时遇到的两个问题：**NPU 设备 ID 配置错误**（环境/脚本）与 **modelslim 量化配置对 MoE experts 量化类型解析的 KeyError**（代码缺口）。给出原始报错与根因分析。

> 硬件：4 × Ascend 950PR（设备 ID 0–3，其中 NPU 0 处于 `Alarm` 状态）
> 模型：`/home/models/gemma-4-26b-a4b-it-w8a8`（W8A8 量化，`quantization=ascend`，modelslim）
> 配置：`--tensor-parallel-size 2 --enable-expert-parallel --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["none"],"cudagraph_capture_sizes":[1,2,4,8]}'`
> 启动脚本：`/home/tongpan/gemma/run_graph13.sh`，日志：`/tmp/gemma_graph_run_8013.log`

---

## 问题一：NPU 设备 ID 无效（脚本配置）

### 原始报错

```
RuntimeError: Initialize:../torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp:165
NPU function error: aclInit, error code is 107001
[ERROR] PTA call acl api failed
[Error]: Invalid device ID.
[PID: 903] ChgUserDevIdToDeviceId failed because value 0 for parameter
userDevId is invalid. Expected value: [0, 0).
rtSetDefaultDeviceId execution failed, reason=device id error
```

`Engine core initialization failed.` 之后整个服务退出。

### 根因分析

启动脚本里设置了 `ASCEND_RT_VISIBLE_DEVICES=6,7`，但本机 `npu-smi info` 与 `ls /dev/davinci*` 显示**只有 4 张 NPU（ID 0–3）**：

```
$ ls /dev/davinci*
/dev/davinci0  /dev/davinci1  /dev/davinci2  /dev/davinci3  /dev/davinci_manager
```

`6,7` 不存在，worker 进程（`multiproc_executor`）在 `check_ascend_device_type()` → `torch_npu.npu.get_soc_version()` → `aclInit` 阶段把可见设备范围过滤为空，报 `Expected value: [0, 0)`（即有效区间为空）。该脚本疑似按 8 卡机型编写，未适配当前 4 卡环境。

调用链：

```
WorkerProc.__init__
  → vllm_ascend/worker/worker.py:126  check_ascend_device_type()
  → vllm_ascend/utils.py:824           torch_npu.npu.get_soc_version()
  → torch_npu/npu/_backends.py:97      torch_npu.npu._lazy_init()
  → torch_npu._C._npu_init()           aclInit → error 107001 (Invalid device ID)
```

### 处置

非代码问题，改脚本即可。本机 NPU 0 处于 `Alarm`，故将 `ASCEND_RT_VISIBLE_DEVICES` 改为 `2,3`（状态 OK 的两张卡）：

```diff
-ASCEND_RT_VISIBLE_DEVICES=6,7 \
+ASCEND_RT_VISIBLE_DEVICES=2,3 \
```

修改后设备初始化通过，引擎继续推进到模型构建阶段，随即暴露问题二。

---

## 问题二：modelslim 量化配置对 MoE experts 解析 KeyError（代码缺口）

### 原始报错

设备改用 2,3 后，NPU 初始化通过，但在构建首个 MoE 层的 `FusedMoE` 时崩溃：

```
File ".../vllm/model_executor/models/gemma4.py", line 348, in __init__
    self.experts = FusedMoE(...)
File ".../vllm_ascend/ops/fused_moe/fused_moe_0_23_0.py", line 155, in __init__
    super().__init__(*args, **kwargs)
File ".../vllm/model_executor/layers/fused_moe/layer.py", line 365, in __init__
    self.quant_method: FusedMoEMethodBase = _get_quant_method()
File ".../vllm/model_executor/layers/fused_moe/layer.py", line 357, in _get_quant_method
    quant_method = self.quant_config.get_quant_method(self, prefix)
File ".../vllm_ascend/quantization/modelslim_config.py", line 687, in get_quant_method
    scheme = create_scheme_for_layer(self.quant_description, prefix, "moe", self.packed_modules_mapping)
File ".../vllm_ascend/quantization/modelslim_config.py", line 453, in create_scheme_for_layer
    quant_type = get_quant_type_for_layer(quant_description, prefix, layer_type, packed_modules_mapping)
File ".../vllm_ascend/quantization/modelslim_config.py", line 432, in get_quant_type_for_layer
    return get_linear_quant_type(quant_description, prefix, packed_modules_mapping)
File ".../vllm_ascend/quantization/modelslim_config.py", line 399, in get_linear_quant_type
    quant_type = quant_description[prefix + ".weight"]
KeyError: 'language_model.model.layers.0.moe.experts.weight'
```

### 根因分析

`get_quant_method` 对 `FusedMoE` 层以 `layer_type="moe"` 调用 `create_scheme_for_layer`，后者经 `get_quant_type_for_layer` 落到 `get_linear_quant_type`（`modelslim_config.py:361`）：

```python
def get_linear_quant_type(quant_description, prefix, packed_modules_mapping):
    proj_name = prefix.split(".")[-1]          # "experts"
    if proj_name in packed_modules_mapping:
        ...  # 走 shard 拆分逻辑
    else:
        quant_type = quant_description[prefix + ".weight"]   # ← 直接查 experts.weight
```

`FusedMoE` 的 `prefix` 为 `language_model.model.layers.0.moe.experts`（见 `gemma4.py:348` `prefix=f"{prefix}.experts"`），`proj_name="experts"` 不在 `packed_modules_mapping` 中，于是走 `else` 分支，直接查 `language_model.model.layers.0.moe.experts.weight`。

但该 W8A8 模型的 `quant_model_description.json` 中，MoE 权重是 **per-expert per-projection** 命名的，根本没有 `experts.weight` 这个聚合键：

```
model.language_model.layers.0.experts.0.gate_proj.weight => W8A8_DYNAMIC
model.language_model.layers.0.experts.0.up_proj.weight    => W8A8_DYNAMIC
model.language_model.layers.0.experts.0.down_proj.weight  => W8A8_DYNAMIC
model.language_model.layers.0.mlp.gate_proj.weight        => FLOAT
```

故必然 `KeyError`。

#### 为何共享 `mlp` 没报错

Gemma4 的 MoE 层同时含一个共享 expert `mlp`，其 `mlp.gate_proj.weight => FLOAT`。`get_quant_method` 在调用 `create_scheme_for_layer` 之前会先走 `is_layer_skipped_ascend`（`modelslim_config.py:681` / `700`），对值为 `"FLOAT"` 的层直接返回 `AscendUnquantizedLinearMethod` / `AscendUnquantizedFusedMoEMethod`，绕开了 `get_linear_quant_type`。因此只有真正量化的 MoE experts 在解析量化类型时才崩。

#### 已有的 minimax 特例未覆盖 gemma4

`get_quant_method` 开头对 `minimax` / `minimax_m2` 有专门的 expert 索引归一化（`modelslim_config.py:640-649`，把 `experts.0` → `experts`），但该逻辑被 `model_type` 门控，gemma4 不命中。上一相关提交 `b50823ac fix(quantization): support gemma4 k_eq_v layers in modelslim config` 只解决了 k_eq_v 层 `v_proj` shard 缺失的 `KeyError`，MoE experts 的量化类型解析仍是缺口。

### 修复方向（待实现）

在 `get_linear_quant_type` 中，当 `prefix + ".weight"` 缺失且 `proj_name == "experts"`（MoE）时，回退到从首个 expert 的投影权重解析量化类型，并校验 `gate_proj/up_proj/down_proj` 三者一致：

```python
else:
    weight_key = prefix + ".weight"
    if weight_key in quant_description:
        quant_type = quant_description[weight_key]
    elif proj_name == "experts":
        # MoE: 权重按 per-expert per-projection 命名，无聚合的 experts.weight
        quant_type = _resolve_moe_experts_quant_type(quant_description, prefix)
    else:
        quant_type = quant_description[weight_key]  # 保留原 KeyError 行为
```

`_resolve_moe_experts_quant_type` 查找 `prefix + ".0.gate_proj.weight"` / `.0.up_proj.weight` / `.0.down_proj.weight`，校验三者量化类型一致后返回；缺失或不一致时抛出明确的 `ValueError`。

> 说明：本次仅记录与分析报错，未在 `A5quant` 分支上落地修复。
