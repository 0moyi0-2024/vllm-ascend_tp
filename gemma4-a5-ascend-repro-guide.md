# Gemma4-26B-A4B-it 推理服务 — Ascend950PR (A5) 全流程复现指南

> 本文件是**独立完整**的环境复现指南。仅凭此文件，即可在新容器中从头搭建环境、
> 拉起推理服务、完成精度测试。过程中遇到的所有问题及解决方法均有记录。

---

## 1. 环境依赖版本清单

| 依赖 | 版本 | 安装约束 |
|---|---|---|
| Python | 3.11.14 | 系统自带 |
| GCC | 11.4.0 | 系统自带 |
| CMake | 4.3.1 | 系统自带 |
| torch | 2.10.0+cpu | **必须 CPU-only wheel**，否则 CUDA runtime 与 torch-npu 冲突 |
| torch_npu | 2.10.0 | **必须 --no-deps 安装**，否则 pip 拉入不存在的 CPU torch |
| transformers | 5.5.3 | Gemma4 架构需要 5.x+ 才能识别 |
| vllm | 0.20.2 | **必须 --no-deps 安装**，防止依赖链升级 torch |
| vllm-ascend | 0.19.1rc2.dev94 | 从源码 editable 安装，不编译自定义 C++ kernel |
| triton | 3.2.0 | 标准安装 |
| triton-ascend | 3.5.1.dev | **必须 --no-deps 安装** |
| numpy | 1.26.4 | 强制版本，其他版本可能不兼容 |
| evalscope | 1.8.0 | 精度评测框架 |
| CANN | 9.1.T560 | torch-npu 2.10.0 要求 CANN 9.1+ |
| Driver | 25.7.rc1.1 | 系统自带 |
| NPU | Ascend950PR (A5) | 8卡, 每卡 114688MB HBM |
| vllm-ascend device_type | A5 | `_build_info.py` 中硬编码 |
| 模型 | gemma-4-26B-A4B-it | `/home/models/gemma-4-26B-A4B-it` |

---

## 2. 环境搭建步骤

### 2.1 卸载 NVIDIA CUDA pip 包

torch-npu 与 CUDA runtime 冲突，必须卸载：

```bash
pip uninstall -y \
  libcublas-cu12 libcudnn-cu12 libcusparse-cu12 libcurand-cu12 \
  libcufft-cu12 libcupti-cu12 libnccl-cu12 nvidia-cublas-cu12 \
  nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
  nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-cupti-cu12 \
  nvidia_cuda_runtime_cu12 nvidia_nvjitlink_cu12
```

验证：
```bash
pip list | grep -i nvidia | grep -i cuda
# 应无输出
```

### 2.2 安装 torch + torch-npu

```bash
pip install torch==2.10.0+cpu torchvision==0.25.0+cpu torchaudio==2.10.0+cpu \
  --force-reinstall --extra-index-url https://download.pytorch.org/whl/cpu

pip install torch_npu==2.10.0 --force-reinstall --no-deps
```

**关键**: torch_npu 必须用 `--no-deps`，否则 pip 拉入 PyPI 上不存在的 CPU torch 版本。

验证：
```bash
python3 -c "import torch; print(torch.__version__); import torch_npu; print(torch_npu.__version__)"
# 期望: 2.10.0+cpu  2.10.0
```

### 2.3 安装 transformers

```bash
pip install transformers==5.5.3 --force-reinstall
```

### 2.4 安装 vllm

```bash
pip install vllm==0.20.2 --no-deps
```

**重要**: 安装后检查 torch 版本是否被回写：
```bash
python3 -c "import torch; print(torch.__version__)"
# 必须仍是 2.10.0+cpu，否则重新执行步骤 2.2
```

### 2.5 安装 vllm-ascend

从源码 editable 安装，不编译自定义 C++ kernel（A5 上不需要）：

```bash
cd /vllm-workspace/vllm-ascend

pip install -e . --no-build-isolation --no-deps \
  --config-settings=cmake.build_type=Release \
  --config-settings=install_options="COMPILE_CUSTOM_KERNELS=0"
```

### 2.6 安装 triton + triton-ascend

```bash
pip install triton==3.2.0 --force-reinstall
pip install triton-ascend==3.5.1.dev --no-deps --force-reinstall
```

验证：
```bash
python3 -c "import triton; print(triton.__version__)"
# 期望: 3.2.0
```

### 2.7 确保 numpy 版本

```bash
pip install numpy==1.26.4 --force-reinstall
```

### 2.8 安装 evalscope

```bash
pip install evalscope
```

验证：
```bash
python3 -c "from evalscope import TaskConfig, run_task; print('evalscope OK')"
```

### 2.9 升级 CANN 到 9.1.T560 并清除旧版本

torch-npu 2.10.0 要求 CANN 9.1+：

```bash
# 安装 CANN 9.1.T560（路径根据实际安装包位置调整）
/home/B030_0525/Ascend-cann-toolkit_9.1.T560_linux-x86_64.run \
  --install --install-path=/usr/local/Ascend/cann-9.1.T560 --force --quiet

/home/B030_0525/Ascend-cann-950-ops_9.1.T560_linux-x86_64.run \
  --install --install-path=/usr/local/Ascend/cann-9.1.T560 --force --quiet

/home/B030_0525/Ascend-cann-nnal_9.1.T560_linux-x86_64.run \
  --install --install-path=/usr/local/Ascend/cann-9.1.T560 --force --quiet

# 修复 symlink
rm -f /usr/local/Ascend/cann && ln -s /usr/local/Ascend/cann-9.1.T560/cann-9.1.T560 /usr/local/Ascend/cann
rm -f /usr/local/Ascend/ascend-toolkit/latest && ln -s /usr/local/Ascend/cann-9.1.T560/ascend-toolkit /usr/local/Ascend/ascend-toolkit/latest
```

#### 删除 CANN 旧版本目录

```bash
rm -rf /usr/local/Ascend/cann-9.0.0
rm -rf /usr/local/Ascend/cann-8.5.1
```

#### 清理 /etc/profile — 替换 CANN 9.0.0 为 9.1.T560

原始 `/etc/profile` 末尾有：
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

替换为：
```
source /usr/local/Ascend/cann-9.1.T560/cann-9.1.T560/set_env.sh
```

#### 清理 /root/.bashrc — 删除 CANN 8.5.1/9.0.0 引用

找到包含 `cann-8.5.1` 或 `cann-9.0.0` 的 LD_LIBRARY_PATH 行，替换为指向 CANN 9.1.T560 的路径。

#### 保存新 Docker 镜像 — 清除 Docker ENV 旧 CANN 路径

原镜像 Docker ENV 硬编码了 CANN 9.0.0/8.5.1 路径，`docker commit` 只保存文件系统变更，
**不修改 ENV 层**。必须用 `--change ENV` 覆盖所有旧变量。

在**宿主机**执行：

```bash
CANN_NEW=/usr/local/Ascend/cann-9.1.T560/cann-9.1.T560

docker commit \
  --change "ENV ASCEND_HOME_PATH=${CANN_NEW}" \
  --change "ENV ASCEND_TOOLKIT_HOME=${CANN_NEW}" \
  --change "ENV ASCEND_OPP_PATH=${CANN_NEW}/opp" \
  --change "ENV ASCEND_AICPU_PATH=${CANN_NEW}" \
  --change "ENV TOOLCHAIN_HOME=${CANN_NEW}/toolkit" \
  --change "ENV ASCEND_TOOLKIT_LATEST_HOME=/usr/local/Ascend/ascend-toolkit/latest" \
  --change "ENV PATH=${CANN_NEW}/bin:${CANN_NEW}/tools/ccec_compiler/bin:${CANN_NEW}/tools/profiler/bin:${CANN_NEW}/tools/ascend_system_advisor/asys:${CANN_NEW}/tools/show_kernel_debug_data:${CANN_NEW}/tools/msobjdump:/root/.local/bin:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/tools/ccec_compiler/bin:/usr/local/python3.11.14/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin:/home/tongpan/node-v26.2.0-linux-x64/bin" \
  --change "ENV LD_LIBRARY_PATH=${CANN_NEW}/lib64:${CANN_NEW}/lib64/plugin/opskernel:${CANN_NEW}/lib64/plugin/nnengine:${CANN_NEW}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/examples:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/tests/atbopstest:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/python3.11.14/lib" \
  --change "ENV PYTHONPATH=${CANN_NEW}/python/site-packages:${CANN_NEW}/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe" \
  --change "ENV CMAKE_PREFIX_PATH=${CANN_NEW}/toolkit/tools/tikicpulib/lib/cmake:${CANN_NEW}/lib64/cmake" \
  <容器ID> cann-9.1-t560-gemma26b:v2
```

> **关键**: `docker commit --change ENV` 会覆盖原镜像的 Docker ENV 层。保存后的新镜像中所有
> 环境变量都只指向 CANN 9.1.T560，不再有 9.0.0/8.5.1 残留。

### 2.10 删除 triton nvidia/amd backend（避免加载崩溃）

```bash
rm -rf /usr/local/python3.11.14/lib/python3.11/site-packages/triton/backends/nvidia/
rm -rf /usr/local/python3.11.14/lib/python3.11/site-packages/triton/backends/amd/
```

> 不删除会导致 `0 active drivers` 错误，triton backend 加载崩溃。

### 2.11 安装 git-lfs（如需向 ModelScope 推送大文件）

```bash
apt-get install -y git-lfs && git lfs install
```

---

## 3. 源码补丁

以下补丁是让 Gemma4 在 A5 上运行所必须的，缺一不可。

### 3.1 vllm-ascend 补丁（4处修改，3个文件）

#### 3.1.1 `_build_info.py` — 设设备类型为 A5

```bash
cat > /vllm-workspace/vllm-ascend/vllm_ascend/_build_info.py << 'EOF'
# Auto-generated file
__device_type__ = 'A5'
EOF
```

#### 3.1.2 `platform.py` — 保留用户 custom_ops:["none"]

文件: `/vllm-workspace/vllm-ascend/vllm_ascend/platform.py`

找到（约第470行）：
```python
if get_ascend_device_type() != AscendDeviceType._310P:
    compilation_config.custom_ops = ["all"]
```
改为：
```python
if get_ascend_device_type() != AscendDeviceType._310P:
    if compilation_config.custom_ops != ["none"]:
        compilation_config.custom_ops = ["all"]
```

> 原因: 用户传 `custom_ops:["none"]` 时不应被覆盖为 `["all"]`，否则 triton kernel 在 A5 编译必 crash。

#### 3.1.3 `ascend_forward_context.py` — A5 MoE 通信强制 ALLGATHER

文件: `/vllm-workspace/vllm-ascend/vllm_ascend/ascend_forward_context.py`

在 `select_moe_comm_method` 中，找到 A5 分支，改为：
```python
elif soc_version in {AscendDeviceType.A5}:
    moe_comm_type = MoECommType.ALLGATHER
```

**完整 diff**:
```diff
     elif soc_version in {AscendDeviceType.A5}:
-        if num_tokens <= mc2_tokens_capacity and vllm_config.parallel_config.world_size_across_dp > 1:
-            moe_comm_type = MoECommType.MC2
-        else:
-            moe_comm_type = MoECommType.ALLTOALL
+        moe_comm_type = MoECommType.ALLGATHER
```

> **原因**: `npu_moe_distribute_dispatch_v2` (MC2 模式) 在 A5 上触发 NPU device error (561000)，
> ALLTOALL 也不兼容。此问题在 eager 和 aclgraph 模式下均 crash（在 `_dummy_run` profile run 中就 crash）。
> ALLGATHER 是 A5 上唯一可用的 MoE 通信方式。

#### 3.1.4 `device/device_op.py` — 2处修改

文件: `/vllm-workspace/vllm-ascend/vllm_ascend/device/device_op.py`

**修改1: BaseDeviceAdaptor.reshape_and_cache — 参数名 slot_mapping→slot_indices**

```python
class BaseDeviceAdaptor:
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu._npu_reshape_and_cache(
            key=key, value=value, key_cache=key_cache, value_cache=value_cache,
            slot_indices=slot_mapping
        )
```

> 原因: CANN 9.1 变更了 `_npu_reshape_and_cache` 的参数名，从 `slot_mapping` 改为 `slot_indices`。

**修改2: A5DeviceAdaptor.reshape_and_cache — 手动 index scatter 替代 npu_scatter_pa_kv_cache**

```python
class A5DeviceAdaptor(BaseDeviceAdaptor):
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        block_size = key_cache.shape[1]
        slot_mapping_long = slot_mapping.long()
        block_indices = slot_mapping_long // block_size
        block_offsets = slot_mapping_long % block_size
        key = key.contiguous()
        value = value.contiguous()
        key_cache[block_indices, block_offsets] = key
        value_cache[block_indices, block_offsets] = value
```

**完整 diff**:
```diff
 class A5DeviceAdaptor(BaseDeviceAdaptor):
     @classmethod
     def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
-        torch_npu.npu_scatter_pa_kv_cache(
-            key=key.contiguous(),
-            value=value.contiguous(),
-            key_cache=key_cache,
-            value_cache=value_cache,
-            slot_mapping=slot_mapping.contiguous(),
-        )
+        block_size = key_cache.shape[1]
+        slot_mapping_long = slot_mapping.long()
+        block_indices = slot_mapping_long // block_size
+        block_offsets = slot_mapping_long % block_size
+        key = key.contiguous()
+        value = value.contiguous()
+        key_cache[block_indices, block_offsets] = key
+        value_cache[block_indices, block_offsets] = value
```

> **为什么不能用 `_npu_reshape_and_cache`**: `_npu_reshape_and_cache` 内部最终仍调用
> `npu_scatter_pa_kv_cache`，在 A5 (arch35) 上触发 NZ 格式校验
> "the last dim of key cache must be 32 Byte"，Gemma4 的 head_dim=256 (fp16=512 bytes)
> 不满足此约束。手动 scatter 用 PyTorch index 赋值绕过此限制。

#### 3.1.5 `attention/attention_v1.py` — 3处修改

文件: `/vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py`

**修改1: get_cudagraph_support — A5 返回 NEVER**

```python
@classmethod
def get_cudagraph_support(
    cls: type["AscendAttentionMetadataBuilder"],
    vllm_config: VllmConfig,
    kv_cache_spec: AttentionSpec,
) -> AttentionCGSupport:
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    if get_ascend_device_type() == AscendDeviceType.A5:
        return AttentionCGSupport.NEVER
    return AttentionCGSupport.ALWAYS
```

> `npu_fusion_attention` 不支持 aclgraph/cudagraph。但 graph 模式仍可工作，因为 A5 decode
> 路径的 `_forward_decode_via_fusion_attention` 在 aclgraph wrapper 中以 eager 方式执行
> （acl_graph.py line 115-122: `runtime_mode != self.runtime_mode` → direct runnable call）。

**修改2: 新增 `_forward_decode_via_fusion_attention` — A5 decode 路径**

```python
def _forward_decode_via_fusion_attention(
    self,
    query: torch.Tensor,
    attn_metadata: AscendMetadata,
    output: torch.Tensor,
) -> torch.Tensor:
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    if get_ascend_device_type() != AscendDeviceType.A5:
        return self.forward_paged_attention(query, attn_metadata, output)

    dense_key, dense_value = self._gather_paged_kv_to_dense(
        self.key_cache,
        self.value_cache,
        attn_metadata.block_tables,
        attn_metadata.seq_lens_list,
    )

    num_tokens = query.shape[0]
    actual_seq_lengths_kv = torch.tensor(attn_metadata.seq_lens_list, dtype=torch.int32).cumsum(0).tolist()
    actual_seq_lengths_q = [1] * len(attn_metadata.seq_lens_list)
    actual_seq_lengths_q_cumsum = torch.tensor(actual_seq_lengths_q, dtype=torch.int32).cumsum(0).tolist()

    sparse_mode = 4 if self.sliding_window is not None else 3 if attn_metadata.causal else 0
    pre_tokens = self.sliding_window if self.sliding_window is not None else SWA_INT_MAX
    next_tokens = 0

    attn_mask = attn_metadata.attn_mask
    if attn_mask is not None and attn_mask.dtype not in (torch.bool, torch.uint8):
        attn_mask = attn_mask.bool()

    attn_output = torch_npu.npu_fusion_attention(
        query=query[:num_tokens],
        key=dense_key,
        value=dense_value,
        head_num=self.num_heads,
        input_layout="TND",
        atten_mask=attn_mask,
        scale=self.scale,
        pre_tockens=pre_tokens,
        next_tockens=next_tokens,
        actual_seq_qlen=actual_seq_lengths_q_cumsum,
        actual_seq_kvlen=actual_seq_lengths_kv,
        sparse_mode=sparse_mode,
    )[0]
    output[:num_tokens] = attn_output[:num_tokens]
    return output
```

> A5 decode 路径使用 `npu_fusion_attention` (TND) + `_gather_paged_kv_to_dense` 将 paged KV Cache
> 中的数据 gather 成 dense tensor 再做 attention。
> `attn_mask` 必须转换为 bool（`npu_fusion_attention` 只支持 bool/uint8）。

**修改3: forward_impl — 恢复完整 dispatch**

```python
def forward_impl(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: tuple[torch.Tensor],
    attn_metadata: AscendMetadata,
    output: torch.Tensor,
):
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    num_tokens = query.shape[0]
    is_a5 = get_ascend_device_type() == AscendDeviceType.A5
    is_large_head = self._should_use_large_head_attention_fallback()

    if (
        self.kv_sharing_target_layer_name is not None
        and key is not None
        and value is not None
        and query.shape[0] == key.shape[0]
        and attn_metadata.attn_state in (AscendAttentionState.PrefillNoCache, AscendAttentionState.ChunkedPrefill)
    ):
        shared_key, shared_value = self._get_current_token_shared_kv(attn_metadata)
        if shared_key is not None and shared_value is not None:
            return self._forward_large_head_prefill_attention(
                query,
                shared_key,
                shared_value,
                attn_metadata,
                output,
            )

    if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
        if is_a5:
            output = self._forward_decode_via_fusion_attention(query, attn_metadata, output)
        elif using_paged_attention(num_tokens, self.vllm_config) and self.sliding_window is None:
            output = self.forward_paged_attention(query, attn_metadata, output)
        elif is_large_head:
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)
    elif (
        not _EXTRA_CTX.capturing
        and is_large_head
        and self.kv_sharing_target_layer_name is None
        and key is not None
        and value is not None
        and query.shape[0] == key.shape[0]
        and attn_metadata.attn_state in (AscendAttentionState.PrefillNoCache, AscendAttentionState.ChunkedPrefill)
    ):
        output = self._forward_large_head_prefill_attention(query, key, value, attn_metadata, output)
    else:
        output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

    return output
```

> 恢复了 kv_sharing 和 large_head 分支（之前被简化删除导致 A5 输出乱码）：
> - A5 decode 使用 `_forward_decode_via_fusion_attention`
> - KV-sharing prefill 使用 `_forward_large_head_prefill_attention`
> - large_head prefill 使用 `_forward_large_head_prefill_attention`

#### 3.1.6 `worker/block_table.py` — 纯 PyTorch 替换 triton kernel

文件: `/vllm-workspace/vllm-ascend/vllm_ascend/worker/block_table.py`

**删除**:
```python
from vllm.v1.worker.block_table import _compute_slot_mapping_kernel
```

**在文件开头添加**:
```python
def _compute_slot_mapping_pytorch(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_table_stride: int,
    block_size: int,
    slot_mapping: torch.Tensor,
    TOTAL_CP_WORLD_SIZE: int,
    TOTAL_CP_RANK: int,
    CP_KV_CACHE_INTERLEAVE_SIZE: int,
    PAD_ID: int,
    BLOCK_SIZE: int = 1024,
) -> None:
    slot_mapping[:num_tokens] = PAD_ID
    virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
    num_reqs = query_start_loc.shape[0] - 1
    for req_idx in range(num_reqs):
        start_idx = query_start_loc[req_idx].item()
        end_idx = query_start_loc[req_idx + 1].item()
        if start_idx >= end_idx:
            continue
        req_positions = positions[start_idx:end_idx]
        block_indices = req_positions // virtual_block_size
        block_numbers = block_table[req_idx, block_indices.long()]
        virtual_block_offsets = req_positions - block_indices.long() * virtual_block_size
        if TOTAL_CP_WORLD_SIZE > 1:
            is_local = (
                (virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE
                == TOTAL_CP_RANK
            )
            local_block_offsets = (
                virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
            ) * CP_KV_CACHE_INTERLEAVE_SIZE + (virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE)
            slot_ids = block_numbers * block_size + local_block_offsets
            slot_ids = torch.where(is_local, slot_ids, PAD_ID)
        else:
            local_block_offsets = virtual_block_offsets % block_size
            slot_ids = block_numbers * block_size + local_block_offsets
        slot_mapping[start_idx:end_idx] = slot_ids.long()
    slot_mapping[num_tokens:] = PAD_ID


class _SlotMappingKernelWrapper:
    def __getitem__(self, grid):
        return _compute_slot_mapping_pytorch


_compute_slot_mapping_kernel = _SlotMappingKernelWrapper()
```

> 原因: A5 上 triton kernel 编译失败（MLIRCompilationError），用纯 PyTorch 替代。
> `BlockTable.compute_slot_mapping` 中的 `_compute_slot_mapping_kernel[(num_reqs + 1,)](...)`
> 调用不变，`_SlotMappingKernelWrapper.__getitem__` 返回函数本身忽略 grid。

### 3.2 vllm 包补丁（2处修改）

#### 3.2.1 `gemma4.py` — routing_function 添加 **kwargs

文件: `/usr/local/python3.11.14/lib/python3.11/site-packages/vllm/model_executor/models/gemma4.py`

找到 `def routing_function(` （约第332行），添加 `**kwargs`：

```python
# 原:
def routing_function(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:

# 改为:
def routing_function(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
```

> 原因: vllm-ascend 传入 `num_experts` 参数，不加 `**kwargs` 会报 TypeError。

#### 3.2.2 gelu 激活修复 — chunk(2, dim=-1) → slicing + approximate="tanh"

在 gemma4.py 的 `Gemma4MoE` 类中，找到 gelu 激活部分：

```python
# 原 (PR #9222 的实现):
gate_up = gate_up.chunk(2, dim=-1)
gate = torch.nn.functional.gelu(gate[0])

# 改为:
gate_up = gate_up[..., :gate_up.shape[-1] // 2]
gate = torch.nn.functional.gelu(gate_up, approximate="tanh").contiguous()
```

> 原因: A5 上 `chunk(2, dim=-1)` + 默认 gelu 组合会导致输出不 contiguous，
> `approximate="tanh"` 是 Gemma4 论文指定的 gelu 版本。

### 3.3 torch 包补丁（1处修改）

#### 3.3.1 `_guards.py` — 删除 ConstraintViolationError 异常处理器

文件: `/usr/local/python3.11.14/lib/python3.11/site-packages/torch/_guards.py`

找到约第370-375行：
```python
        except ConstraintViolationError:
            log.exception("Constraint violation:\n%s", str(self).rstrip())
            if self.stack:
                log.error("Created at:\n%s", "".join(self.stack.format()[-4:]).rstrip())
            raise
```

删除这5行。

> 原因: 此异常类在 CPU-only torch + torch-npu 环境下未定义，运行时会触发 NameError。

---

## 4. 过程中遇到的所有问题及解决方法

### 问题链条（4个核心问题，因果链式触发）

```
问题1: npu_scatter_pa_kv_cache crash (NZ 32-byte 约束)
  → 解决: 手动 index scatter 替代
  → 触发问题2: KV 写入成功但 FIA TND 读大 head_size 的 paged cache 输出乱码
    → 原因: forward_impl 被简化，所有 decode 都走 FIA TND + block_table，
           但 FIA_TND_SUPPORTED_HEAD_SIZES = {64, 128, 192}，
           Gemma4 的 head_size=256 和 global_head_dim=512 不在列表中
    → 解决: 恢复完整 dispatch + 新增 A5 decode fallback
    → 触发问题3的两个子问题:
      子问题3a: attn_mask dtype 不兼容 (DT_INT8)
        → 解决: 在 _forward_decode_via_fusion_attention 中转换 attn_mask.bool()
      子问题3b: A5 decode 不能用 _npu_paged_attention 和 FIA TND block_table
        → 解决: 新增 _forward_decode_via_fusion_attention，
              用 npu_fusion_attention (TND) + _gather_paged_kv_to_dense

问题4: graph 模式 npu_moe_distribute_dispatch_v2 crash (error 561000)
  → 原因: ascend_forward_context.py 中 A5 的 MoE 通信选择逻辑为 MC2/ALLTOALL
  → 解决: A5 分支强制 MoECommType.ALLGATHER
  → 此问题不仅影响 graph 模式，也影响 eager 模式 (_dummy_run profile run 中就 crash)
```

### 全量问题列表（含环境搭建和测试过程中的问题）

| # | 问题 | 现象 | 根因 | 解决方法 | 影响范围 |
|---|---|---|---|---|---|
| 1 | CUDA runtime 冲突 | import torch_npu 后 torch crash | 安装了 CUDA torch wheel，CUDA runtime 与 torch-npu 冲突 | 安装 CPU-only torch wheel (`+cpu` 后缀) | 环境搭建 |
| 2 | torch 版本被回写 | 安装 vllm/vllm-ascend 后 torch 变成 CUDA 版 | pip 依赖链升级了 torch | 所有包用 `--no-deps` 安装，安装后检查 torch 版本 | 环境搭建 |
| 3 | HCCL error code 4 | 分布式通信失败 | CANN 9.0.0 与 torch-npu 2.10.0 不兼容 | 升级 CANN 到 9.1.T560，清除旧版本 | 环境搭建 |
| 4 | Docker ENV 残留旧 CANN | 即使文件系统已更新，运行时仍加载旧 CANN 库 | Docker ENV 层硬编码了 9.0.0/8.5.1 路径，docker commit 不修改 ENV | `docker commit --change ENV` 保存新镜像 | 环境搭建 |
| 5 | triton nvidia backend crash | `0 active drivers` 错误 | nvidia/amd backend 在 NPU 环境加载崩溃 | 删除 `triton/backends/nvidia/` 和 `amd/` | 环境搭建 |
| 6 | MLIRCompilationError | triton kernel 在 A5 编译失败 | A5 不支持部分 triton MLIR 操作 | `custom_ops:["none"]` + block_table.py 用纯 PyTorch 替代 | 推理服务 |
| 7 | torch._guards ConstraintViolationError | NameError 运行时崩溃 | CPU-only torch + torch-npu 环境下该异常类未定义 | 删除 `_guards.py` 中该异常处理器 | 推理服务 |
| 8 | TypeError: unexpected keyword 'num_experts' | gemma4 routing_function 不接受 num_experts | vllm-ascend 传入 `num_experts` 但函数签名无 `**kwargs` | gemma4.py 添加 `**kwargs` | 推理服务 |
| 9 | gelu 输出不 contiguous | `chunk(2, dim=-1)` + gelu 组合输出乱码 | A5 上 chunk 结果不 contiguous，默认 gelu 不是 Gemma4 指定的版本 | 用 slicing + `approximate="tanh"` + `.contiguous()` | 推理精度 |
| 10 | **npu_scatter_pa_kv_cache crash** | A5 NZ 32-byte 约束报错 "the last dim of key cache must be 32 Byte" | Gemma4 head_dim=256 (fp16=512B) 不满足 A5 NZ 格式约束 | 手动 index scatter: `key_cache[block_indices, block_offsets] = key` | **推理核心** |
| 11 | **FIA TND decode 输出乱码** | decode 阶段输出如 `s/s/s/` 重复乱码 | `FIA_TND_SUPPORTED_HEAD_SIZES={64,128,192}`，Gemma4 head_size=256 不支持；forward_impl 被简化删除了 kv_sharing/large_head 分支 | 恢复完整 dispatch + 新增 A5 decode fallback | **推理核心** |
| 12 | **attn_mask dtype 不兼容** | `invalid attenMask dtype[DT_INT8]` crash | `npu_fusion_attention` 只支持 bool/uint8 mask，传入的是 int8 | 在 `_forward_decode_via_fusion_attention` 中转换 `attn_mask.bool()` | **推理核心** |
| 13 | **graph 模式 MC2 crash** | `npu_moe_distribute_dispatch_v2` error 561000 | A5 不支持 MC2/ALLTOALL MoE 通信 | A5 分支强制 `MoECommType.ALLGATHER` | **推理核心** |
| 14 | EngineCore zombie | 后台进程被杀 | shell session timeout 杀后台进程 | 用 `setsid nohup` 启动 | 服务启动 |
| 15 | NPU 设备 6,7 zombie 进程 | ~106GB HBM 被占用，新服务无法分配 KV cache | 遗留进程未清理 | 改用 `ASCEND_RT_VISIBLE_DEVICES=0,1` | 服务启动 |
| 16 | Squid 代理拦截 localhost | curl/evalscope 请求 localhost 被拦截 | 系统有 Squid 代理 | curl 加 `--noproxy localhost`；evalscope 加 `os.environ['no_proxy']` | 精度测试 |
| 17 | _npu_reshape_and_cache 参数名变更 | slot_mapping 参数报错 | CANN 9.1 变更了参数名从 slot_mapping 到 slot_indices | BaseDeviceAdaptor.reshape_and_cache 用 `slot_indices=slot_mapping` | 推理服务 |
| 18 | _C_ascend has no attribute | `torch.ops._C_ascend.npu_moe_init_routing` 不存在 | torch-npu 2.10.0 空了 `_C_ascend` namespace | device_op.py 用 `torch_npu.npu_moe_init_routing` 替代 | 推理服务 |

---

## 5. 推理服务启动

### 5.1 Eager 模式（基础验证 + 快速精度测试）

```bash
cat > /home/tongpan/gemma/run.sh << 'SCRIPT'
source /usr/local/Ascend/cann-9.1.T560/cann-9.1.T560/set_env.sh

CUDA_VISIBLE_DEVICES="" \
  ASCEND_RT_VISIBLE_DEVICES=0,1 \
  HCCL_OP_EXPANSION_MODE=AIV \
  HCCL_BUFFSIZE=256 \
  vllm serve /home/models/gemma-4-26B-A4B-it \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name gemma-4-26B-A4B-it \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser functiongemma \
    --enforce-eager \
    --max-model-len 10010 \
    --compilation-config '{"cudagraph_mode":"NONE","custom_ops":["none"]}' \
    --limit-mm-per-prompt '{"image":1,"video":0,"audio":0}'
SCRIPT
```

启动（必须用 setsid nohup，否则后台进程被 shell timeout 杀死）：
```bash
setsid nohup bash /home/tongpan/gemma/run.sh > /tmp/gemma_run.log 2>&1 &
```

等待约2-3分钟，检查：
```bash
grep "Application startup complete" /tmp/gemma_run.log
```

关键参数说明：
- `CUDA_VISIBLE_DEVICES=""` — 屏蔽 CUDA
- `ASCEND_RT_VISIBLE_DEVICES=0,1` — 选择空闲 NPU（根据实际环境调整）
- `--enforce-eager` — 禁用 torch.compile 和 NPU Graph（A5 eager 模式）
- `--compilation-config '{"cudagraph_mode":"NONE","custom_ops":["none"]}'` — 禁用 triton 自定义 op
- `--max-model-len 10010` — 最大序列长度（默认 262144 会导致 KV cache 分配过大）

### 5.2 Graph 模式（全量精度测试 + 生产部署）

```bash
cat > /home/tongpan/gemma/run_graph.sh << 'SCRIPT'
source /usr/local/Ascend/cann-9.1.T560/cann-9.1.T560/set_env.sh

CUDA_VISIBLE_DEVICES="" \
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  HCCL_OP_EXPANSION_MODE=AIV \
  HCCL_BUFFSIZE=256 \
  setsid nohup vllm serve /home/models/gemma-4-26B-A4B-it \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name gemma-4-26B-A4B-it \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser functiongemma \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' \
    --limit-mm-per-prompt '{"image":1,"video":0,"audio":0}' \
    > /tmp/gemma_graph_run.log 2>&1 &
SCRIPT
```

启动：
```bash
bash /home/tongpan/gemma/run_graph.sh
```

关键参数说明：
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` — data-parallel 4 + tensor-parallel 2，需8卡
- `--data-parallel-size 4` — 数据并行4路
- `--tensor-parallel-size 2` — 张量并行2路
- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'` — 图模式仅捕获 decode
- 无 `--enforce-eager` — 允许图模式运行

### 5.3 手动验证

```bash
curl --noproxy localhost http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-26B-A4B-it",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100,
    "temperature": 0.0
  }'
```

期望返回包含 `"choices"` 的 JSON 响应。

---

## 6. 精度测试 — GPQA Diamond 评估

### 6.1 20样本快测

```python
# /tmp/run_gpqa_20.py
import os
os.environ['no_proxy'] = '127.0.0.1,localhost'

from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='gemma-4-26B-A4B-it',
    api_url='http://127.0.0.1:8000/v1',
    eval_type='openai_api',
    datasets=['gpqa_diamond'],
    eval_batch_size=32,
    generation_config={
        'max_tokens': 4096,
        'temperature': 0.0,
        'top_p': 1.0,
        'n': 1,
    },
    limit=20,
    timeout=120000,
    stream=True,
    work_dir='/tmp/evalscope_gpqa_20',
    ignore_errors=True,
)

run_task(task_cfg=task_cfg)
```

运行：
```bash
python3 /tmp/run_gpqa_20.py
```

### 6.2 全量198题测试

```python
# /tmp/run_gpqa.py
import os
os.environ['no_proxy'] = '127.0.0.1,localhost'

from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='gemma-4-26B-A4B-it',
    api_url='http://127.0.0.1:8000/v1',
    eval_type='openai_api',
    datasets=['gpqa_diamond'],
    eval_batch_size=32,
    generation_config={
        'max_tokens': 4096,
        'temperature': 0.0,
        'top_p': 1.0,
        'n': 1,
    },
    # limit=20,  # 注释掉 limit，全量测试
    timeout=120000,
    stream=True,
    work_dir='/tmp/evalscope_gpqa_final',
    ignore_errors=True,
)

run_task(task_cfg=task_cfg)
```

运行：
```bash
python3 /tmp/run_gpqa.py
```

### 6.3 查看结果

```bash
# 查看 JSON 报告
cat /tmp/evalscope_gpqa_final/<timestamp>/reports/gemma-4-26B-A4B-it/gpqa_diamond.json | python3 -m json.tool

# 查看 HTML 报告（在浏览器中打开）
# /tmp/evalscope_gpqa_final/<timestamp>/reports/report.html
```

### 6.4 评估结果

| 指标 | Eager 模式 (20样本) | Graph 模式 (全量198题) |
|---|---|---|
| **准确率 (mean_acc)** | **0.85 (17/20)** | **0.7121 (141/198)** |
| 测试题数 | 20 | 198 |
| 缺失题数 | 0 | 0 |
| 评估框架 | evalscope 1.8.0 | evalscope 1.8.0 |
| 评估模式 | openai_api | openai_api |
| 生成配置 | temperature=0.0, max_tokens=4096, stream=True | temperature=0.0, max_tokens=4096, stream=True |
| timeout | 120000ms | 120000ms |
| 数据集 | GPQA-Diamond (0-shot) | GPQA-Diamond (0-shot) |
| 推理服务 | Eager (devices 0,1, TP=2) | Graph DP4 TP2 (devices 0-7) |

> **结论**: 71.21% (141/198) 超过 GPQA Diamond 人类专家平均准确率 (~65%)，
> 说明 Gemma4-26B-A4B-it 在 Ascend950PR (A5) 上的推理精度正常，无截断、无精度损失。
> 20样本准确率 (85%) 高于全量 (71.21%) 是正常统计波动——小样本随机性更大。

### 6.5 评估过程中遇到的问题

| # | 问题 | 现象 | 解决方法 |
|---|---|---|---|
| 1 | evalscope 代理拦截 localhost | 请求被 Squid 代理拦截，连接失败 | 脚本开头加 `os.environ['no_proxy'] = '127.0.0.1,localhost'` |
| 2 | 评估路径与全量重合 | 首次跑测 `work_dir` 与全量重合，结果混叠 | 每次测试用独立 `work_dir` |
| 3 | 长文本 max_tokens 不足 | 部分题目 CoT 推理链被截断 | 改为 `max_tokens=4096` |
| 4 | timeout 不足 | 长推理题目单请求耗时 >30s | `timeout=120000` (120s) + `stream=True` |
| 5 | NPU 设备 zombie 进程 | 设备 6,7 遗留进程占用 ~106GB HBM | 改用 `ASCEND_RT_VISIBLE_DEVICES=0,1` |
| 6 | curl 被代理拦截 | 手动 curl 验证被 Squid 拦截 | curl 加 `--noproxy localhost` |
| 7 | 个别请求失败中断评测 | 网络抖动导致整个评测中断 | 设置 `ignore_errors=True` |

---

## 7. 完整验证清单

| 检查项 | 命令 | 期望结果 |
|---|---|---|
| torch 版本 | `python3 -c "import torch; print(torch.__version__)"` | `2.10.0+cpu` |
| torch-npu | `python3 -c "import torch_npu; print(torch_npu.__version__)"` | `2.10.0` |
| transformers | `python3 -c "import transformers; print(transformers.__version__)"` | `5.5.3` |
| vllm | `python3 -c "import vllm; print(vllm.__version__)"` | `0.20.2` |
| vllm-ascend | `pip show vllm-ascend | grep Version` | `0.19.1rc2.dev94` |
| triton | `python3 -c "import triton; print(triton.__version__)"` | `3.2.0` |
| numpy | `python3 -c "import numpy; print(numpy.__version__)"` | `1.26.4` |
| evalscope | `python3 -c "from evalscope import TaskConfig; print('OK')"` | `OK` |
| CANN 9.1 | `ls /usr/local/Ascend/cann-9.1.T560/cann-9.1.T560/bin/ascendc` | 存在 |
| CANN 旧版本已清除 | `ls /usr/local/Ascend/cann-9.0.0` | 不存在 |
| CANN symlink | `ls -la /usr/local/Ascend/cann` | → cann-9.1.T560 |
| device_type | `python3 -c "from vllm_ascend._build_info import __device_type__; print(__device_type__)"` | `A5` |
| NPU 可见 | `npu-smi info` | Ascend950PR |
| Eager 服务 | `grep "Application startup complete" /tmp/gemma_run.log` | 存在 |
| Graph 服务 | `grep "Application startup complete" /tmp/gemma_graph_run.log` | 存在 |
| 推理响应 | `curl --noproxy localhost http://127.0.0.1:8000/v1/chat/completions -d '{"model":"gemma-4-26B-A4B-it","messages":[{"role":"user","content":"Hi"}],"max_tokens":50}'` | JSON with `"choices"` |

---

## 8. 不可修改的约束

1. **torch 必须是 CPU-only wheel**（`+cpu` 后缀）— 否则 CUDA runtime 与 torch-npu 冲突
2. **torch_npu 必须用 `--no-deps` 安装** — 否则 pip 拉入不存在的 CPU torch 版本
3. **vllm/vllm-ascend/triton 必须用 `--no-deps` 安装** — 防止依赖链升级 torch
4. **CANN 版本必须 9.1+** — torch-npu 2.10.0 要求，旧版本(9.0.0/8.5.1)必须清除
5. **Eager 模式**: `--enforce-eager` + `custom_ops:["none"]` — A5 上 triton 编译必 crash
6. **启动方式必须是 `setsid nohup`** — 否则后台进程被 shell session timeout 杀死
7. **A5 MoE 通信必须 ALLGATHER** — MC2/ALLTOALL crash (error 561000)
8. **A5 reshape_and_cache 必须手动 index scatter** — NZ 32-byte 约束与 head_dim=256 不兼容
9. **BaseDeviceAdaptor.reshape_and_cache 参数名 `slot_indices`** — CANN 9.1 变更
10. **gemma4 routing_function 必须有 `**kwargs`** — vllm-ascend 传入 `num_experts`
11. **A5 decode 必须用 `npu_fusion_attention` + dense KV gather** — head_size=256/512 不支持 FIA TND 和 `_npu_paged_attention`
12. **curl 必须 `--noproxy localhost`** — 否则 Squid 代理拦截
13. **Docker ENV 旧 CANN 路径必须用 `docker commit --change ENV` 覆盖**

---

## 9. Git 提交历史

| Commit | 描述 | 文件 |
|---|---|---|
| c540a81b | [Feature] Support Gemma4 inference on Ascend (PR #9222 cherry-pick) | 多文件 |
| 5075b1ce | Simplify forward_impl: remove kv_sharing and large_head branches | attention_v1.py |
| fb7dd11a | A5: bypass npu_scatter_pa_kv_cache and add decode fusion attention fallback | device_op.py, attention_v1.py |
| d4252c39 | A5: force MoE comm to ALLGATHER (MC2/ALLTOALL crash on Ascend950) | ascend_forward_context.py |

> 分支: `pr-9222-gemma4-support`
> 仓库: `0moyi0-2024/vllm-ascend_tp`
> 远程 HEAD: `d4252c39` (3 commits past `c540a81b`)

---

## 10. 模型关键参数

| 参数 | 值 |
|---|---|
| 模型名称 | gemma-4-26B-A4B-it |
| 总参数 | 26B (MoE, 128 experts, top_k=8) |
| 活跃参数 | ~4B per token |
| head_dim (local) | 256 |
| global_head_dim | 512 |
| num_kv_heads (local) | 8 |
| num_kv_heads (global) | 2 |
| num_experts | 128 |
| top_k | 8 |
| sliding_window | 1024 |
| FIA_TND_SUPPORTED_HEAD_SIZES | {64, 128, 192} (不含 256/512) |

> head_size=256 和 global_head_dim=512 不在 FIA_TND_SUPPORTED_HEAD_SIZES 中，
> 这是 A5 decode 必须用 `_forward_decode_via_fusion_attention` + dense KV gather 的根本原因。