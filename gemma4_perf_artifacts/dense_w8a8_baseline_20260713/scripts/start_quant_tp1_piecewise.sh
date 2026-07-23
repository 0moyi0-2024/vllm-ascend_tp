#!/bin/bash
cd /home/vllm-ascend_tp/vllm-ascend_tp
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=256
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /home/quant_ms/gemma4_w8a8 \
  --served-model-name gemma4_w8a8 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 6144 \
  --port 8080 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
