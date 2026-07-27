#!/bin/bash
cd /home/vllm-ascend_tp/vllm-ascend_tp
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=256 VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /home/xty/gemma4/31B \
  --host 0.0.0.0 --port 8000 --served-model-name gemma4-eagle3 \
  --tensor-parallel-size 4 --max-num-seqs 8 \
  --max-model-len 8192 --block-size 128 \
  --speculative-config '{"model":"/home/xty/eagle3/gemma-4-31b-it-eagle3","method":"eagle3","num_speculative_tokens":2}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
