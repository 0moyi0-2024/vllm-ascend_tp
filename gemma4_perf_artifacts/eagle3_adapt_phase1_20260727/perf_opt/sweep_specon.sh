#!/bin/bash
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
cd /home/vllm-ascend_tp/vllm-ascend_tp
OUT=$(cat /tmp/eagle3_out_path.txt)
for PAR in 16 32 64; do
  N=$((PAR*4))
  echo "=== spec-on parallel=$PAR number=$N ==="
  evalscope perf --parallel "$PAR" --number "$N" \
    --model gemma4-eagle3 --url http://127.0.0.1:8000/v1/chat/completions \
    --api openai --dataset random --max-tokens 256 --min-tokens 256 \
    --prefix-length 0 --min-prompt-length 512 --max-prompt-length 512 \
    --tokenizer-path /home/xty/gemma4/31B \
    --extra-args '{"ignore_eos": true, "temperature": 0}' \
    --image-num 0 --no-test-connection > ${OUT}/perf_opt/eval_specon_p${PAR}.log 2>&1
  S=$(find outputs -name performance_summary.txt -newer ${OUT}/perf_opt/eval_specon_p${PAR}.log 2>/dev/null | sort | tail -1)
  cp "$S" ${OUT}/perf_opt/eval_specon_p${PAR}_summary.txt 2>/dev/null
  echo "done p$PAR"
  sleep 5
done
echo "SWEEP_SPECON_DONE"
