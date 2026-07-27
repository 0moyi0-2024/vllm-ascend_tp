#!/bin/bash
P=${1:-8000}; M=${2:-gemma4-eagle3}; PAR=${3:-16}; N=${4:-64}
evalscope perf --parallel "$PAR" --number "$N" \
  --model "$M" --url "http://127.0.0.1:${P}/v1/chat/completions" \
  --api openai --dataset random \
  --max-tokens 256 --min-tokens 256 --prefix-length 0 \
  --min-prompt-length 512 --max-prompt-length 512 \
  --tokenizer-path /home/xty/gemma4/31B \
  --extra-args '{"ignore_eos": true, "temperature": 0}' \
  --image-num 0 --no-test-connection
