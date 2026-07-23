evalscope perf \
  --parallel 1 --number 10 \
  --model gemma4_moe_w8a8_only_experts \
  --url http://127.0.0.1:8080/v1/chat/completions \
  --api openai --dataset random \
  --max-tokens 1536 --min-tokens 1536 \
  --prefix-length 0 --min-prompt-length 4096 --max-prompt-length 4096 \
  --tokenizer-path /home/wangminghua/gemma4_moe_w8a8_only_experts \
  --extra-args '{"ignore_eos": true}' \
  --image-num 0 --no-test-connection
