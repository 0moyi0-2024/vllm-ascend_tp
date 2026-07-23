evalscope perf \
  --parallel 1 --model gemma4_moe_w8a8_only_experts \
  --url http://127.0.0.1:8080/v1/chat/completions \
  --api openai --dataset random_vl \
  --min-tokens 256 --max-tokens 256 \
  --prefix-length 0 --min-prompt-length 30 --max-prompt-length 30 \
  --image-width 1920 --image-height 1080 --image-format RGB --image-num 1 \
  --number 12 \
  --tokenizer-path /home/wangminghua/gemma4_moe_w8a8_only_experts --debug
