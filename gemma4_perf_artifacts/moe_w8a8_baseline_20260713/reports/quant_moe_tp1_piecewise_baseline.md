# Gemma4 26B MoE W8A8（仅 experts 量化）— TP1 PIECEWISE 图模式性能基线

- 日期：2026-07-13
- 权重：`/home/wangminghua/gemma4_moe_w8a8_only_experts`（128 experts，top-k 8，30 层；仅 experts 做 W8A8_DYNAMIC，其余 FLOAT；tie=False 无 lm_head 问题）
- 路径：`/home/vllm-ascend_tp/vllm-ascend_tp`（vllm 0.23.0 + vllm-ascend @ a83b055e）
- 方式：与 dense 完全相同 —— v1 引擎 + TP1 + 显式 `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`，未改代码

## 结果（两组 TPOT 均 ≤ 50ms ✅）

| 评测 | 输入→输出 | TPOT | TTFT | Decode tok/s | 成功率 |
|---|---|---|---|---|---|
| 文本（random） | 4096→1536 | **22.2ms** ✅ | 422ms | 44.94 | 100% |
| 图片（random_vl, 1920×1080） | 297→256 | **22.9ms** ✅ | 458ms | 43.76 | 100% |

MoE 比 dense（37ms）更快，因为每 token 仅激活 8/128 experts、30 层（dense 60 层）。与之前 MoE 测试（FULL_DECODE_ONLY+capture_sizes[1]，16.2ms/15.4ms）量级一致，PIECEWISE 方式同样达标且更省心（不用手调 capture_sizes）。

## 启动命令

```bash
cd /home/vllm-ascend_tp/vllm-ascend_tp
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=256 VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /home/wangminghua/gemma4_moe_w8a8_only_experts \
  --served-model-name gemma4_moe_w8a8_only_experts \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 6144 \
  --port 8080 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
```

PIECEWISE 捕获成功：`Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100% 35/35`，`Graph capturing finished in 17 secs`。KV cache 20.22 GiB。

## 评测命令

文本（10 请求 4096→1536）：
```bash
evalscope perf --parallel 1 --number 10 \
  --model gemma4_moe_w8a8_only_experts \
  --url http://127.0.0.1:8080/v1/chat/completions \
  --api openai --dataset random \
  --max-tokens 1536 --min-tokens 1536 \
  --prefix-length 0 --min-prompt-length 4096 --max-prompt-length 4096 \
  --tokenizer-path /home/wangminghua/gemma4_moe_w8a8_only_experts \
  --extra-args '{"ignore_eos": true}' --image-num 0 --no-test-connection
```
图片（12 请求 30+img→256，1920×1080）：
```bash
evalscope perf --parallel 1 --model gemma4_moe_w8a8_only_experts \
  --url http://127.0.0.1:8080/v1/chat/completions \
  --api openai --dataset random_vl \
  --min-tokens 256 --max-tokens 256 \
  --prefix-length 0 --min-prompt-length 30 --max-prompt-length 30 \
  --image-width 1920 --image-height 1080 --image-format RGB --image-num 1 \
  --number 12 --tokenizer-path /home/wangminghua/gemma4_moe_w8a8_only_experts --debug
```

## 复现最小步骤
```bash
cd /home/vllm-ascend_tp/vllm-ascend_tp
setsid nohup bash /home/xty/gemma4_moe_w8a8_baseline_20260713_084326/scripts/start_moe_tp1_piecewise.sh > /tmp/moe_server.log 2>&1 &
until curl -s http://127.0.0.1:8080/v1/models | grep -q moe; do sleep 10; done
bash /home/xty/gemma4_moe_w8a8_baseline_20260713_084326/scripts/evalpertext.sh
bash /home/xty/gemma4_moe_w8a8_baseline_20260713_084326/scripts/evalperimage.sh
find . -name performance_summary.txt -newer /tmp/moe_server.log -exec grep TPOT {} +
pkill -f "vllm serve /home/wangminghua/gemma4_moe_w8a8_only_experts"
```

## 原始日志
- 服务：`logs/server.log`（PIECEWISE 捕获成功）
- 文本：`logs/eval_text.log` + `logs/eval_text_summary.txt`（TPOT 22.2ms）
- 图片：`logs/eval_image.log` + `logs/eval_image_summary.txt`（TPOT 22.9ms）

## 结论
dense 与 MoE 两份量化权重，均用 **v1 + TP1 + 显式 PIECEWISE** 跑通，两组评测 TPOT 都 ≤ 50ms（dense 37/36ms，MoE 22/23ms），未改任何代码。8 卡已全部释放。
