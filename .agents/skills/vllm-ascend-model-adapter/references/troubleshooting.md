# Troubleshooting

## Direct run doesn't pick your code changes

Symptoms:

- `vllm serve` behavior still old after code edits.

Actions:

1. Check runtime import path:
   ```bash
   python - <<'PY'
   import vllm
   print(vllm.__file__)
   PY
   ```
2. Ensure edits were made under `/vllm-workspace/vllm` and/or `/vllm-workspace/vllm-ascend`.
3. Avoid PYTHONPATH-overlay workflow unless as temporary debugging fallback.

## Server fails to bind on `:8000` or fails with HCCL bind errors

Symptoms:

- Port bind fail on startup.
- HCCL error like `Communication_Error_Bind_IP_Port(EJ0003)`.

Actions:

1. Kill stale `vllm serve` processes.
2. Ensure `:8000` is free.
3. Retry clean startup before changing code.

## Startup appears "stuck" in graph mode

Symptoms:

- Process alive, but `curl /v1/models` not ready yet.
- Logs show compile/graph capture messages for a long time.

Actions:

1. Keep waiting until graph capture completes.
2. Look for `Capturing CUDA graphs ...` and `Graph capturing finished`.
3. Only declare failure after an explicit error or timeout window.

## False-ready: startup succeeds but first request crashes

Symptoms:

- `Application startup complete` exists.
- `GET /v1/models` may return 200.
- First text or VL request crashes workers/engine.

Actions:

1. Always run at least one text smoke request immediately after ready.
2. For VL models, always run one text+image smoke request as well.
3. Treat first-request crash as runtime failure (do not mark as success).
4. Capture first runtime error signature and branch to targeted fallback.

## Architecture not recognized

Symptoms:

- `ValueError` or log shows unresolved architecture.

Actions:

1. Verify `architectures` in model `config.json`.
2. Add mapping to `vllm/model_executor/models/registry.py`.
3. Ensure module and class names exactly match.

## Remote code import fails on transformers symbols

Symptoms:

- Missing class/function in current `transformers`.

Actions:

1. Do not upgrade `transformers`.
2. Prefer native vLLM implementation.
3. If unavoidable, copy required modeling files from sibling transformers source.

## Weight loading key mismatch

Symptoms:

- Missing/unexpected key warnings during load.

Actions:

1. Inspect checkpoint key prefixes.
2. Add explicit mapping logic.
3. Keep mapping minimal and auditable.
4. Re-test with full shards, not only tiny-layer smoke runs.

## FP8 checkpoint on Ascend A2/A3 (must dequant to bf16)

Symptoms:

- fp8 kernels unsupported or unstable on Ascend A2/A3.

Actions:

1. Do not force fp8 quantization kernels on Ascend.
2. Use load-time fp8->bf16 dequantization path (weight + scale pairing).
3. Add strict unpaired scale/weight checks to avoid silent corruption.

## QK norm mismatch (KV heads / TP / head divisibility)

Symptoms:

- Shape mismatch like `128 vs 64` when `tp_size > num_key_value_heads`.
- Similar mismatch when head topology is not cleanly divisible.

Actions:

1. Detect KV-head replication case.
2. Use local `k_norm` shard path for replicated KV heads.
3. Avoid assumptions that all head dimensions split evenly under current TP.
4. Validate both normal and edge topology cases explicitly.

## MLA attention runtime failures after ready

Symptoms:

- First request fails with signatures like `AtbRingMLAGetWorkspaceSize` / `AtbRingMLA`.
- May also show `aclnnFusedInferAttentionScoreV3 ... error code 561002`.

Actions:

1. Reproduce with one minimal text request (deterministic payload).
2. Try eager isolation (`--enforce-eager`) once to verify whether issue is graph-only.
3. If eager still fails, prioritize model/backend code fix path (not runtime flags only).
4. Check `vllm-ascend` MLA/rope/platform implementation used by known-good runs.

## VL + TorchDynamo interpolate contiguous failure

Symptoms:

- `torch._dynamo.exc.TorchRuntimeError`.
- Stack contains `torch.nn.functional.interpolate`.
- Error contains `NPU contiguous operator only supported contiguous memory format`.

Actions:

1. Add `TORCHDYNAMO_DISABLE=1` and retry with same serve args.
2. Validate both text and text+image after startup.
3. If this stabilizes startup and inference, record it as current fallback path.
4. Keep code-level fix exploration as next step, but do not block delivery if fallback is accepted.

## Multimodal processor signature mismatch (`skip_tensor_conversion`)

Symptoms:

- Early failure before engine ready.
- `convert_to_tensors() got an unexpected keyword argument 'skip_tensor_conversion'`.

Actions:

1. Identify processor compatibility mismatch (HF remote processor vs current transformers API).
2. Use text-only isolation (`--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'`) only to separate layers, not as final fix.
3. Expect potential follow-up core failures after bypassing processor path; keep logs for both layers.
4. Align to known-good model dispatch and processor compatibility implementation.

## Text-only isolation triggers meta tensor load errors

Symptoms:

- `NotImplementedError: Cannot copy out of meta tensor; no data!`
- May occur after disabling multimodal prompt items.

Actions:

1. Treat as secondary failure signature (after bypassing earlier MM-processor failure).
2. Do not assume text-only isolation is universally safe for all VL models.
3. Return to model-specific code-fix path with captured signatures.

## Config max length works on paper but not in runtime

Symptoms:

- `max_position_embeddings` is large, but service fails or OOM with that value.

Actions:

1. Record config max (theoretical).
2. Find practical max by successful startup + serving under target TP/EP setup.
3. Report both values explicitly in docs.

## flashcomm1 / MTP confusion on VL checkpoints

Symptoms:

- flashcomm1 enabled but startup fails.
- MTP expected but no effect.

Actions:

1. Only validate flashcomm1 for MoE models; non-MoE mark as not-applicable.
2. Verify MTP from both config and weight index (`mtp/nextn` keys).
3. Mark unsupported vs checkpoint-missing clearly.

## ACL graph capture fails (507903)

Symptoms:

- `AclmdlRICaptureEnd ... 507903`
- `rtStreamEndCapture ... invalidated stream capture sequence`

Actions:

1. Prefer `HCCL_OP_EXPANSION_MODE=AIV` for graph capture stability.
2. Reduce shape pressure (`--max-model-len`) and retry.
3. Temporarily fallback `--enforce-eager` for isolation.

## API reachable but output quality odd

Symptoms:

- `/v1/models` works but output has template artifacts.

Actions:

1. Use deterministic request (`temperature=0`, bounded `max_tokens`).
2. Verify endpoint (`/v1/chat/completions` vs `/v1/completions`) matches model template.
3. Confirm non-empty output and HTTP 200 before success declaration.

## Graph mode accuracy degradation on A5 (MoE models)

Symptoms:

- GPQA Diamond score drops in graph mode vs eager mode (e.g., 0.7273 → 0.5707, ~15.7% drop).
- At `temperature=0.0`, graph mode systematically over-selects options C/D and under-selects B in 4-choice questions.
- 49% of wrong answers mention the correct option text in reasoning but pick a different letter in the ANSWER line — this is **logits-level bias**, not reasoning-depth loss.
- Occasional repetition/garbled output (2.5% of responses at temp=0.0, longest 13707 tokens).

Root cause chain:

```
Cudagraph padding (5 tokens → 8) → padding tokens' router_logits join softmax
  → npu_moe_init_routing internal permutation differs from eager
  → real tokens' expert computation results subtly shifted
  → logits at ANSWER token position biased: C/D ↑, B ↓
  → model systematically over-selects C/D, under-selects B
  → even when reasoning content mentions correct option, final letter is wrong
  → GPQA score drops from 0.7273 to 0.5707
```

**Critical evidence**: In GPQA Diamond evaluation (198 questions), prediction letter distribution shows B under-selected by 14 (predicted 41 vs actual 55) and C/D over-selected by 15/13. **48.8% of wrong answers (42/86) explicitly mention the correct option in their reasoning but select a different letter in the ANSWER line**. Wrong answers have MORE avg completion_tokens (1359.7 vs 1247.2) — the model is NOT producing shorter responses, it produces verbose but less structured output. This proves the degradation is **logits-level ANSWER letter bias**, not shallow reasoning. Full data: `/home/tongpan/outputs/20260611_200627/predictions/`, analysis summary: `/home/tongpan/result.txt`, full report: `docs/gemma4-graph-mode-accuracy-analysis.md`, repro guide: `/home/tongpan/gemma4-a5-ascend-repro-guide.md` section 4.6.

| File | Line | Issue |
|------|------|-------|
| `patch_cudagraph.py` | 15 | `num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]` — pads real tokens to capture sizes |
| `prepare_finalize.py` | 424-428 | ALLGATHER pads both `hidden_states` and `router_logits` — padding tokens produce routing weights |
| `prepare_finalize.py` | 514 | `hidden_states[:self.num_tokens]` — finalize slices but init_routing permutation already different |
| `ascend_forward_context.py` | 177-179 | `num_actual_tokens` defaults to padded `num_tokens` when not explicitly set |
| `experts_selector.py` | 240-246 | TODO: bfloat16 topk not supported in "ge graph" NPU execution mode |
| `moe_comm_method.py` | 195-201 | `npu_moe_finalize_routing` has known accuracy bug; workaround uses `npu_moe_token_unpermute` |

A5-specific constraints:

1. A5 attention declares `NEVER` for cudagraph support — decode runs eagerly within aclgraph wrapper (`acl_graph.py:157-164`), but `_EXTRA_CTX.capturing=True` changes code paths during capture.
2. MC2/ALLTOALL MoE comm crash on A5 (error 561000), forcing ALLGATHER — padding contamination cannot be mitigated via `mc2_mask` (mc2_mask is MC2-specific).
3. ALLGATHER path has no mechanism to exclude padding tokens from routing.

Experimental evidence (temperature=0.0, Gemma4-26B-A4B-it on A5):

| Test | Graph(8006) | Eager(8011) | Match? | Key Difference |
|------|-------------|-------------|--------|----------------|
| Maxwell equations MCQ | A (wrong), 448ct | D (correct), 919ct | **No** | Graph misses Faraday law change |
| Thermodynamics MCQ | D (wrong), 1189ct | B (correct), 521ct | **No** | Graph over-selects D |
| Cosmology MCQ | B (wrong), 518ct | C (correct), 385ct | **No** | Graph under-selects |
| Organic chemistry MCQ | garbled, 6553ct | B (correct), 587ct | **No** | Repetition output |
| Simple MCQs (5 questions) | All correct | All correct | Yes | No effect on simple questions |

Diagnosis steps:

1. **Compare same prompt at `temperature=0.0` on both modes** — if answers differ on multi-step questions, confirms logits-level ANSWER letter bias from MoE padding.
2. **Check prediction letter distribution** — if B is systematically under-selected and C/D over-selected vs gold distribution, confirms logits shift pattern.
3. **Check if wrong answers mention correct option in reasoning** — if >40% of wrong answers contain correct option text, confirms the issue is final letter selection, not reasoning depth.
4. **Log `router_logits` shape** in ALLGATHER prepare step — if `router_logits` is padded to capture_size instead of real token count, confirms padding contamination.

Mitigation options:

1. **Short-term**: Use `--enforce-eager` for MoE models on A5 until ALLGATHER padding treatment is fixed.
2. **Medium-term**: Zero out padding tokens' `router_logits` in ALLGATHER prepare before `npu_moe_init_routing` (`router_logits[num_tokens:, :] = 0.0`). Add `mc2_mask`-like mechanism to ALLGATHER path.
3. **Long-term**: Fix `npu_moe_finalize_routing` accuracy bug. Upgrade CANN/torch-npu so MC2 works on A5. Fix bfloat16 topk in NPU "ge graph" mode.

**Full analysis report**: See `docs/gemma4-graph-mode-accuracy-analysis.md`.
