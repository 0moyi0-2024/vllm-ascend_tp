# [Feature] Adapt Gemma4 31B EAGLE3 speculative decoding on Ascend

## Summary

Gemma4 31B + EAGLE3 spec decoding fails to start on vllm-ascend with `AttributeError: 'Gemma4Config' object has no attribute 'image_token_index'`. This issue tracks the full adaptation: the required fix, graph-mode selection, correctness verification, acceptance, and the performance ceiling + CANN dependencies to reach paper-level speedup.

Upstream vLLM already supports Gemma4 EAGLE3; vLLM-Ascend has the generic EAGLE3 framework. This is an adaptation task, not a new algorithm.

## Environment

- Ascend 910B3 (8 × 64GB HBM), CANN, torch-npu 2.10.0, vllm 0.23.0
- Target: `/home/xty/gemma4/31B` (Gemma4ForConditionalGeneration, bf16, dense, 512-dim global attention heads)
- Draft: Eagle3DraftModel (`eagle_aux_hidden_state_layer_ids: [2, 30, 57]`, `draft_vocab_size: 32000`)
- Target TP=4, Draft TP=1, BF16, text-only, low concurrency, short context

## Problem 1 — Draft loader AttributeError (code fix needed)

### Error
```
File ".../vllm_ascend/spec_decode/llm_base_proposer.py", line 374, in load_model
    self.model.config.image_token_index = model.config.image_token_index
AttributeError: 'Gemma4Config' object has no attribute 'image_token_index'
```

### Root cause
`AscendSpecDecodeBaseProposer.load_model` sets the draft's `image_token_index` for multimodal targets by matching arch name. Gemma4 is multimodal but uses `image_token_id` (not `image_token_index`), and `Gemma4ForConditionalGeneration` is not in the matched list → falls to the `else` branch → `AttributeError`.

**Upstream vLLM already fixed this** (`vllm/v1/spec_decode/llm_base_proposer.py:1371` includes `Gemma4ForConditionalGeneration`/`Gemma4UnifiedForConditionalGeneration`). vLLM-Ascend's version is stale.

### Fix (Mod 1, +5 lines, align with upstream)
```diff
--- a/vllm_ascend/spec_decode/llm_base_proposer.py
+++ b/vllm_ascend/spec_decode/llm_base_proposer.py
@@ -364,6 +364,11 @@ class AscendSpecDecodeBaseProposer(SpecDecodeBaseProposer):
                 "Qwen3_5ForConditionalGeneration",
                 "Qwen3_5MoeForConditionalGeneration",
                 "Step3p7ForConditionalGeneration",
+                # Align with upstream vLLM: Gemma4 VL models use image_token_id
+                # (not image_token_index), so map it explicitly here. Without
+                # this the else-branch raises AttributeError on Gemma4Config.
+                "Gemma4ForConditionalGeneration",
+                "Gemma4UnifiedForConditionalGeneration",
             ]:
                 self.model.config.image_token_index = model.config.image_token_id
```

## Problem 2 — Graph mode: FULL_DECODE_ONLY blocked by 512-dim TND

Gemma4 has heterogeneous attention heads (`head_dim=256`, `global_head_dim=512`). `using_paged_attention()` forces 512-dim heads to PagedAttention (PA) because FIA-TND doesn't support 512-dim. But under spec decoding, `using_paged_attention` returns `False` (spec guard) → 512-dim heads go to FIA.

- **FULL_DECODE_ONLY + spec**: FIA TND layout → `aclnnFusedInferAttentionScoreV3` error 561002: *"input_layout is TND, only headDim = 64/128/192 supported, but got 512"*. **Blocked.**
- **PIECEWISE + spec**: FIA uses BNSD layout (supports 512-dim) → captures + runs correctly. **Works.**

Forcing PA under spec (moving the 512-dim fallback before the spec guard) lets FULL_DECODE_ONLY capture, but produces **garbage output** (PA is decode-only, incompatible with spec's multi-token target verify; PA graph replay doesn't handle spec-decode attention metadata). Reverted.

**Recommendation**: Gemma4 EAGLE3 graph mode = **PIECEWISE**. FULL_DECODE_ONLY requires CANN FIA TND to support 512-dim (see Problem 4).

## Correctness verification

With Mod 1 + PIECEWISE:
- **Greedy token consistency**: spec-on (EAGLE3+PIECEWISE) vs spec-off (Target-only+PIECEWISE), 4 prompts → **ALL MATCH** (Paris / 1-10 / ML explanation / multi-language hello).
- **Stability**: 5 rounds × 4 prompts = 20 consecutive requests, greedy output identical → **PASS**.
- **Launch command**:
```bash
vllm serve <gemma4-31b> --tensor-parallel-size 4 --max-num-seqs 8 \
  --max-model-len 8192 --block-size 128 \
  --speculative-config '{"model":"<eagle3-draft>","method":"eagle3","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
```
(`--max-model-len 8192`: default 256k OOMs at profiling; `--block-size 128`: Ascend backends only support [128], default 16 → "No common block size")

## Acceptance rate

`num_speculative_tokens=3`, natural output (128 tok), steady-state:

| pos0 | pos1 | pos2 | mean acceptance length | avg draft acceptance |
|------|------|------|------------------------|----------------------|
| ~0.71 | ~0.48 | ~0.30 | ~2.5 / 3 | ~50% |

Healthy EAGLE3 decay pattern; draft quality is normal.

## Performance — current ceiling + bottleneck

### Speedup curve (spec-on vs spec-off, same PIECEWISE, TP4, natural 128-token output)

| parallel | spec-on (tok/s) | spec-off (tok/s) | speedup |
|----------|-----------------|------------------|---------|
| 1 | 28.1 | 24.6 | **+14.2% (peak)** |
| 2 | 51.0 | 48.3 | +5.6% |
| 4 | 91.6 | 96.0 | ~flat |
| 8 | 188.1 | 186.0 | ~flat |
| 16 | 177.6 | 360.5 | **-50.8%** |
| 32 | 264.3 | 519.9 | -49.2% |

### Bottleneck (instrumented)

| Component | Cost | Note |
|-----------|------|------|
| draft forward | **5.7 ms/step** | negligible, NOT the bottleneck |
| target spec-verify (FIA BNSD, 512-dim, 4 tok, graph) | **125 ms** | real bottleneck |
| non-spec PA decode (1 tok) | 40.7 ms | baseline |

- verify 4-token = **3.07× PA 1-token**: FIA 512-dim attention scales per-token; at low concurrency NPU is overhead-bound, 4-token verify does NOT amortize to 1-token cost.
- High concurrency (p16+): target saturated, 4× verify work not offset by acceptance 3.5 → -50%.
- Tried (no help): `--async-scheduling` (draft still serial), draft TP4 (small draft, comm overhead), `num_spec=2` (worse), `parallel_drafting` (eagle3 unsupported).

### Why not paper-level 3×

Paper (GPU): 4-token verify amortizes to ~1-token (compute-bound) → spec-step ≈ 46 ms → 75 tok/s → ~3×.
NPU: 4-token verify = 125 ms (not amortized) → +14%.

## Problem 3 (CANN dependency) — Path to paper-level 3×

Two CANN-side requirements to reach paper-level:

1. **FIA TND support for head_dim=512** — currently TND only supports 64/128/192 (`561002`). This would:
   - Unblock FULL_DECODE_ONLY for gemma4 EAGLE3 (remove PIECEWISE workaround).
   - Enable TND layout (more efficient than BNSD for 512-dim) → reduce verify cost.

2. **FIA 512-dim multi-token amortization** — the 4-token spec-verify should approach 1-token cost (like GPU compute-bound). Currently 4-token = 3× 1-token (per-token FIA 512-dim attention ~21 ms does not amortize at low concurrency).

Both are CANN/operator-level, not solvable in vLLM-Ascend Python.

## Proposed changes for vLLM-Ascend

| Change | Type | Upstream? |
|--------|------|-----------|
| **Mod 1**: add Gemma4 archs to `image_token_index` list | Code (+5 lines, align upstream vLLM) | ✅ Yes — clean, minimal, general |
| PIECEWISE config recommendation for gemma4 EAGLE3 | Docs | ✅ Yes |
| adaptive spec / draft overlap | — | ❌ No — CANN workaround, Ascend-specific, low benefit, becomes obsolete with CANN fix |
| CANN: TND-512 + multi-token amortization | CANN request | Track as CANN dependency |

## Checklist

- [ ] Merge Mod 1 (image_token_index fix) — unblocks Gemma4 EAGLE3 startup
- [ ] Document PIECEWISE as the graph mode for Gemma4 EAGLE3 (FULL_DECODE_ONLY needs CANN TND-512)
- [ ] File CANN request: FIA TND support head_dim=512 + multi-token amortization
- [ ] (Future, after CANN) Re-evaluate FULL_DECODE_ONLY + perf vs paper 3×

## References

- Upstream vLLM fix: `vllm/v1/spec_decode/llm_base_proposer.py:1371` (Gemma4 in image_token_index list)
- Detailed analysis + profiling evidence: branch `gemma4-eagle3-graph` in fork `0moyi0-2024/vllm-ascend_tp`, dir `gemma4_perf_artifacts/eagle3_adapt_phase1_20260727/`
