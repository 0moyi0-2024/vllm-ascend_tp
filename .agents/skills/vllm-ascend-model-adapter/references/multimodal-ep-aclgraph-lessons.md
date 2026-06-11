# Multimodal + EP + ACLGraph Lessons

This note captures practical patterns that repeatedly matter for VL checkpoints on Ascend.

## 1) Out-of-box feature expectation

Try best to validate key features by default:

- ACLGraph
- MTP
- multimodal (if model supports VL)
- EP (MoE models only)
- flashcomm1 (MoE models only)

If any feature fails, keep logs and explain the reason in the final report.
For non-MoE models, EP/flashcomm1 should be marked not-applicable.

## 2) Validate in this order

1. Single text request success (`/v1/models` + `/v1/chat/completions`).
2. Single text+image request success.
3. Graph evidence (`Replaying aclgraph`) when graph mode is expected.
4. Capacity baseline: `128k + bs16`.
5. Concurrency expansion if needed (`32/64` suggested).

## 3) EP + graph startup expectations

- Startup latency is much higher than eager due to:
    - compile warmup
    - graph capture rounds
    - multimodal encoder profiling
- Do not treat slow startup as failure unless logs show hard errors.

## 4) Always distinguish two max lengths

- **Theoretical max**: from model config (`max_position_embeddings`).
- **Practical max**: largest value that actually starts and serves on current hardware + TP/EP settings.

Report both values explicitly.

## 5) Multimodal testing with temporary layer reduction

- Reducing `num_hidden_layers` can speed smoke tests.
- This does **not** remove ViT structure itself.
- Still require one full-layer validation before final sign-off.

## 6) Feature-status semantics

Use four categories:

- ✅ supported and verified
- ❌ framework-level unsupported
- ⚠️ checkpoint missing (weights/config do not provide feature)
- N/A not-applicable (for example EP/flashcomm1 on non-MoE models)

Typical examples:

- flashcomm1 on non-MoE VL models is often N/A or ❌ depending on framework gate.
- MTP may be ⚠️ checkpoint missing even if framework has code paths.

## 7) Keep docs and defaults aligned with latest success path

- If EP+graph is validated and requested/expected, it should be the default runbook path.
- Eager mode should be documented as fallback/troubleshooting only.

## 8) Graph mode accuracy risk on MoE models (A5) — logits-level ANSWER letter bias

For MoE models on Ascend950PR (A5), graph mode has a **confirmed accuracy degradation risk**:

- A5 forces ALLGATHER MoE communication (MC2/ALLTOALL crash with error 561000).
- ALLGATHER path pads `router_logits` alongside `hidden_states` — padding tokens produce softmax routing weights that contaminate `npu_moe_init_routing` permutation.
- This causes **logits-level bias** in the final ANSWER letter selection: model systematically over-selects C/D and under-selects B.
- **48.8% of wrong answers (42/86) mention correct option in reasoning but pick wrong letter** — NOT shallow reasoning, logits-level ANSWER letter bias
- Wrong answers avg completion_tokens 1359.7 > correct 1247.2 (model NOT producing shorter responses)

GPQA Diamond evidence (gemma-4-26B-A4B-it, 198 questions):
- Eager score: 0.7273, Graph score: 0.5707 (15.7% drop)
- Letter B predicted 41 times vs actual 55 (under-selected by 14)
- Letter C predicted 50 times vs actual 35 (over-selected by 15)
- Letter D predicted 49 times vs actual 36 (over-selected by 13)

**Always validate graph mode accuracy against eager mode baseline** before declaring graph mode as default:

1. Run same test set at `temperature=0.0` on both modes.
2. Compare answers on multi-step reasoning questions (GPQA-style).
3. Check prediction letter distribution — if B is systematically under-selected, confirms logits shift.
4. Check if wrong answers mention correct option text — if >40% do, confirms logits-level letter bias (not reasoning depth).
5. If graph mode score drops >5%, treat as accuracy regression and document the reason.
6. Consider keeping eager mode as production default until ALLGATHER padding treatment is fixed.

**Reference**: See `troubleshooting.md` → "Graph mode accuracy degradation on A5 (MoE models)", `/home/tongpan/result.txt`, `docs/gemma4-graph-mode-accuracy-analysis.md`, `/home/tongpan/gemma4-a5-ascend-repro-guide.md` section 4.6 for full root cause, experimental evidence, and mitigation options.
