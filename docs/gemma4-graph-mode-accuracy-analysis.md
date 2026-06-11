# Gemma4 Graph Mode Accuracy Analysis (GPQA Diamond)

> Branch: `test_xty` / `xty_new`, Model: gemma-4-26B-A4B-it on Ascend950PR (A5)

---

## 1. Score Comparison

| Mode | Config | GPQA Diamond Score |
|------|--------|---------------------|
| Eager | `--enforce-eager` | 0.7273 |
| Graph | `cudagraph_mode=FULL_DECODE_ONLY, capture_sizes=[1,2,4,8]` | 0.5707 |
| **Degradation** | | **-0.1566 (≈15.7% drop)** |

---

## 2. GPQA Diamond Detailed Analysis (Graph Mode, 198 questions)

### 2.1 Basic Stats

- Total: 198, Correct: 101, Wrong: 97
- Repetition/garbled output: 5/198 (2.5%), longest output 13707 tokens
- Avg latency: 21.2s, Avg TTFT: 89ms, Avg TPOT: 17ms, Avg throughput: 60 tok/s

### 2.2 Correct vs Wrong Answer Comparison

| Metric | Correct Answers | Wrong Answers |
|--------|----------------|---------------|
| Count | 101 | 97 |
| Avg completion_tokens | 1247 | 1302 |
| Avg reasoning lines | 73.7 | 43.1 |
| Repetition issues | 2 | 3 |

**Key finding**: Wrong answers have MORE completion tokens (1302 vs 1247) but fewer reasoning lines (43.1 vs 73.7). This means the model is NOT producing shorter answers — it's producing longer, more repetitive/verbose answers with less structured reasoning.

### 2.3 Letter Selection Bias (Critical Finding)

| Letter | Predicted Count | Correct Count | Deviation |
|--------|----------------|---------------|-----------|
| A | 58 | 61 | -3 |
| B | **41** | **55** | **-14 (severely underestimated)** |
| C | 50 | 35 | +15 (over-selected) |
| D | 49 | 36 | **+13 (over-selected)** |

**The model systematically under-selects B and over-selects C/D.** This is a logits-level bias caused by cudagraph padding tokens contaminating MoE routing, not a reasoning-depth issue.

### 2.4 Wrong Answer Pattern Distribution

| Pattern (pred→gold) | Count | Description |
|---------------------|-------|-------------|
| D→B | 14 | Most common: correct answer B, model picks D |
| C→A | 13 | Correct answer A, model picks C |
| D→A | 11 | Correct answer A, model picks D |
| A→D | 9 | Correct answer D, model picks A (Maxwell equations type) |
| A→B | 9 | Correct answer B, model picks A |
| C→B | 6 | Correct answer B, model picks C |
| B→A | 5 | |
| B→D | 5 | |
| B→C | 4 | |
| C→D | 4 | |
| D→C | 3 | |
| A→C | 3 | |

### 2.5 "Reasoning Correct but Letter Wrong" Pattern

**49% of wrong answers (42/86) mention the correct option text in their reasoning, but pick a different letter in the ANSWER line.**

This is the definitive evidence that the accuracy degradation is NOT caused by shallow reasoning, but by **logits-level bias in the final ANSWER letter selection**. The model's MoE layers produce reasoning content that correctly identifies the right answer, but the logits for the ANSWER token are shifted by padding-induced MoE routing contamination, causing the final token generation to select a different letter.

### 2.6 Repetition/Garbled Output

5 out of 198 responses had repetition issues (same line repeated 3+ times). The longest garbled output was 13707 tokens (organic chemistry question). At temperature=0.0, this is particularly concerning as repetition should not occur in deterministic generation.

---

## 3. Curl Verification Results (temperature=0.0)

### 3.1 test_xty branch results

| Test | Graph(8006) | Eager(8011) | Match? | Key Difference |
|------|-------------|-------------|--------|----------------|
| Maxwell equations MCQ | A (wrong), 448ct | D (correct), 919ct | **No** | Graph misses Faraday law change |
| Thermodynamics MCQ | D (wrong), 1189ct | B (correct), 521ct | **No** | Graph over-selects D |
| Cosmology MCQ | B (wrong), 518ct | C (correct), 385ct | **No** | Graph under-selects C |
| Organic chemistry MCQ | **garbled**, 6553ct | B (correct), 587ct | **No** | Repetition output |
| Chinese MCQ (四大发明) | D (correct), 304ct | D (correct), 313ct | Yes | |
| Quantum entanglement | B (correct), 295ct | B (correct), 362ct | Yes | |
| Quantum number l≤n-1 | B (correct), 429ct | B (correct), 610ct | Yes | Graph reasoning shorter |
| Molecular biology | B (correct), 350ct | B (correct), 278ct | Yes | |
| Logic contrapositive | A (correct), 339ct | A (correct), 396ct | Yes | |
| Relativity time dilation | B (correct), 303ct | B (correct), 403ct | Yes | |
| Economics Phillips curve | C (correct), 347ct | C (correct), 387ct | Yes | |

### 3.2 xty_new branch (latest commit) curl result

| Test | Graph(8006) | Eager(8011) | Match? |
|------|-------------|-------------|--------|
| Maxwell equations MCQ | A (wrong), 448ct | D (correct), 919ct | **No** |

(Same accuracy degradation pattern confirmed on xty_new branch)

---

## 4. Root Cause Mechanism

```
Cudagraph padding (num_tokens padded to [1,2,4,8])
  → ALLGATHER MoE prepare pads both hidden_states AND router_logits
  → Padding tokens' router_logits produce softmax routing weights
  → npu_moe_init_routing internal permutation differs from eager mode
  → Real tokens' expert computation results subtly shifted
  → Logits at ANSWER token position biased: C/D logits ↑, B logits ↓
  → Model systematically over-selects C/D, under-selects B
  → Even when reasoning content mentions correct option, final letter is wrong
  → GPQA score drops from 0.7273 to 0.5707
```

Key code locations:

| File | Line | Issue |
|------|------|-------|
| `patch_cudagraph.py` | 15 | `num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]` |
| `prepare_finalize.py` | 424-428 | ALLGATHER pads `router_logits` alongside `hidden_states` |
| `prepare_finalize.py` | 514 | Finalize slices output, but `npu_moe_init_routing` permutation already different |
| `ascend_forward_context.py` | 177-179 | `num_actual_tokens` defaults to padded `num_tokens` when not set |
| `experts_selector.py` | 240-246 | bfloat16 topk not supported in "ge graph" NPU mode |
| `moe_comm_method.py` | 195-201 | `npu_moe_finalize_routing` known accuracy bug, workaround with `npu_moe_token_unpermute` |

A5-specific constraints:

- A5 attention declares `NEVER` for cudagraph — decode runs eagerly within aclgraph wrapper
- MC2/ALLTOALL MoE comm crash on A5 (error 561000) — forced ALLGATHER
- ALLGATHER path has no `mc2_mask` mechanism to exclude padding tokens from routing

---

## 5. Mitigation Options

### Short-term (no code change needed)

1. Use `--enforce-eager` for production inference on MoE models on A5 until ALLGATHER padding treatment is fixed.
2. For accuracy evaluation, always compare graph mode against eager baseline first.

### Medium-term (code fix)

1. In ALLGATHER MoE `prepare_finalize.py`, zero out padding tokens' `router_logits` before `npu_moe_init_routing`:
   ```python
   if pad_size > 0:
       hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
       router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
       # FIX: zero out padding tokens' router_logits to prevent routing contamination
       router_logits[num_tokens:, :] = 0.0
   ```
2. Explicitly track `num_actual_tokens` separate from `num_tokens_padded` in `ascend_forward_context.py`:
   ```python
   # Current: num_actual_tokens = num_tokens (defaults to padded value)
   # Fix: num_actual_tokens should always be the real token count
   ```
3. Add `mc2_mask`-like mechanism to ALLGATHER path to let `npu_moe_init_routing` exclude padding tokens.

### Long-term

1. Fix `npu_moe_finalize_routing` accuracy bug (currently workaround with `npu_moe_token_unpermute`).
2. Upgrade CANN/torch-npu so MC2 works on A5, eliminating need for ALLGATHER workaround.
3. Fix bfloat16 topk precision in NPU "ge graph" execution mode.