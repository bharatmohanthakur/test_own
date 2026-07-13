---
name: crypt-specialist-v46-diagnosis
description: "Crypt-only SFT specialist (727 solver traces, fresh r=32 LoRA) = 5/30 (16.7%) at 7680 tok — at ceiling. Root cause (Sonnet-4.6 on 25 fails): model never VERIFIES hypotheses against all examples."
metadata: 
  node_type: memory
  type: project
  originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---

**2026-06-06.** Trained a fresh-from-base crypt-ONLY LoRA specialist (r=32 α=32, explicit
linear targets, NO embed_tokens/lm_head, 727 solver traces, 3 epochs, max_seq 7680) on a
B200. Final train_loss 0.52, token_acc 0.94 (memorized trace format well).

**Bench: 5/30 = 16.7%** on held-out crypt (vLLM, temp=0, max_tokens=7680, seed=42). This is
AT the existing ceiling (v26 crypt ≈ 2/10 = 20%). A dedicated specialist did NOT crack crypt.

**Not a length/format problem:** all 25 failures finished `</think>` and emitted a boxed
answer; avg 965 gen tokens (max 1806) — nowhere near the 7680 budget. The model produces a
COMPLETE but WRONG deduction. Length was never the bottleneck here.

**Root cause (DeepSeek-V4-Pro too slow → switched to Claude Sonnet 4.6, 8-way parallel, 36s):**
failure-mode histogram over 25 fails:
- 11 (44%) wrong_operator — misreads what each operator symbol does
- 8 (32%) wrong_symbol_mapping — invents an inconsistent symbol→digit map
- 3 misread_problem, 2 arithmetic_error, 1 other
→ **76% are hypothesis errors.** Unanimous recommendation: the model **commits to a rule from
one example and never verifies it against the others.** Our 727 traces show the ANSWER
(recover mapping → apply) but NOT the verification/search process.

**Why:** SFT on terse solver-dump traces teaches the output shape, not robust deduction. The
missing capability = explicit hypothesis-verification: enumerate all examples as a table,
propose each candidate operator/mapping, CHECK it against EVERY example, reject on any
mismatch, confirm final mapping on all examples, then apply. These loops are long (justifies
7680 budget).

**How to apply (next step):** regenerate the 727 crypt traces via OpenRouter (Sonnet/GLM/
DeepSeek, ~$9.78 budget) with verification-heavy reasoning, verify each `\boxed{}` against
ground truth (>=95%), retrain crypt-only, re-bench. Do NOT just add MORE short traces or just
raise max_tokens — the gap is reasoning STRUCTURE (verify-before-commit), not quantity/length.
Pipeline built: `scripts/bench_vllm.py` (dumps full gens), `scripts/analyze_failures_llm.py`
(LLM failure diagnosis). See [[v40_pathA_result]] (embed_tokens also failed crypt),
[[crypt_dpo_v2_solver]] (DPO failed crypt too). Pattern: crypt is search, not pattern-match.
