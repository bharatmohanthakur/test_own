---
name: crypt-v2-sonnet-overthink
description: "crypt_v2 (309 pure Sonnet verification traces, fresh-base, 3ep) = 1/30 (3.3%), WORSE than v1's 16.7%. Failure FLIPPED: model never terminates — 27/30 hit the 7680 token ceiling with no </think>/no boxed."
metadata: 
  node_type: memory
  type: project
  originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---

**2026-06-06/07.** Trained crypt_v2 on 309 PURE Claude-Sonnet-4.6 verification traces
(hardest-sorted, each verifies every candidate rule vs all examples + self-corrects;
verified boxed==truth). Fresh-from-base r=32, batch 2 × accum 16, max_seq 7680, 3 epochs,
LR 2e-4. train_loss 1.065.

**Bench (vLLM, temp=0, 7680 tok, n=30): 1/30 = 3.3% — WORSE than crypt v1's 5/30 (16.7%).**

**Failure mode FLIPPED — the key finding:**
- v1 (solver traces): finished fast (~965 tok), committed to a WRONG answer (no verification).
- v2 (Sonnet verification): **27/30 never emitted </think> or a boxed answer**; avg 7193 gen
  tokens, max 7680 — it hits the generation ceiling. The model learned the verification STYLE
  (verify/backtrack/self-correct) but **never learned to TERMINATE** — it loops forever.

**Why:** the deleted solver traces were supplying the *decisive-termination* signal. Pure
verification traces = endless exploration with no commit. Training data style → inference style.

**How to apply (next):** MIX is required, not pure-Sonnet. Combine 727 solver traces (teach
"recover rule → apply → STOP") with the ~309 Sonnet verification traces (teach "verify before
commit"). The solver traces bound the reasoning length; the verification traces add the missing
check-against-all-examples skill. Train on the mix, re-bench. Also consider: verification traces
that END DECISIVELY (after one verification pass, commit — no open-ended backtracking).

Pattern across attempts: v1 too-hasty (16.7%), v2 too-verbose (3.3%) → the answer is in the
middle (verify once, then commit). See [[crypt_specialist_v46_diagnosis]] (v1 diagnosis),
[[v40_pathA_result]], [[crypt_dpo_v2_solver]]. Budget: OpenRouter key2 ~$80+ remains.
