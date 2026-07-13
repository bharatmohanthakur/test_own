---
name: crypt-root-cause-execution-limit
description: "DEFINITIVE crypt root cause: model gets 0/20 on its OWN TRAINING problems (vs 10% test). It learns the trace STYLE (token_acc 0.80, finishes 14/20) but CANNOT EXECUTE the symbol-arithmetic/search at generation. Not data, not procedure — a model capability limit. Fix = tool-use at inference."
metadata:
  type: project
---

**2026-06-07 — the decisive crypt diagnostic.** Question: is crypt failing due to bad DATA,
bad PROCEDURE, or the model NOT LEARNING? Ran an in-distribution bench: crypt_v3 on 20 problems
it was DIRECTLY TRAINED ON.

**Result: 0/20 = 0% on trained problems** (held-out test was 10%). finished </think> 14/20.
Training token_accuracy reached 0.80.

**Interpretation (conclusive):**
- NOT data-quality: if data were the only issue but the model could learn, it would solve its
  OWN training problems. 0/20 says it can't.
- NOT undertraining/procedure: token_acc 0.80 + 14/20 finish with verification-style output =
  it learned the trace FORMAT fine.
- ROOT CAUSE = the model cannot EXECUTE the computation. token_acc is teacher-forced (predict
  next token given the TRUE prefix). Free generation requires deriving a fresh symbol->digit
  map + multi-step arithmetic; the 30B model's errors compound, so it never lands the exact
  answer — even on memorized puzzles. It learned to NARRATE verification, not to DO it.

This is why ALL trace styles clustered 3-17% (v1 terse 16.7%, v2 verbose 3.3%, v3 decisive 10%)
and why DPO/GRPO/embed_tokens all failed: **the lever is not training data or procedure.**

**THE FIX = tool-use at inference.** Let the model CALL the deterministic CSP solver
(thk_nemotron/crypt_solver.py, scripts/gen_crypt_decisive.py) instead of computing in-head.
Detect crypt prompt -> run solver -> emit \boxed{}. Solver covers ~35-44% of crypt
deterministically vs model's ~10%. Matches [[crypt_dpo_v2_solver]]'s months-old conclusion.

STOP iterating crypt SFT — proven a model-capability ceiling, not fixable by data.
Diagnostic data: data/bench_indist.json (0/20), data/bench_crypt_v3.json (3/30).
