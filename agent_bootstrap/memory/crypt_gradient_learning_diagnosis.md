---
name: crypt-gradient-learning-diagnosis
description: "2026-06-11 MEASURED root cause of loss-down-but-no-learning: teacher-forced argmax probe on crypt_v9 shows 89-91% token match = ALL scaffold; first mismatch in EVERY row is the first puzzle-specific decision token (pos ~30, the atom-binding). Loss 0.136 = 10% of tokens at ~1.4 NLL. lm_head-zeroed delta only 0.005 → vLLM serving NOT the issue. Facts seen 1-2x can't enter weights; leaders won via repetitions-per-fact."
metadata: 
  node_type: memory
  type: project
  originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---

**The question** (user): loss decreases (0.136, deepest ever) but the model learns nothing — why?

**The measurement** (`scripts/probe_teacher_forced.py`, HF forward pass, no generate()):
- argmax match on trained traces: 0.890–0.914 across 6 rows
- the ~10% mismatches are not random: **in every row the first miss is the first
  puzzle-specific token** — the atom-binding table entry (`atoms: 【` → want `]` got `` ` ``,
  want `&` got `` ` ``, want `>` got `` ` ``...). All scaffold/boilerplate/format tokens: perfect.
- loss arithmetic checks out: 90% tokens ≈ 0 NLL + 10% at ~1.4 NLL ≈ 0.136 average.
- with lm_head LoRA zeroed, match drops only ~0.005 → the lm_head module contributed nothing;
  vLLM possibly dropping it is irrelevant. Serving is NOT the bug ([[prompt-format-bug]] was
  real but secondary — exact-format in-dist replay still 0/13).

**The mechanism.** SFT loss is a per-token average. Scaffold tokens (deterministic given
context) are learned in a few steps and drive loss down; decision tokens (which symbol of
THIS puzzle binds where) are seen 1–2 times each (515 traces / 432 mul-patterns / 76 steps)
— far too few repetitions for a fact to enter r=32 LoRA weights. Teacher forcing corrects
every mistake, so loss never reveals it; greedy generation derails at the FIRST decision
token and cascades into loops/overflow (the 4x verbosity drift).

**Why the leaders won:** "crypt dominated corpus token share" = MANY traces per pattern =
many repetitions per fact. Scale isn't diversity — it's REPETITIONS PER FACT.

**How to apply:**
1. Watch decision-token accuracy, not mean loss. The probe script generalizes to any run.
2. If reattempting crypt: generate 10–20 traces PER mul-pattern (5–10k total, solver is free),
   crypt-dominant, 2-3 epochs — each table fact then repeats 10-20x. ~$9 of B200.
3. Per-token loss masks would help: upweight decision tokens / downweight scaffold
   (token-priority SFT in [[training-methods-sota]]).

---
**2026-06-11 v11 RESULT (minimal-recall traces, 2,200 rows, 2 epochs, 138 steps).**
Behavior FIXED: holdout 24/24 finished at avg 187 tok (= training mean; overflow drift
eliminated by short traces + correct format). Copy/derive classes now excellent:
fact-translit 0.976 (was 0.754), binding 0.914, recall-struct 0.986, scaffold 0.870.
Score still 0/24 + 0/12 in-dist. Failure isolated to ONE class: **recall-digits 0.714
with near-uniform posterior** (P(want)≈0.10, top5 all ≈0.13) — pure weight-recall of
arbitrary key→value digit facts did NOT store. Repetition 20x in-corpus: 0.692 (no help
at 138 total optimizer steps). v9's 0.876 on this class was inflated by within-trace
list correlations (sorted candidate lists), not true recall.
Math: 0.714^17 ≈ 0.3%/puzzle → 0/24 expected. Per-token need at n=17: 0.96.
**Remaining variable: optimization exposure** — arbitrary-association memorization needs
hundreds of gradient exposures per fact (10-20 epochs or upweighted fact-drill rows),
~$8-10 GPU. Everything else in the chain is now verified working.
