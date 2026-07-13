---
name: crypt-v4-decomposed-result
description: "crypt_v4 (598 decomposed single-digit-arithmetic traces, 93% coverage, 7129 verified steps) = 0/30 bench, 0/20 in-dist. FINAL crypt verdict: comprehension-stage collapse — model garbles example parsing even on memorized problems. Crypt SFT permanently closed; 0.93 unreachable."
metadata:
  type: project
---

**2026-06-10.** The strongest possible crypt SFT data — 598 traces (93.1% problem coverage via
29-family brute-force solver), EVERY arithmetic op decomposed to single-digit steps with carries,
7,129 equation claims independently verified, 100% boxed==truth, bounded ~600 tok — trained
clean (final loss 0.23, lowest ever) and scored **0/30 held-out, 0/20 in-distribution**.

Failure anatomy (bench dump): 21/30 terminated with a WRONG answer, 8 didn't terminate, 1
extraction artifact. Brace-aware re-extraction recovers 0. Sample gens show the model reproduces
the decomposed FORM flawlessly but **garbles the comprehension stage** — misparses the worked
examples, conflates digits/symbols, then executes perfect single-digit arithmetic on wrong inputs.

**Five-style scoreboard (all fresh-from-base r=32 LoRA, temp=0):**
terse 727 = 16.7% | verbose-verify 309 = 3.3% | decisive 185 = 10% | decomposed 598 = 0% |
in-dist ≈ 0/20 every time measured. v1's 16.7% = pattern-luck on easy cases, not derivation.

**Closed conclusion:** Nemotron-3-Nano-30B + rank≤32 LoRA at temp=0 cannot learn to execute
novel symbol-mapping derivation. Not data quality, not style, not length, not procedure —
capability. STOP all crypt SFT permanently. Tool-use is banned by comp rules → crypt stays
~0.2-0.35 for everyone; 0.93 overall is unreachable (needs crypt ≈0.75).

**Strategy after this:** protect v26 (0.85), squeeze non-crypt (THK weak spots: bit_manip,
eq_guess) via careful refine-from-v26 at LR ≤2e-5 → realistic ceiling 0.86-0.88 (would top
leaderboard). NOTE: v26 adapter = Unsloth fused-experts format (experts.w1/w2/w3) → vLLM LoRA
loader rejects it; bench v26 via HF+PEFT (bench_specialist.py); Kaggle eval accepts it fine.
See [[crypt-root-cause-execution-limit]], [[v27_regression]] [[v28_regression]] (why refines
regress: LR too high — stay ≤2e-5, small steps, replay mix).
