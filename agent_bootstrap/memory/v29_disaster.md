---
name: v29 from-base disaster (0.22 local bench)
description: v29 scored 13/60 = 0.217 local (est 0.34 Kaggle). Root cause: thk_training_v3.jsonl has only 3.2% of examples with correct \boxed{} emit format. Model learned reasoning but not answer emission.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Date:** 2026-04-19 (training ran Apr 18-19)
**Local bench:** 13/60 = **0.217** (est Kaggle ~0.34 with +0.12 offset)
**v26 baseline bench:** 44/60 = 0.733 ↔ Kaggle 0.85 ✅

## Config
- Base: Nemotron-3-Nano-30B from scratch (not refining v26)
- Data: `thk_training_v3.jsonl` (16720 examples — THK's v3 with our 538 crypt corrections)
- Unsloth FastLanguageModel, r=32, alpha=32, dropout=0, 11 target modules incl `embed_tokens` + `lm_head`
- LR=2e-4, batch=2 × grad_accum=16 = eff 32, max_seq=8192, packing=True
- ~900 steps, 1 epoch, ~6 hours on RTX Pro 6000 Blackwell 102GB
- Final loss stuck at 8-10 range (never converged to normal 1-3)

## Bench per-type (v26 → v29)

| Type | v26 | v29 | Δ |
|---|---|---|---|
| binary | 9/10 | 1/10 | −8 |
| cipher | 10/10 | **0/10** | −10 |
| cryptarithm | 2/10 | 0/10 | −2 |
| gravity | 7/10 | **0/10** | −7 |
| roman | 10/10 | 8/10 | −2 |
| unit | 6/10 | 4/10 | −2 |
| **total** | 44/60 | **13/60** |

## Failure mode in outputs
- Cipher: model loops infinitely printing symbol mappings, hits max_tokens with no `\boxed{}`
- Gravity: math is computed but model outputs the rate constant k (9.224) instead of distance d. Doesn't understand what's asked.
- Binary: spews multi-column analysis, never produces a clean 8-bit answer
- Unit: math CORRECT but rounds off (23.23 vs 23.25 expected) → near-misses that fail exact match
- Roman: logic survives, boxed gets emitted (even if inside think, regex catches it)
- Cryptarithm: defaults to "concatenation" fallback when solver doesn't know the op

## Root cause — training data malformed for SFT
Quantified on `thk_training_v3.jsonl` (16720 total):
- 538 (3.2%) have `\boxed{ANSWER}` AFTER `</think>` — the CORRECT emit format
- 7719 (46%) have `\boxed{}` inside `<think>` only (or as placeholder)
- 8463 (50.6%) have NO `\boxed{}` at all

**97% of examples never teach the `</think>` → `\boxed{answer}` handoff.**

Only the 538 correct-format examples are our own regenerated crypt traces from `reasoning_cryptarithm_legacy`. THK's original 16182 are internal CoT dumps with no final emit — apparently THK's training framework (Tinker) handles the answer emission separately.

## Why v26 scored 0.85 and v28 scored 0.67 but v29 = 0.22
- **v26:** THK's original weights — format baked in during Tinker training
- **v28:** Refined v26 with same v3 data → format preserved from v26 weights, just learned new crypt traces (badly)
- **v29:** FROM BASE → had to learn format from scratch → 97% of data doesn't teach it → model produces CoT then stops

## Lessons (for future iterations)
1. **SFT data for this competition MUST have `\boxed{ANSWER}` AFTER `</think>`** — verify before training
2. **Don't train from base on CoT-only data** — base model has no format prior, 3% correct examples aren't enough signal
3. **Refining from v26 is safer** — format knowledge is preserved even if data is imperfect
4. **Loss 8-10 at epoch end is a red flag** — for this task normal is 1-3. Not "from-base inflation."
5. **ALWAYS bench before submit** — saved a wasted submission today

## Path forward
Option A (fastest): filter thk_training_v3 to only the 538 correct-format examples + refine v26 at LR=5e-5
Option B: post-process all 16720 examples to extract answer and append `\boxed{}` — requires answer inference from the trace
Option C: use different data source (Donald templates, v22 verbose, Grok teacher) with known-good format
