---
name: DPO results 2026-04-20
description: DPO on v26 samples (step100 = 0.84 Kaggle, regressed) and Crypt-DPO v1 (ckpt25 = 43/60 local, rejected). Key: crypt failure is structural, can't be moved by sample-based DPO.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Two DPO experiments, both failed to beat v26 (0.85).**

## 1. DPO from v26 on v26-sampled positives (step 100)
- Init: v26, LR 5e-6, beta 0.05, 100 steps, 107 pairs from rejection-sampling v26 at temp=1.0
- Local bench: 45/60 (+2 vs v26 43/60)
- **Kaggle: 0.84** (REGRESSED -0.01 from v26 = 0.85)
- The +2 local gain was noise — reran bench shows v26 = step100 = 44/60 (identical per-prompt)

## 2. Crypt-DPO v1 (targeted crypt-only)
- Init: step100, LR 2e-6, 50 steps planned (crashed at ckpt25 from disk full)
- Pairs: 356 total = 251 oracle-short + 5 sampled-positive + 100 preserve
- Sampling: 256 wrong crypt × 32 samples = **only 5/256 (2%) prompts had ≥1 correct**
- **Local ckpt25 = 43/60** (regressed -1 from v26 44)
- Crypt stayed at 2/10 — per-prompt outputs IDENTICAL on all 10 crypt test prompts

**Why:**
- Crypt failure mode is deterministic "concat-default on unknown operator"
- Same 10 crypt bench prompts produce same wrong outputs on v26, step100, ckpt25
- 5 positive pairs can't reliably teach new crypt behavior
- DPO on mostly-oracle pairs teaches format compliance, not new reasoning

**How to apply:**
- Do not retry sample-based DPO on crypt — the positive signal is too thin (2% @ temp=1)
- Crypt requires a structural fix: external CSP solver, candidate banks, or SFT on corrected traces
- Local bench has ~1-point noise — don't trust a single +1 or +2 delta as signal
