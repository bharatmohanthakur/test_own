---
name: v27 bit_manip refinement regression (0.82)
description: Refining THK v26 (0.85) on 100 bit_manip 3-input traces SCORED WORSE (0.82). Narrow data + lm_head fix hurt overall.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Date:** 2026-04-16
**Score:** **0.82** (vs THK v26 baseline = 0.85, regression -0.03)

## Config
- Base: THK v26 transformed adapter (loaded via PEFT with lm_head key rename fix)
- Data: 100 bit_manipulation 3-input traces from THK's public dataset
- Training: SFT, r=32, alpha=32, dropout=0, 9 target_modules incl lm_head
- LR=2e-5, batch=1×4=4, 1 epoch (25 steps), max_seq=2048
- GPU: Vast.ai B200 192GB
- Wandb: r3cg8ow3
- Loss: 2.93 → 1.32 (55% drop, strong learning signal)

## Why 0.82 (hypotheses)

### #1 — Narrow data + catastrophic forgetting (MOST LIKELY)
Training on ONLY 100 bit_manip examples with lm_head LoRA trainable shifted the OUTPUT DISTRIBUTION toward bit_manip patterns. The model may now do marginally better on bit_manip 3-input problems but worse on cipher/numeral/unit_conversion/gravity/equation. Without per-type local eval we can't confirm, but -0.03 overall suggests broad regression, not one-category gain.

### #2 — lm_head rename "fix" backfired
THK's original v26 has lm_head LoRA keys at `base_model.model.backbone.lm_head.*` which vLLM can't match to any module → lm_head LoRA silently ignored at inference → base model's lm_head used. Our v27 fix renamed keys to `base_model.model.lm_head.*` so vLLM APPLIES the LoRA at inference. This means:
- THK's original `backbone.lm_head` weights (trained on 16720 diverse traces) were loaded into the correctly-named slot
- Then we trained lm_head further on only 100 bit_manip examples at LR=2e-5 × 25 steps
- The delta from ONLY bit_manip pushed lm_head toward bit_manip token distributions
- vLLM now uses these tweaked weights, whereas v26 used untouched base lm_head

### #3 — 100 examples too small for LoRA refinement
Kh0a's 0.73 recipe uses 2450+ examples. THK's 0.85 uses 16720. Our 100 is <1% of theirs. With such narrow data, even low LR induces distribution drift.

## Lessons

1. **Never refine on <500 samples of a single category.** Minimum should be mixed-category data with proportional representation.
2. **ALWAYS run local eval before burning a submit slot.** We wasted 1/5 daily submissions learning what local eval would have shown in 30 min on H100.
3. **Don't "fix" adapter key paths without testing.** THK's `backbone.lm_head` mis-rename may have been a FEATURE (effectively disabling lm_head training), not a bug. Our "fix" enabled it, allowing our short bit_manip-only training to corrupt the distribution.
4. **THK's adapter IS carefully tuned.** Modifying it in ANY way without broad data + low LR + long training will regress. His 0.85 is a saddle point, not an easy base to build on.

## Score timeline
| Date | Adapter | Score |
|---|---|---|
| Apr 12 | v38a (our Kh0a) | 0.72 |
| Apr 13 | GRPO B200 | 0.47 |
| Apr 14 | Unsloth refine | 0.66 |
| Apr 14 | B200 full SFT | 0.26 |
| **Apr 16** | **THK v26 as-is** | **0.85** (best) |
| Apr 16 | v27 bit_manip refine | 0.82 |

## Submissions used today (UTC)
1. 07:21 — THK v26 → 0.85
2. 13:49 — v27 → 0.82

3 remaining before midnight UTC.

## Next move
- v26 stays our best (0.85). Submit it for final leaderboard position if needed.
- Do NOT try small-data refinement again without local eval validation.
- If going for 0.87+, must use 500+ mixed-category examples OR just stop at 0.85.
