---
name: v40 GRPO from base model result
description: v40 GRPO (50 steps from base model, no SFT) scored 0.52. Confirms GRPO cannot replace SFT — must build on SFT base.
type: feedback
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
v40 GRPO from base = **0.52** (vs baseline 0.48, vs SFT v38a 0.72)

**Key learning:** GRPO from untrained base gives only +0.04 over zero-shot. The model barely produces `\boxed{}` answers without SFT training, so reward signal is sparse. SFT is mandatory first.

**Why:** GRPO optimizes existing capability — if the model can't solve puzzles at all, there's nothing to reinforce. SFT teaches the format + reasoning patterns first.

**How to apply:** Always SFT first (0.48→0.72), then GRPO on top (0.72→?). Never skip SFT. v41 GRPO on v38a base is the correct approach.
