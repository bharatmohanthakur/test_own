---
name: BOXED_LOSS_WEIGHT hurts score
description: v42 (same data as v38a + BOXED_LOSS_WEIGHT=5.0) scored 0.70 vs v38a's 0.72. The 5x answer weighting hurts by 0.02.
type: feedback
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
v42 = v38a data + BOXED_LOSS_WEIGHT=5.0 → **0.70** (down 0.02 from v38a's 0.72)

**Why it hurts:** The 5x upweighting on tokens after `\boxed{` causes the model to over-optimize for answer format at the expense of reasoning quality. The model learns to output `\boxed{}` quickly but with worse answers.

**How to apply:** Do NOT use BOXED_LOSS_WEIGHT. Standard uniform loss works better for this competition. Kh0a's 0.73 used it but their data was 3910 (vs our 2681) — maybe larger data absorbs the format bias better.
