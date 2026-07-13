---
name: B200 SFT disaster (0.26)
description: B200 full SFT on THK's 16720 data with Kh0a config scored 0.26 — worst ever. Config mismatch killed it.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Date:** 2026-04-14
**Score:** 0.26 (worst submission ever, baseline untrained = 0.48)
**Adapter:** bharatmohan/nemotron-sft-thk-b200

## What we trained
- **GPU:** Vast.ai B200 192GB, ~90 min training
- **Data:** THK's thk_training_v2.jsonl (16720 examples: 9500 reasoning + 8463 augmentation + boxed)
- **LoRA (Kh0a):** r=32, alpha=16, dropout=0.1, 9 targets (q/k/v/o/in/out/up/down/gate_proj)
- **NO lm_head target** ← suspected cause #1
- **Train:** batch=8, grad_accum=1, 1 epoch = 2090 steps, LR=1e-4, max_seq=4096, packing=True, train_on_responses_only
- **Logging:** NONE (nohup + report_to="none") — ran blind

## Score progression
| Run | Config | Score |
|---|---|---|
| Untrained baseline | — | 0.48 |
| v38a SFT (our best) | Kh0a on 2450 Kh0a data | **0.72** |
| GRPO B200 | beta=0 + low reward | 0.47 |
| THK adapter refinement | Unsloth overwrote THK weights | 0.66 |
| **B200 full SFT (this)** | **Kh0a on THK 16720** | **0.26** |
| THK public adapter alone | — | 0.85 |

## Why it scored 0.26 (hypotheses)

### #1 — Config mismatch with THK's data
THK's adapter was trained with:
- alpha=32 (we used 16 → weaker LoRA signal)
- train_unembed=True (lm_head target — we DIDN'T train lm_head)
- max_length=8192 (we used 4096 — truncated reasoning!)
- LR=2e-4 (we used 1e-4)
- batch_size=64 (we used 8)

THK's data contains long CoT traces that EXPECT lm_head training. Without it, the model can't produce the right output distribution for the structured answers.

### #2 — max_seq=4096 truncated responses
THK traces for bit_manipulation/cryptarithm are 3000-7000 tokens. Packing=True + 4096 max_seq means long traces got chopped, training on partial reasoning.

### #3 — train_on_responses_only masked wrong tokens
With Unsloth's response-only masking + packing, boundaries can get fuzzy. Loss may have been computed on wrong tokens.

### #4 — Ran BLIND
No wandb, no loss visibility. Could have been diverging from step 100 and we'd never know. **See memory/logging_policy.md** — NEVER AGAIN.

## Lessons (burn into memory)

1. **When refining someone else's data, match THEIR config exactly.** THK trained with alpha=32 + lm_head + seq=8192. We didn't. Catastrophic.
2. **More data ≠ better.** 16720 examples with WRONG config scored 0.26. 2450 examples with RIGHT config (v38a Kh0a) scored 0.72.
3. **wandb is mandatory.** Running blind cost us $8 + a submission slot.
4. **Local eval BEFORE submit is mandatory.** If we'd run 60 tracking prompts, we'd have seen 0.20 locally and NOT submitted.
5. **Check the teacher's config before using their data.** THK posted his config publicly — read it.

## Next move
- Don't retry B200 with wrong config.
- Either: (a) use Tinker with THK's exact config (needs credits), or (b) use THK's adapter AS-IS (already 0.85) and only add tiny targeted refinement.
- If refining THK: LR ≤ 1e-6, ≤50 steps, match alpha=32 + lm_head, local eval EVERY 10 steps.

## Submissions used today (2026-04-14)
1. 06:25 — THK Unsloth refinement → 0.66 ❌
2. 09:34 — B200 full SFT → **0.26** ❌

2/5 burned with no improvement. 3 left before midnight UTC.
