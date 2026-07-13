---
name: GRPO B200 result
description: GRPO B200 2-phase scored 0.47 — WORSE than 0.72 base. GRPO with low reward signal degrades the model. Don't submit GRPO without local validation.
type: feedback
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
## GRPO B200 Result: 0.47 (Apr 13, 2026)

**Base: v38a SFT = 0.72. After GRPO = 0.47. MASSIVE REGRESSION.**

### What was done
- Phase 1: 100 steps, all 500 prompts, gen=4, 1024 completion, LR=5e-6
- Phase 2: 50 steps, 200 hard prompts (binary+equation), gen=2, 2048 completion, LR=3e-6
- Cost: ~$10 on B200

### Why it failed
1. **Only 21% of Phase 1 steps had nonzero reward** — model couldn't solve most puzzles in 1024 tokens
2. **80% of completions hit max length** without producing \boxed{} — model rambled instead of answering
3. **beta=0.0 (no KL penalty)** — nothing prevented the model from drifting far from the good SFT base
4. **Reward declining over time** — avg reward dropped from 0.88 (steps 1-10) to 0.09 (steps 71-77)
5. Phase 2 compounded the damage on an already degraded adapter

### Lessons
- **NEVER submit GRPO without local validation first** — run tracking system benchmark
- **GRPO needs high reward signal** — if <30% of steps get nonzero reward, model is learning noise
- **Use beta > 0** (KL penalty) to prevent drifting from the SFT base
- **Use shorter max_completion** on Phase 1 (512?) so model learns to give concise answers
- **Filter training data to only prompts the model CAN solve** (Goldilocks zone)
- **The v38a 0.72 adapter is still our best** — don't overwrite it

**Why:** Wasted $10+ and a submission slot. GRPO without sufficient reward signal actively hurts the model.
**How to apply:** Before any future GRPO submission, run local eval on 60 prompts. If score < 0.72, don't submit.
