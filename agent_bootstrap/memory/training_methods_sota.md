---
name: SOTA training methods research (Apr 2026)
description: Which training methods to try for crossing 0.85 ceiling. RAFT first, then token-priority SFT, GSPO last. Skip DPO/GRPO.
type: reference
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Full doc:** `/Users/bharat/Downloads/kaggle/notes/training-methods-research-apr17.md`

## Ranked methods

### ✅ Try these (in order)

1. **RAFT (Rejection-sampling SFT)** — highest EV
   - Sample 32 completions from v26 adapter on 1000 unsolved problems
   - Keep only boxed-correct + verifier-passing traces (20-40% accept)
   - SFT on accepted set + THK's 16720 at 1:3
   - Paper: arXiv 2504.11343
   - Expected: 0.85 → 0.87-0.89, budget ~$80

2. **Token-priority SFT** (formalizes THK's min-logprob intuition)
   - After epoch 1, mark tokens with logprob < 0.69 as "hard"
   - Up-weight hard tokens 3-5× in epoch 2
   - Keep uniform NLL component to avoid format collapse
   - Paper: SFTKey arXiv 2512.21017

3. **GSPO** (NOT vanilla GRPO!) — last resort
   - `importance_sampling_level="sequence"` in TRL
   - KL β=0.05 (NEVER 0)
   - Goldilocks data only (30-70% accept rate)
   - Local eval every 25 steps, STOP on regression
   - Paper: Qwen arXiv 2507.18071

### ❌ Skip these

- **DPO / ORPO / KTO / SimPO / IPO** — proven to regress on reasoning tasks
- **Vanilla GRPO** — already failed at 0.47 for us, MoE-incompatible
- **Full fine-tune** — rule-forbidden (rank ≤ 32)

## Key learnings

1. THK's "maximize minimum logprob" is operational form of token-priority SFT
2. For NemotronH MoE, GSPO is the only stable RL (fixes GRPO's MoE collapse)
3. DAPO (NVIDIA's variant) recipe matches our model family exactly — use as scaffolding
4. Data matters more than method: mixed 1:3 ratios, curriculum easy→hard, verify every teacher trace

## Published recipes matching our stack (PEFT + TRL + transformers)

- `RLHFlow/Minimal-RL` — RAFT reference
- `NVIDIA-NeMo/Nemotron/usage-cookbook/Nemotron-3-Super/grpo-dapo` — official NVIDIA Nemotron recipe
- `inclusionAI/AReaL` — GSPO implementation
- `sail-sg/understand-r1-zero` — Dr. GRPO code

## Budget estimate for full push to 0.87+
- Phase 1 (RAFT): ~$80
- Phase 2 (token-priority SFT): ~$20
- Phase 3 (GSPO if needed): ~$80
- Total: $180 worst case, 2-3 days

## MoE-specific warning
Nemotron-3-Nano-A3B (128 experts + Mamba) is particularly sensitive to RL collapse. GRPO is unsafe — use GSPO specifically.
