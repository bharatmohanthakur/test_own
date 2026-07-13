---
name: grpo_strategy
description: GRPO reinforcement learning strategy — difficulty-aware data selection for post-SFT training
type: project
---

## GRPO Strategy (after SFT baseline is established)

### Data Selection — Goldilocks Zone
- Only use prompts where model accuracy is **30-70%** (from multiple attempts)
- Too easy (>70%) = no signal. Too hard (<30%) = no signal.
- Weight harder problems MORE in the loss

### Pipeline
1. Run best SFT model on problems (4 completions each, Unsloth inference mode)
2. Score each: correct/incorrect using competition verify()
3. Keep problems with 1-3 correct out of 4 (25-75%)
4. GRPO train on those — model learns from its own correct attempts
5. After each epoch, re-evaluate difficulty, drop easy ones, add harder

### Practical Setup
- ~500 Goldilocks-zone prompts × 4 completions = 2000 generations
- ~1h inference + ~1h GRPO = fits in single Kaggle session
- Use Unsloth `FastLanguageModel.for_inference()` for 2x faster generation
- Reward function: binary (correct=+1, wrong=-1) using verify()

### References
- GRPO-LEAD (arxiv 2504.09696): difficulty-aware reweighting
- DeepSeekMath (arxiv 2402.03300): GRPO for math reasoning
- Key: two-stage curriculum, dynamic data selection

### Known Issues
- **Mamba generation**: Must patch `is_fast_path_available = False` for generation during GRPO
- **Padding direction**: Switch `tokenizer.padding_side` from "right" (SFT) to "left" (GRPO)
- **Adapter merge**: Two approaches — (1) continue existing LoRA (simpler but mixed SFT+GRPO weights), (2) merge SFT into base then fresh LoRA for GRPO (cleaner submission)
- **For_inference/for_training**: Use `FastLanguageModel.for_inference(model)` before GRPO generation
- **Chunked-log-softmax shape mismatch on Nemotron** (errors_fixes #20): need v28 monkey-patch on `chunked_hidden_states_selective_log_softmax` to detect logits-shaped input and fall back to `chunked_selective_log_softmax`. Pattern saved in `notebooks/grpo_only_v28/train.py` and reused unchanged in v31/v32 RunPod scripts.
- **VRAM growth per step** (errors_fixes #2470, #3864): known Unsloth GRPO leak. Mitigations: `save_steps=2` (small) so OOM at late steps still leaves a usable checkpoint, `gradient_accumulation_steps=2-4`, `max_completion_length=512`, `num_generations=2`, `PYTORCH_ALLOC_CONF=expandable_segments:True`. v31 ran 10 steps at peak ~94.5GB on 96GB RTX Pro 6000 — leak plateaued at step 3.

### Goldilocks data construction (validated Apr 8, 2026)
Don't sample randomly — combine **distilled correct examples from multiple teachers**, dedupe by id:
1. Run Grok 4.20 over all puzzle prompts → keep only the ones where Grok's answer matches ground truth → `distilled_grok.jsonl`
2. Run DeepSeek R1 over the same → `distilled_r1.jsonl`
3. Union by id (179 unique from 166 Grok + 60 R1, 47 overlap) — these are problems PROVEN solvable by a reasoning model
4. Use as GRPO prompt set
5. v31 used all 179. v32 used a balanced 50 (17 hardest types + 17 longest + 16 random) for the polish pass.

### GSPO via TRL flag (validated Apr 8, 2026)
TRL 0.24.0+ supports Group SEQUENCE Policy Optimization (Qwen team, mid-2025) via a single config flag:
```python
GRPOConfig(..., importance_sampling_level="sequence")  # default is "token" = standard GRPO
```
GSPO is more stable for long completions because it collapses the per-token importance ratio to one ratio per sequence. v32 used this on top of v31's GRPO adapter for a 2-step polish — same code path, same TRL trainer, same Mamba patches all work unchanged.

### Distillation Results (Grok 4.20 vs DeepSeek R1)
- Grok 4.20-reasoning: ~70% accuracy on puzzles, solves bit_manipulation!
- DeepSeek R1: ~20% accuracy, fails most hard puzzles
- Grok is 3.5x better teacher for this competition
- xAI API key available, ~$8-10 for 500 puzzles, covered by free tier

**Why:** SFT teaches format + basic reasoning. GRPO teaches the model to EXPLORE and self-correct. Top teams (0.82) likely use SFT+GRPO pipeline.
**How to apply:** Only after SFT score is established. Use Grok 4.20 distilled data for GRPO prompts.
