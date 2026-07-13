---
name: NVIDIA official GRPO/DAPO recipe
description: Exact hyperparameters from NVIDIA's Nemotron-3-Super GRPO/DAPO training cookbook. Use as reference for our RL training — these are the settings that produced the base model.
type: reference
---

# NVIDIA GRPO/DAPO Recipe (from grpo_training_cookbook.ipynb)

Source: github.com/NVIDIA-NeMo/Nemotron/usage-cookbook/Nemotron-3-Super/grpo-dapo/

## Key hyperparameters
```yaml
grpo:
  max_num_steps: 2000
  num_prompts_per_step: 32
  num_generations_per_prompt: 16
  normalize_rewards: true
  use_leave_one_out_baseline: true  # DAPO feature

loss_fn:
  ratio_clip_min: 0.2    # DAPO asymmetric clipping
  ratio_clip_max: 0.28   # clip_higher > clip_lower
  token_level_loss: true  # per-token, not per-sequence
  reference_policy_kl_penalty: 0.0  # NO KL penalty

reward_shaping:
  overlong_buffer_length: 512
  overlong_buffer_penalty: 1.0

reward_scaling:
  source_min: 0.0, source_max: 1.0
  target_min: -1.0, target_max: 1.0
```

## Dataset
- DAPO-Math-17k (verifiable math)
- Validation: AIME-2024

## Hardware
- 20-24 GPUs (non-colocated: 4-8 for vLLM rollout, 16 for training)
- We have 1 GPU — can't replicate scale but CAN use the hyperparameters

## What we can adopt in TRL GRPOTrainer
1. `epsilon_high=0.28` (DAPO asymmetric clipping — already available in TRL)
2. No KL penalty (already doing this)
3. `mask_truncated_completions=True` (from Unsloth notebook)
4. Reward scaling to [-1, 1] range
5. Token-level loss (check TRL support)
6. More generations per prompt if budget allows (16 ideal, 4-8 practical)

## NeMo Gym integration
- NeMo Gym provides verified reward servers (math verification, code execution)
- Can collect offline rollouts (fast with vLLM) then train SFT/DPO on good completions
- This avoids the slow `transformers.generate` bottleneck in our single-GPU GRPO
