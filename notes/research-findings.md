# Research Findings — March 19, 2026

## Key Insight: Hidden Test Set is Broader
The evaluation benchmark spans: math, coding, science (GPQA Diamond), instruction following (IFEval), and logic puzzles (Reasoning Gym) — NOT just the 6 puzzle types in train.csv.

## Top Approaches
1. SFT distillation from strong teacher (DeepSeek-R1, Qwen3) — NemoSkills 1st place AIMO-2
2. GRPO reinforcement learning on top of SFT
3. Tool-Integrated Reasoning (model emits Python for computation)
4. Cascaded domain-wise RL (math → code → science → logic sequentially)
5. Parallel sampling with majority voting

## Training Data
- NVIDIA's Llama-Nemotron-Post-Training-Dataset: 18M+ SFT samples (HuggingFace)
- Nemotron-Pretraining-SFT-v1 dataset (HuggingFace)
- 75% reasoning / 25% non-reasoning ratio
- Use <think></think> tokens for reasoning traces

## Frameworks
- Unsloth: best for memory efficiency (80% VRAM savings), GRPO support, free Colab notebooks
- TRL: native GRPO/DPO support
- QLoRA (4-bit) is practical default

## Key Papers
- Nemotron-Cascade (arXiv:2512.13607): cascaded domain-wise RL
- ProRL: extends GRPO to 2000+ training steps across diverse domains
- NemoSkills 1st place: synthetic CoT from DeepSeek-R1 + Tool-Integrated Reasoning
