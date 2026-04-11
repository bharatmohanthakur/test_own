# NVIDIA Nemotron Model Reasoning Challenge — Summary

## What You Submit
- A **LoRA adapter** (rank ≤ 32) for Nemotron-3-Nano-30B base model
- Packaged as `submission.zip` containing LoRA weights + `adapter_config.json`
- NO code submission — they run inference with vLLM on their side

## How Evaluation Works
1. They load Nemotron-3-Nano-30B + your LoRA via vLLM
2. Each test prompt gets appended with: `Please put your final answer inside \boxed{}`
3. Chat template applied with `enable_thinking=True`
4. Answer extracted from `\boxed{}` in model output
5. Verified: exact string match (case-insensitive) OR numerical tolerance 1e-2
6. Score = proportion of correct answers

## Inference Parameters (from Evaluation page — these override metric code defaults)
| Parameter | Value |
|-----------|-------|
| max_lora_rank | 32 |
| max_tokens | 7680 |
| top_p | 1.0 |
| temperature | 0.0 |
| max_num_seqs | 64 |
| gpu_memory_utilization | 0.85 |
| max_model_len | 8192 |

## Timeline
- Started: March 16, 2026
- **Midpoint cut-off: April 9, 2026** (progress prize: $5K + 1 DGX Spark)
- Entry deadline: June 8, 2026
- **Final deadline: June 15, 2026**

## Prizes ($106,388 total)
- 1st: $25K + 5 DGX Sparks
- 2nd: $15K + 2 DGX Sparks
- 3rd: $5K + 1 DGX Spark
- Progress prize: $5K + 1 DGX Spark (midpoint leader)
- Open contribution awards: 3x DGX Sparks (best data, best RL, best fine-tuning)

## Leaderboard (as of March 19, 2026)
- Top score: 0.68 (multiple teams tied)
- ~218 teams, 519 submissions
- Public LB uses ~50% of test data; private LB uses other 50%

## Data
- train.csv: 9,500 rows (id, prompt, answer)
- test.csv: 3 public rows (real test is hidden)
- 6 puzzle types (~1,550 each): bit manipulation, cipher, gravity, unit conversion, numeral system, equation transform

## Allowed Approaches
- Prompting strategies
- Data filtering & curation (NeMo Curator)
- Synthetic data generation (NeMo Data Designer)
- Reinforcement learning (NeMo RL, GRPO)
- Lightweight fine-tuning (any framework: HF, Unsloth, Axolotl, TRL)

## Key Resources
- Submission demo: kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo
- Metric code: kaggle.com/code/metric/nvidia-nemotron-metric
- Model: kaggle.com/models/metric/nemotron-3-nano-30b-a3b-bf16
- Nemotron GitHub: github.com/NVIDIA-NeMo/Nemotron
- NeMo RL: github.com/NVIDIA-NeMo/RL
- NeMo Gym: github.com/NVIDIA-NeMo/gym
- Data Designer: github.com/NVIDIA-NeMo/DataDesigner
- Discord: discord.gg/kRwhDfTW

## Key Discussion Insights
- "Are problem types the same for train and test?" — important question, unclear
- CUDA Blackwell GPU fix needed for RTX PRO 6000
- mamba_ssm dependency issues in no-internet environments
- "Increasing token limit doesn't help" — SFT and RL are the way
- Kaggle provides RTX PRO 6000 Blackwell GPUs (G4 VMs)
