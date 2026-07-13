---
name: competition
description: NVIDIA Nemotron Model Reasoning Challenge - submit LoRA adapter for Nemotron-3-Nano-30B, scored on accuracy of reasoning puzzles
type: project
---

Competition: NVIDIA Nemotron Model Reasoning Challenge
- Submit LoRA adapter (rank ≤ 32) for Nemotron-3-Nano-30B
- Eval: vLLM loads model+LoRA, appends "\boxed{}" instruction, temp=0.0, max_tokens=7680
- Score: accuracy (exact string match or 1e-2 numerical tolerance)
- 6 puzzle types in train data (~1550 each): bit_manipulation, cipher, gravity, unit_conversion, numeral_system, equation_transform
- Deadline: June 15, 2026. Midpoint prize: April 9, 2026
- Top score as of March 19: 0.68
- Top score as of April 7, 2026: **0.84** ("Just a test"). Top 10 cluster: 0.80–0.84. Our best 0.69 (SFT v7, March 21) → gap of **0.15** to top.
- 2 months remain until June 15 deadline. Midpoint prize April 9 — **2 days away**.

**Why:** Prize $106K + DGX Sparks. Midpoint prize April 9.
**How to apply:** All work must produce a LoRA adapter. No code submissions. Training data generation + SFT + RL are the paths to improve.
