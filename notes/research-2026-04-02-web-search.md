# Research: Web Search for Nemotron Reasoning Challenge Techniques
**Date**: 2026-04-02

## Current Position
- Our best score: 0.69 (SFT with 500 curated examples)
- #1 score: 0.82
- Gap to close: 0.13
- Midpoint prize deadline: April 9, 2026

---

## KEY FINDING 1: Nemotron-Cascade 2 Training Recipe (NVIDIA's own approach)

### Training Pipeline (sequential):
1. **SFT** → 15.9M samples, packed into 256K token sequences, ~1.5 epochs optimal
2. **IF-RL** (~180 steps) — instruction following first
3. **Multi-domain RL** (~70 steps) — MCQA, tool calling, structured output
4. **Multi-domain On-Policy Distillation** (~40-50 steps) — key innovation
5. **RLHF** (~30 steps) — human preference alignment
6. **Long-context RL** (~30 steps)
7. **Code RL** — binary rewards, hardest problems only
8. **SWE RL** (~40-50 steps)

### SFT Hyperparameters:
- Global Batch Size: 64
- Max LR: 5e-5, Min LR: 5e-6
- Warmup: 200 steps
- Scheduler: cosine
- Max steps: 40,000 (optimal at ~33,000)
- Optimizer: AdamW (beta1=0.9, beta2=0.98)
- Weight decay: 0.1

### RL Details:
- GRPO with strict on-policy training (importance ratio always 1.0)
- **NO KL divergence term** (reduces to REINFORCE with group-normalized rewards)
- Exception: RLHF stage uses KL coefficient 0.03
- Token-level loss calculation
- For Code RL: aggressively filtered to 3.5K hardest prompts only

### Available Datasets:
- `nvidia/Nemotron-Cascade-2-SFT-Data` — 25.8M samples across 8 domains
- `nvidia/Nemotron-Cascade-2-RL-data` — RL training prompts
- `nvidia/Nemotron-Post-Training-Dataset-v2` — 5.7M reasoning samples from DeepSeek-R1

**Source**: https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/
**Source**: https://maximelabonne.substack.com/p/nemotron-cascade-2-on-policy-distillation

---

## KEY FINDING 2: Data Quality > Data Quantity

### Teacher Models Used for Synthetic Data:
- **DeepSeek-R1-0528** — 5.7M reasoning samples (primary reasoning teacher)
- **DeepSeek-V3.2** and **DeepSeek-V3.2-Speciale** — math responses
- **GPT-OSS-120B** — science, agentic tasks
- **Qwen3-235B-A22B-Thinking-2507** — agentic tasks
- **Qwen3-32B** — agentic tasks

### Quality Filtering (critical):
- Aggressively filter reasoning traces with pathological repetition
- Remove repeated n-grams within sliding window or across trajectory
- Remove inconsistent prompts, easy-to-guess answers, incorrect syntax
- For code: remove problems solved by all rollouts (too easy)
- Data deduplication: I/O fingerprinting removed ~24.2% redundancy

### Dataset Composition for Reasoning:
- **75% reasoning + 25% non-reasoning** to preserve reasoning capabilities
- Include truncated reasoning traces (~5% of data) for budget forcing
- Truncate to 1-2K tokens while preserving final answer

---

## KEY FINDING 3: Nemotron Nano 2 Post-Training Recipe

### Multi-Stage SFT:
- Stage 1: Foundation skills across domains
- Stage 2: Targeted skills (tool use, long-context)
- Stage 3: Truncated/budgeted training (reasoning traces cut to 1-2K tokens)

### GRPO Implementation:
- Used English-only contexts from HelpSteer3
- Responses generated with AND without thinking traces
- Qwen-based reward model judged rollouts
- Multi-environment RL across: math, code, science, IF, tool use, conversations, structured output

### DPO + GRPO Critical for:
- Instruction following (IFEval)
- Function calling (BFCL v3)
- Both temporarily degrade MMLU-Pro, recovered by post-GRPO knowledge distillation

---

## KEY FINDING 4: LoRA Configuration Best Practices

### Recommended Settings:
- Rank: 32 (max allowed by competition)
- Alpha: 32 or 64 (equal to rank or 2x rank)
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, in_proj, out_proj
- **NEVER target router layers** — will collapse expert specialization
- Dropout: 0.05

### Unsloth Support:
- 30B model requires ~60GB VRAM for 16-bit LoRA
- GRPO training available via NeMo Gym integration
- Notebooks available for both SFT and GRPO

---

## KEY FINDING 5: Winning Strategies from Related Competitions

### AIMO-2 (Math Olympiad) — 1st Place (NVIDIA):
- Foundation: Qwen2.5-14B-Base
- Fine-tuned on **millions of synthetically generated solutions**
- Teacher models: DeepSeek-R1 and QwQ-32B
- Combination of natural language reasoning + Python code execution
- OpenMath-Nemotron-14B-Kaggle: 73.7% pass@1 on AIME24

### ARC-AGI-2 Prize — 1st Place (NVIDIA):
- Synthetic data generation using staged puzzle generation
- Concept decomposition approach
- Test-time training (TTT) for adaptation
- Smaller models (4B) outperformed larger models

### Key Pattern Across Winners:
1. Massive synthetic data generation with strong teacher models
2. Aggressive data filtering and quality control
3. Domain-specific fine-tuning rather than generic
4. Combined SFT + RL pipeline

---

## KEY FINDING 6: Cooperative SFT-RL (BRIDGE Method)

- Recent research shows tightly integrated SFT-RL optimization outperforms sequential
- Budget forcing RL: accuracy jumps from 34% to 72% at 1K token budget
- LoRA provides best balance: skill acquisition with minimal forgetting
- Dense SFT excels at skill acquisition but causes catastrophic forgetting

---

## ACTIONABLE RECOMMENDATIONS (Priority Order)

### 1. Generate High-Quality Synthetic CoT with Teacher Models (HIGHEST IMPACT)
- Use DeepSeek-R1 or Qwen3-235B to generate detailed reasoning traces for all 9,500 training examples
- Also generate traces for BROADER reasoning tasks (math, science, logic)
- Filter for quality: remove repetitive traces, check answer correctness
- Target: 2,000-5,000 high-quality examples with verified answers

### 2. Use Nemotron-Cascade-2-SFT-Data Subsets
- Download math and science subsets from HuggingFace
- Filter for relevant reasoning tasks
- Mix with competition-specific data (75/25 reasoning/non-reasoning)

### 3. Improve SFT Hyperparameters
- LR: 5e-5 (lower than our current 2e-4)
- Cosine scheduler with warmup
- Weight decay: 0.1
- AdamW with beta2=0.98
- 1.5 epochs optimal

### 4. Add Truncated Reasoning Traces
- Include 5% of training data with reasoning truncated to 1-2K tokens
- Preserves final answer — teaches model to be concise when needed
- Helps with inference token budget (max_tokens=7680)

### 5. Expand Training Data Diversity
- Add general reasoning tasks (not just the 6 puzzle types)
- Include math, science, instruction following
- Hidden test set likely includes broader tasks

### 6. Consider GRPO After SFT
- Implement sequential: SFT first, then GRPO
- Binary rewards for verifiable puzzles
- No KL term (per Cascade 2 recipe)
- Filter to hardest examples only for RL
