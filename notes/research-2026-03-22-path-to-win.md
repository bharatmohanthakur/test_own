# Research: Complete Path from 0.68 to 0.76+
# Date: 2026-03-22

## Current State
- Score: 0.68, #1 is 0.76 (gap = 0.08)
- Training: SFT v5, 2000 quality CoT examples, 3 epochs, rank=32, alpha=32
- LoRA targets: in_proj|out_proj|up_proj|down_proj (missing QKV attention!)
- Max seq len: 2048 (eval uses 8192, max_tokens=7680)
- Data: only 6 puzzle types from train.csv

## Gap Analysis: What's Missing (0.08 gap)

### 1. Hidden test includes BROADER tasks beyond 6 puzzle types
- Math (AIME-level, MATH, competition math)
- Coding (LiveCodeBench, HumanEval)
- Science (GPQA Diamond - biology, physics, chemistry)
- Instruction following (IFEval)
- Logic puzzles (Reasoning Gym tasks)
- Our current training ONLY covers 6 puzzle types = overfitting to train distribution

### 2. LoRA alpha too low (alpha=32 for rank=32 = ratio 1:1)
- Research shows alpha=2*rank performs better (alpha=64 for rank=32)
- Higher alpha = stronger adaptation signal

### 3. Missing attention layer targets
- Model has 6 GQA attention layers (critical for cross-token reasoning)
- We target in_proj|out_proj|up_proj|down_proj but NOT q_proj|k_proj|v_proj|o_proj
- Adding QKV+O targets = +3-8% based on research

### 4. Seq length mismatch
- Training at 2048, eval at 8192 with max_tokens=7680
- Long reasoning traces get truncated during training
- Should train at 4096 minimum

### 5. Data quality: hand-crafted CoT vs teacher distillation
- Current: regex-based CoT generation (brittle, doesn't generalize)
- Better: Use DeepSeek-R1 or Qwen3 to generate high-quality reasoning traces
- Best: Rejection sampling - generate multiple solutions, keep verified correct ones

---

## THE PLAN: 5 Steps from 0.68 to 0.76+

### Step 1: Better Data (Expected: +0.04 to +0.06)
**This is the single biggest lever.**

#### A. Teacher Distillation for All 9500 Training Examples
- Use DeepSeek-R1 API (or Qwen3-30B on Vast.ai) to generate detailed CoT for ALL 9500 examples
- Prompt format: "Solve this puzzle step by step. Show your reasoning in <think> tags. Put final answer in \\boxed{}"
- Generate 4 solutions per problem, keep only those matching ground truth (rejection sampling)
- This replaces hand-crafted regex CoT with genuine reasoning traces
- Cost: ~$5-10 via DeepSeek API for 9500 problems x 4 generations

#### B. Add Broader Reasoning Data (Critical for hidden test)
From HuggingFace, curate 3000-5000 additional examples across:
- **Math**: OpenMathInstruct-2 or OpenMathReasoning (NVIDIA) - 500 examples
- **Code**: LiveCodeBench-style or APPS problems with CoT - 500 examples
- **Science**: GPQA Diamond training split - 500 examples
- **Logic**: Reasoning Gym puzzles (procedurally generated) - 500 examples
- **Instruction Following**: IFEval-style examples - 300 examples
- **General Reasoning**: OpenThoughts-114k (subset, DeepSeek-R1 traces) - 700 examples
All with <think>reasoning</think>\n\n\\boxed{answer} format

#### C. Data Format Requirements
- 75% reasoning (with <think> traces), 25% direct answers
- Ensure \\boxed{} is never truncated (max_seq_len must accommodate full trace)
- Use tokenizer.apply_chat_template with enable_thinking=True format

### Step 2: LoRA Config Fix (Expected: +0.02 to +0.04)

```python
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,  # alpha = 2 * rank (was 32)
    target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|up_proj|down_proj|gate_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

Key changes:
- alpha: 32 -> 64 (2x rank, empirically better for generalization)
- targets: ADD q_proj, k_proj, v_proj, o_proj (attention layers for reasoning)
- NEVER target router layers (collapse expert specialization)
- NEVER target SSM core modules (LoRA ineffective on SSM, target projectors only)

### Step 3: Training Config Fix (Expected: +0.01 to +0.02)

```python
MAX_SEQ_LEN = 4096  # was 2048, matches eval max_model_len=8192
NUM_EPOCHS = 3      # for ~12K total examples
LR = 1e-4           # slightly lower for larger dataset + higher alpha
GRAD_ACCUM = 16     # effective batch size 16
WARMUP_RATIO = 0.05
```

Key changes:
- Seq len: 2048 -> 4096 (prevents truncation of reasoning traces)
- LR: 2e-4 -> 1e-4 (compensate for alpha doubling; alpha/rank scaling)
- Warmup: add cosine schedule with warmup
- Use HF Trainer for proper scheduling (not manual loop)

### Step 4: Two-Stage Training (Expected: +0.01 to +0.03)

#### Stage 1: SFT (primary)
- Train on full dataset (9500 puzzles + 3000-5000 broader reasoning)
- 3 epochs, LR=1e-4, cosine decay
- Focus on learning format + reasoning patterns

#### Stage 2: DPO (refinement, if time allows)
- Generate model predictions on training set after SFT
- Create preference pairs: correct answers (chosen) vs incorrect (rejected)
- 1-2 epochs DPO to sharpen accuracy and reduce verbosity
- This is what AIMO-2 2nd place used for +2-3% improvement

### Step 5: Validation Before Submit
- Hold out 500 examples (balanced across types) for validation
- Run local inference with vLLM settings matching eval (temp=0.0, max_tokens=7680)
- Compare accuracy per puzzle type to identify weak areas
- Re-train with more data for weakest categories

---

## Key Resources

### Datasets (HuggingFace)
- nvidia/OpenMathReasoning - 306K math problems with CoT
- nvidia/OpenMathInstruct-2 - 14M math problem-solution pairs
- open-thoughts/OpenThoughts-114k - 114K reasoning examples (math, science, code, puzzles)
- nvidia/Llama-Nemotron-Post-Training-Dataset - 30M samples (math, code, instruction)
- open-thought/reasoning-gym - procedural puzzle generators with verifiers

### Frameworks
- Unsloth: 80% VRAM savings, GRPO support, Nemotron 3 Nano LoRA support
- TRL: GRPO + DPO native support
- NeMo RL: official NVIDIA GRPO implementation

### Key Papers
- NemoSkills 1st place AIMO-2: SFT + TIR + GenSelect
- Fast-Math-R1 2nd place AIMO-2: Extended SFT (10 epochs) then GRPO
- Enigmata: Synthetic verifiable puzzles for RL training
- Reasoning Core: Procedural symbolic data generation

---

## Estimated Score Progression

| Step | Change | Score | Notes |
|------|--------|-------|-------|
| Current | - | 0.68 | SFT v5, 2000 examples |
| +Teacher distillation | +0.03 | 0.71 | Better CoT quality on same data |
| +Broader data | +0.03 | 0.74 | Math, code, science, logic |
| +LoRA config fix | +0.02 | 0.76 | alpha=64, +QKV targets |
| +Training config | +0.01 | 0.77 | seq_len=4096, proper LR |
| +DPO stage | +0.01 | 0.78 | Sharpen accuracy |

## Priority Order (if GPU budget is tight)
1. Teacher distillation + broader data (biggest ROI)
2. LoRA alpha=64 + add QKV targets
3. Seq len 4096
4. DPO refinement (only if Steps 1-3 done)
