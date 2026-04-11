# Comprehensive Plan: Path to 0.90 Accuracy on NVIDIA Nemotron Reasoning Challenge
# Research Date: 2026-03-22
# Current Score: 0.69 | #1 Score: 0.76 | Target: 0.90

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Competition Analysis](#competition-analysis)
3. [Chat Template & Eval Format](#chat-template--eval-format)
4. [Architecture Deep Dive: LoRA Target Modules](#architecture-deep-dive)
5. [AIMO-2 Winning Solution Analysis](#aimo-2-winning-solution)
6. [Nemotron-Cascade 2 Training Recipe](#nemotron-cascade-2)
7. [Data Recipe](#data-recipe)
8. [Training Pipeline: Step by Step](#training-pipeline)
9. [GRPO Reinforcement Learning](#grpo-reinforcement-learning)
10. [DPO / Step-DPO Refinement](#dpo-refinement)
11. [Hidden Test Benchmark Preparation](#hidden-test-preparation)
12. [Hyperparameters](#hyperparameters)
13. [Common Mistakes to Avoid](#common-mistakes)
14. [Estimated Score Progression](#score-progression)
15. [Sources](#sources)

---

## 1. Executive Summary

The gap from 0.69 to 0.90 requires attacking **five dimensions simultaneously**:

1. **Data Quality** (biggest lever, +0.08-0.12): Replace hand-crafted CoT with teacher-distilled reasoning traces from DeepSeek-R1/Qwen3. Use rejection sampling to keep only verified-correct solutions.
2. **Data Breadth** (critical, +0.04-0.06): Hidden test includes math, coding, science (GPQA), instruction following (IFEval), and logic puzzles (Reasoning Gym) -- not just the 6 puzzle types.
3. **LoRA Config** (important, +0.02-0.04): Add QKV attention targets, increase alpha to 64, add gate_proj.
4. **Training Method** (significant, +0.03-0.06): Move from SFT-only to SFT -> GRPO pipeline. Possibly add DPO refinement.
5. **Format Matching** (critical, +0.02-0.03): Exact match to eval chat template (`<|im_start|>`, `<think>`, `\boxed{}`), train at seq_len=4096.

---

## 2. Competition Analysis

### What We Know
- **Model**: Nemotron-3-Nano-30B (30B total, 3B active, hybrid Mamba-Transformer MoE)
  - 52 total layers: 23 Mamba-2 + 23 MoE + 6 GQA attention layers
  - Each MoE layer: 128 routed experts + 1 shared expert, 6 activated per token
- **Submit**: LoRA adapter only (rank <= 32), `submission.zip`
- **Eval**: vLLM with temp=0.0, top_p=1.0, max_tokens=7680, max_model_len=8192
- **Scoring**: `\boxed{}` extraction, case-insensitive string match OR numeric tolerance 1e-2
- **Chat template**: `enable_thinking=True` applied, model uses `<think>` reasoning

### Hidden Test Categories (CRITICAL)
The test set is BROADER than the 6 training puzzle types. Based on NVIDIA's Nemotron evaluation benchmarks and competition hints:
- **Math** (AIME, MATH, competition math)
- **Coding** (LiveCodeBench-style)
- **Science** (GPQA Diamond: biology, physics, chemistry PhD-level)
- **Instruction Following** (IFEval: verifiable format constraints)
- **Logic Puzzles** (Reasoning Gym: 100+ procedural puzzle types)
- **The 6 train types** (bit_manipulation, cipher, gravity, unit_conversion, numeral_system, equation_transform)

**Implication**: Training ONLY on the 6 puzzle types leads to overfitting on ~30-50% of the test set. Must train for generalization.

### Leaderboard Context
- Current #1: 0.76
- Our best: 0.69
- Gap: 0.07 to #1, 0.21 to target 0.90
- Midpoint prize deadline: April 9, 2026

---

## 3. Chat Template & Eval Format (EXACT MATCH REQUIRED)

### The Exact Chat Template
The model uses ChatML-style tokens:
```
<|im_start|>system
{system_message}
<|im_end|>
<|im_start|>user
{user_content}
<|im_end|>
<|im_start|>assistant
<think>
{reasoning_content}
</think>
{final_answer_content}
<|im_end|>
```

### Key Details
- Special tokens: `<|im_start|>` and `<|im_end|>` for message delimiters
- Thinking tokens: `<think>` (token ID 12) and `</think>` (token ID 13)
- `enable_thinking=True` is the DEFAULT -- model generates `<think>` traces
- When `enable_thinking=False` and `add_generation_prompt=True`, the template generates `<think></think>` (empty thinking)
- `truncate_history_thinking`: removes thinking content from previous assistant messages to save context

### Training Data Format (must match exactly)
```python
messages = [
    {"role": "user", "content": prompt + "\nPlease put your final answer inside \\boxed{}."},
    {"role": "assistant", "content": "<think>\n{step-by-step reasoning}\n</think>\n\n\\boxed{answer}"}
]

# Use tokenizer with enable_thinking=True
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    enable_thinking=True,  # DEFAULT but be explicit
    add_generation_prompt=False,  # False for training (we provide the assistant response)
    return_tensors="pt"
)
```

### Answer Extraction
The eval script extracts the LAST `\boxed{}` from the model output. Handle nested braces with stack-based parsing, not regex.

---

## 4. Architecture Deep Dive: LoRA Target Modules

### Model Architecture Summary
- 52 layers total
- 23 Mamba-2 layers (SSM with linear projections)
- 23 MoE layers (128 routed + 1 shared expert per layer, 6 active)
- 6 GQA attention layers (grouped query attention, 2 groups)

### Recommended LoRA Target Modules
```python
target_modules = r".*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|up_proj|down_proj|gate_proj)$"
```

#### What each module does:
- **q_proj, k_proj, v_proj, o_proj**: GQA attention layers (6 layers). Critical for cross-token reasoning and logical deduction
- **in_proj, out_proj**: Mamba-2 linear projections (23 layers). These are the EFFECTIVE targets for Mamba layers -- LoRA on SSM core modules (A, B, C, D) is INEFFECTIVE per research
- **up_proj, down_proj, gate_proj**: MoE expert FFN layers (23 layers). These are the computation-heavy layers

#### What to NEVER target:
- **Router layers**: Targeting MoE routers collapses expert specialization. Unsloth disables this by default
- **SSM core modules** (A_log, D, x_proj, dt_proj): LoRA is ineffective on SSM modules. Research shows targeting linear projections yields comparable or better results
- **Shared expert layers**: May cause instability if targeted along with routed experts

### Research-Backed Findings on LoRA + Mamba
From "Parameter-Efficient Fine-Tuning of State Space Models" (ICML 2025):
- LoRA on linear projections (in_proj, out_proj) >= LoRA on both projections + SSM modules
- LoRA on SSM modules alone is significantly worse
- For MoE models, HELLoRA approach (target hot experts only) uses 15.74% params, improves accuracy by 9.24%

### LoRA Configuration
```python
lora_config = LoraConfig(
    r=32,          # Maximum allowed by competition
    lora_alpha=64, # 2x rank -- empirically better for generalization
    target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|up_proj|down_proj|gate_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

**Why alpha=64?** The scaling factor is alpha/rank = 64/32 = 2.0. Research (Unsloth docs, Tina paper) shows that alpha = 2*rank provides stronger adaptation signal while maintaining stability. The previous alpha=32 (ratio 1.0) was too conservative.

---

## 5. AIMO-2 Winning Solution Analysis (NemoSkills)

### Paper: arXiv:2504.16891
The NemoSkills team won AIMO-2 by correctly solving 34/50 Olympiad questions in 5 hours using 4 L4 GPUs.

### Three Pillars

#### Pillar 1: Large-Scale Dataset (OpenMathReasoning)
- 306K unique high-quality math problems (from AoPS forums, preprocessed by Qwen2.5-32B-Instruct)
- 3.2M Chain-of-Thought solutions (generated by DeepSeek-R1 and QwQ-32B)
- 1.7M Tool-Integrated Reasoning solutions (model interleaves Python code execution)
- 565K GenSelect examples (model trained to select best solution from candidates)
- **Total SFT dataset: 5.5M samples**

#### Pillar 2: Tool-Integrated Reasoning (TIR)
- Novel method: model generates reasoning + Python code, code output feeds back into reasoning
- Iterative process: train -> generate TIR solutions -> quality filter -> retrain
- 1.7M high-quality TIR solutions produced through this iterative pipeline

#### Pillar 3: Generative Solution Selection (GenSelect)
- Instead of majority voting, train the model to SELECT the best solution
- Model reads multiple candidate solutions, generates a judgment of which is correct
- Significantly outperforms majority voting baseline

### Training Details
- Base model: Qwen2.5-14B-Base (fine-tuned from scratch)
- Training: SFT only (no RL needed due to data quality)
- Epochs: 6 epochs on the 5.5M dataset
- RoPE base changed to 500K
- Second stage: additional SFT on subset of harder problems
- **Key insight: Data quality > training method. SFT on excellent data beats RL on mediocre data.**

### What We Can Apply
- We can't do full 5.5M sample training, but we can:
  1. Use rejection sampling with a teacher model to generate verified-correct CoT
  2. Include diverse problem types (not just competition math)
  3. Focus on data quality over quantity
  4. Use the `\boxed{}` format consistently

---

## 6. Nemotron-Cascade 2 Training Recipe

### Paper: arXiv:2603.19220 (March 2026 -- BRAND NEW)
**This uses the EXACT same base model as our competition**: Nemotron-3-Nano-30B-A3B-Base

### Training Pipeline
```
Nemotron-3-Nano-30B-A3B-Base
    → SFT (single stage, packed sequences up to 256K tokens)
    → RLHF (72B reward model, ~82K preference pairs)
    → IF-RL (instruction following with deterministic verifiers)
    → Math-RL (symbolic math, dynamic token-budget curricula)
    → Code-RL
    → SWE-RL
    + Multi-Domain On-Policy Distillation (MOPD) throughout
```

### SFT Data Composition
- 1.9M Python reasoning traces
- 1.3M Python tool-calling samples
- 816K mathematical NL proofs
- 125K agentic SWE samples + 389K agentless SWE samples
- General chat from Nemotron-Instruction-Following-Chat-v1
- Safety samples
- Science (physics, chemistry, biology) from Nemotron-Science-v1
- **All packed into sequences up to 256K tokens, trained ~1.5 epochs**

### Key Innovation: MOPD (Multi-Domain On-Policy Distillation)
- During Cascade RL, save best intermediate checkpoint per domain
- These "teacher" checkpoints provide token-level distillation signal
- Efficiently recovers benchmark regressions from domain switching
- This is why Cascade RL doesn't suffer catastrophic forgetting

### Results
- IFEval: 90.2% (14B variant)
- Math (AIME24): 90.4%
- Code (LCB v6): 74.6% (14B-think)
- SWE-bench Verified: 43.1%
- Gold Medal in IMO, IOI, and ICPC World Finals

### What We Can Apply
- The SFT data is publicly available on HuggingFace: `nvidia/Nemotron-Cascade-2-SFT-Data`
- We can DIRECTLY USE subsets of this data for our LoRA training
- The data already uses the correct chat template format for Nemotron-3-Nano
- Focus on: math + science + instruction following + general reasoning subsets

---

## 7. Data Recipe

### Overview: What Data to Use

The data recipe is the SINGLE MOST IMPORTANT factor. NemoSkills won AIMO-2 with SFT alone because the data was exceptional.

### Data Sources (Priority Order)

#### Source 1: Competition Train Data with Teacher CoT (9,500 examples)
- Use DeepSeek-R1 API to generate detailed reasoning traces for all 9,500 training examples
- Prompt: `"Solve this step by step. Show detailed reasoning. Put final answer in \boxed{}."`
- Generate 4 solutions per problem, keep only those matching ground truth answer
- **Rejection sampling** ensures correctness
- Cost estimate: ~$5-15 via DeepSeek API
- Format: `<think>\n{detailed reasoning}\n</think>\n\n\boxed{answer}`

#### Source 2: Nemotron-Cascade-2-SFT-Data (from HuggingFace)
This is NVIDIA's own SFT data for the SAME base model. Cherry-pick subsets:
- Math (non-proof): 2,000-3,000 examples
- Science (physics, chemistry, biology): 1,000 examples
- General chat / instruction following: 1,000 examples
- Already in correct format for Nemotron-3-Nano

#### Source 3: OpenMathReasoning (from HuggingFace, nvidia/OpenMathReasoning)
- 306K problems with 3.2M CoT solutions
- Cherry-pick: 2,000-3,000 medium-difficulty problems (not just Olympiad-level)
- Focus on problems solvable with step-by-step reasoning
- Include diverse difficulty: 50% easy/medium, 50% hard

#### Source 4: Broader Reasoning Data
- **GPQA Diamond** training split: 400 examples (graduate-level science MCQ with 4 options)
- **IFEval-style**: 300-500 examples of verifiable instruction-following
- **Reasoning Gym** puzzles: 500-1,000 procedurally generated logic puzzles
- **Code reasoning**: 500 examples from APPS or LiveCodeBench with step-by-step solutions

#### Source 5: Enigmata Puzzles (for RL stage)
- 36 puzzle types, 7 categories (Crypto, Arithmetic, Logic, Grid, Graph, Search, Sequential)
- Generators produce unlimited examples with controllable difficulty
- Rule-based verifiers for automatic evaluation
- Perfect for GRPO reward signals

### Data Composition Target
| Category | Examples | % of Total | Source |
|----------|----------|-----------|--------|
| Competition puzzles (6 types) | 9,500 | 45% | train.csv + teacher CoT |
| Math reasoning | 3,000 | 14% | OpenMathReasoning |
| Cascade-2 SFT (mixed) | 4,000 | 19% | nvidia/Nemotron-Cascade-2-SFT-Data |
| Science (GPQA) | 500 | 2% | GPQA Diamond |
| Instruction following | 500 | 2% | IFEval-style |
| Logic puzzles | 1,000 | 5% | Reasoning Gym / Enigmata |
| Code reasoning | 500 | 2% | APPS / LiveCodeBench |
| General reasoning | 2,000 | 10% | OpenThoughts-114k subset |
| **TOTAL** | **~21,000** | **100%** | |

### Data Quality Checklist
- [ ] Every example has `<think>` reasoning traces (at least 100 tokens of reasoning)
- [ ] Every example ends with `\boxed{answer}` (never truncated)
- [ ] 75% reasoning with `<think>` traces, 25% direct short answers
- [ ] Answers verified correct (rejection sampling for teacher-generated)
- [ ] No duplicate problems
- [ ] Diverse difficulty levels (not all easy, not all impossible)
- [ ] Maximum sequence length 4096 tokens (to avoid truncation)

---

## 8. Training Pipeline: Step by Step

### Phase 0: Data Preparation (runs on CPU, no GPU needed)

```python
# 1. Generate teacher CoT for competition data
# Use DeepSeek-R1 API or MinMax-2.7 API
for problem in train_csv:
    solutions = generate_n_solutions(problem, n=4, model="deepseek-r1")
    correct_solutions = [s for s in solutions if extract_boxed(s) == problem.answer]
    if correct_solutions:
        best = select_longest_reasoning(correct_solutions)
        training_data.append(format_as_chat(problem.prompt, best))

# 2. Download and filter Cascade-2 SFT data
cascade_data = load_dataset("nvidia/Nemotron-Cascade-2-SFT-Data")
math_subset = cascade_data.filter(lambda x: x["category"] == "math")[:3000]
science_subset = cascade_data.filter(lambda x: x["category"] == "science")[:1000]

# 3. Format everything with proper chat template
for example in all_data:
    messages = [
        {"role": "user", "content": example["prompt"] + "\nPlease put your final answer inside \\boxed{}."},
        {"role": "assistant", "content": f"<think>\n{example['reasoning']}\n</think>\n\n\\boxed{{{example['answer']}}}"}
    ]
    tokenized = tokenizer.apply_chat_template(messages, enable_thinking=True, ...)
```

### Phase 1: SFT (Primary Training) — ~45-90 min on RTX Pro 6000

```python
# Model setup
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto",
    trust_remote_code=True, torch_dtype=torch.bfloat16)

# LoRA config
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|up_proj|down_proj|gate_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()

# Apply Blackwell fixes
# ... (patch is_fast_path_available, rmsnorm_fn, ptxas)

# Training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # effective batch = 16
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    save_strategy="no",
    report_to="wandb",
    max_grad_norm=1.0,
    dataloader_num_workers=0,
)

# Data
MAX_SEQ_LEN = 4096
# ... tokenize, truncate at MAX_SEQ_LEN, set labels

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

# Save
model.save_pretrained(OUTPUT_DIR)
```

### Phase 2: GRPO Reinforcement Learning (Optional, +0.03-0.06)

If Phase 1 achieves 0.75+, invest in GRPO for further gains:

```python
from trl import GRPOConfig, GRPOTrainer

# Reward functions
def accuracy_reward(completions, ground_truths):
    """1.0 if \boxed{} answer matches ground truth, 0.0 otherwise"""
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        extracted = extract_last_boxed(completion)
        if matches(extracted, gt):  # case-insensitive or numeric 1e-2
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def format_reward(completions):
    """0.5 if response has <think> and \boxed{}, 0.0 otherwise"""
    rewards = []
    for c in completions:
        has_think = "<think>" in c and "</think>" in c
        has_boxed = "\\boxed{" in c
        rewards.append(0.5 if (has_think and has_boxed) else 0.0)
    return rewards

grpo_config = GRPOConfig(
    learning_rate=5e-6,         # Much lower than SFT
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=8,           # Group size for GRPO
    max_prompt_length=1024,
    max_completion_length=3072,
    max_steps=300,               # Start seeing improvements at ~150 steps
    warmup_ratio=0.1,
    weight_decay=0.01,
    temperature=1.0,             # For generation diversity
    bf16=True,
    report_to="wandb",
)

trainer = GRPOTrainer(
    model=model,
    config=grpo_config,
    reward_funcs=[accuracy_reward, format_reward],
    train_dataset=grpo_dataset,  # Problems with known answers
)
trainer.train()
```

### Phase 3: DPO/Step-DPO Refinement (Optional, +0.01-0.03)

After SFT (and optionally GRPO), generate preference pairs:

```python
# 1. Generate predictions on training set with trained model
for problem in validation_set:
    solutions = generate_n_solutions(problem, n=8, model=sft_model, temp=0.7)
    correct = [s for s in solutions if is_correct(s, problem.answer)]
    incorrect = [s for s in solutions if not is_correct(s, problem.answer)]
    if correct and incorrect:
        preference_pairs.append({
            "prompt": problem.prompt,
            "chosen": random.choice(correct),
            "rejected": random.choice(incorrect),
        })

# 2. Train DPO
from trl import DPOTrainer, DPOConfig
dpo_config = DPOConfig(
    learning_rate=5e-7,  # Very low for DPO
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    beta=0.1,  # KL penalty
    bf16=True,
)
dpo_trainer = DPOTrainer(model=model, args=dpo_config, train_dataset=preference_data)
dpo_trainer.train()
```

---

## 9. GRPO Reinforcement Learning Details

### Why GRPO?
- GRPO achieved 0.755 greedy accuracy and 0.885 self-consistency on 14B Qwen2.5
- +56.4% gain in faithfulness vs DPO at larger scales
- Works well with LoRA (Tina paper: $9 cost for 20%+ reasoning improvement)
- Natural fit: we have verifiable puzzles with known answers = perfect reward signal

### GRPO Hyperparameters (from Tina + TRL research)
| Param | Value | Notes |
|-------|-------|-------|
| Learning rate | 5e-6 | Much lower than SFT |
| Group size | 8 | 4-16 recommended, 8 is good balance |
| Max steps | 300-500 | Improvements start at ~150 steps |
| Temperature | 1.0 | For diverse generation |
| Max completion length | 3072 | Leave room for long reasoning |
| Batch size | 1 | With grad accum = 8 |
| LoRA rank | 16 | Tina found rank 16 sweet spot for GRPO |
| Epsilon | 0.2 | PPO-style clipping |
| Warmup | 10% | Higher for RL stability |

### Reward Function Design
```python
def combined_reward(completions, problems):
    rewards = []
    for completion, problem in zip(completions, problems):
        score = 0.0

        # Accuracy: primary reward (0 or 1)
        answer = extract_last_boxed(completion)
        if answer and matches(answer, problem["answer"]):
            score += 1.0

        # Format: secondary reward (0 or 0.3)
        if "<think>" in completion and "</think>" in completion and "\\boxed{" in completion:
            score += 0.3

        # Length penalty: discourage excessively short reasoning
        think_content = extract_think_content(completion)
        if think_content and len(think_content) > 50:
            score += 0.1

        rewards.append(score)
    return rewards
```

### GRPO Data: Use Competition Puzzles
- The 6 puzzle types from train.csv are PERFECT for GRPO
- Known correct answers = deterministic reward signal
- Can also use Reasoning Gym / Enigmata puzzles (procedurally generated, with verifiers)
- Mix: 70% competition puzzles + 30% Reasoning Gym / Enigmata puzzles

### Important: GRPO VRAM Requirements
- With LoRA 16-bit: ~60GB VRAM (fits RTX Pro 6000 48GB only with 4-bit quantization)
- With QLoRA 4-bit: ~30GB VRAM (fits comfortably)
- Unsloth provides 80% VRAM savings for GRPO

---

## 10. DPO / Step-DPO Refinement

### Step-DPO (most effective for reasoning)
From the Step-DPO paper: treats individual reasoning STEPS as preference units, not whole answers.

**Pipeline:**
1. Error collection: Generate solutions, identify wrong ones
2. Step localization: Find the FIRST incorrect step in the wrong solution
3. Rectification: Generate correct continuation from that step
4. Preference pair: (correct continuation, incorrect continuation) at the step level

**Results:** Qwen2-72B-Instruct + Step-DPO achieved 70.8% on MATH and 94.0% on GSM8K

### Iterative RPO (Iterative Reasoning Preference Optimization)
- Each iteration: sample multiple CoT, create preference pairs (correct=winner, wrong=loser)
- Train DPO variant with NLL loss term
- Performance improves over multiple iterations until saturation
- Especially effective when combined with SFT initialization

---

## 11. Hidden Test Benchmark Preparation

### GPQA Diamond (Graduate-Level Science)
- 198 questions in biology, physics, chemistry
- Multiple choice with 4 options (25% random baseline)
- PhD experts score 69.7%
- Format: "ANSWER: LETTER"
- **Training approach**: Include 400 GPQA-style questions in SFT data with `<think>` reasoning and `\boxed{letter}` format

### IFEval (Instruction Following)
- 500 prompts with verifiable instructions
- Types: word count constraints, keyword inclusion, format requirements (JSON, lists), capitalization rules
- Format: direct compliance, no `\boxed{}` needed
- **Training approach**: Include 300-500 IFEval-style examples. The model needs to follow format constraints exactly.

### Reasoning Gym (Procedural Puzzles)
- 100+ puzzle types across algebra, arithmetic, logic, graphs, games
- Procedurally generated with adjustable difficulty
- Binary or graded correctness verifiers
- Types overlap with competition puzzles: bit operations, cipher, numeral systems
- **Training approach**: Generate 500-1000 examples from Reasoning Gym generators at medium difficulty

### Math (AIME-level)
- Olympiad-level competition math
- Requires multi-step algebraic/geometric/combinatorial reasoning
- **Training approach**: Cherry-pick 2000-3000 problems from OpenMathReasoning at varied difficulty

### Coding (LiveCodeBench-style)
- Algorithm implementation, bug finding, code reasoning
- **Training approach**: Include 500 code reasoning problems where the answer is a number or string extractable as `\boxed{}`

---

## 12. Hyperparameters

### SFT Phase
| Param | Value | Rationale |
|-------|-------|-----------|
| Learning rate | 1e-4 | Compensates for alpha=64 (effective LR = LR * alpha/rank = 1e-4 * 2 = 2e-4) |
| Epochs | 3 | For ~21K examples. Research shows 1.5 epochs optimal for large datasets, 3 for medium |
| Effective batch size | 16 | batch=1 * grad_accum=16 |
| Max seq length | 4096 | Prevents truncation. Eval uses 8192 but 4096 covers most reasoning |
| LoRA rank | 32 | Maximum allowed |
| LoRA alpha | 64 | 2x rank, empirically optimal |
| Warmup | 5% | Standard for SFT |
| Weight decay | 0.01 | Light regularization |
| LR scheduler | cosine | Smooth decay |
| Gradient checkpointing | True | Required for memory |
| bf16 | True | Required for Blackwell |
| use_reentrant | False | Required for Mamba layers |

### GRPO Phase
| Param | Value | Rationale |
|-------|-------|-----------|
| Learning rate | 5e-6 | 20x lower than SFT |
| Group size | 8 | Balance diversity vs compute |
| Max steps | 300-500 | Improvements visible at 150 |
| LoRA rank | 32 (same adapter) | Continue from SFT |
| Temperature | 1.0 | For diverse generation |
| Max completion length | 3072 | Long reasoning allowed |
| KL penalty (beta) | 0.04 | Light constraint to base |

### DPO Phase (if used)
| Param | Value | Rationale |
|-------|-------|-----------|
| Learning rate | 5e-7 | Very conservative |
| Epochs | 1 | Avoid overfit |
| Beta | 0.1 | Standard KL penalty |
| Preference pairs | 2000-5000 | Generated from SFT model |

---

## 13. Common Mistakes to Avoid

### Data Mistakes
1. **Training only on 6 puzzle types** -- overfits to 30-50% of test. Must add broader reasoning data
2. **Truncating \boxed{} during training** -- if MAX_SEQ_LEN too short, the answer gets cut off. Use 4096
3. **Poor CoT quality** -- hand-crafted regex traces << teacher-distilled traces
4. **No rejection sampling** -- training on wrong answers teaches wrong patterns
5. **Wrong data ratio** -- too much puzzle data, not enough general reasoning. Target 75/25 reasoning/direct

### LoRA Mistakes
6. **Missing QKV attention targets** -- the 6 GQA layers are critical for reasoning
7. **Targeting router layers** -- collapses expert specialization, catastrophic
8. **Alpha too low** -- alpha=rank is conservative. alpha=2*rank works better
9. **LoRA on SSM core modules** -- ineffective, waste of parameters

### Training Mistakes
10. **LR too high with high alpha** -- effective LR = LR * alpha/rank. If alpha doubled, halve LR
11. **No gradient checkpointing** -- OOM guaranteed
12. **use_reentrant=True** -- breaks Mamba layers on Blackwell
13. **Forgetting Blackwell patches** -- is_fast_path_available, rmsnorm_fn, ptxas

### Format Mistakes
14. **Wrong chat template** -- must use `<|im_start|>` / `<|im_end|>`, NOT `<s>` / `</s>`
15. **Missing enable_thinking** -- eval uses enable_thinking=True, training must match
16. **Wrong \boxed{} format** -- must be `\boxed{answer}` not `\boxed answer` or `[answer]`

### Submission Mistakes
17. **Submitting untested code** -- always verify training completes before submitting
18. **GPU resets after error** -- always verify GPU setting in Kaggle UI before every run
19. **CLI push runs on P100** -- must switch to RTX Pro 6000 in UI after every push
20. **wandb blocking** -- use try/except fallback to offline mode

---

## 14. Estimated Score Progression

### Conservative Estimates
| Step | Score | Delta | Cumulative Improvement |
|------|-------|-------|----------------------|
| Current (SFT v5, 2000 examples) | 0.69 | - | - |
| + Teacher distillation (9500 examples) | 0.73 | +0.04 | +0.04 |
| + Broader reasoning data (+12K examples) | 0.78 | +0.05 | +0.09 |
| + LoRA config fix (alpha=64, +QKV) | 0.80 | +0.02 | +0.11 |
| + Training config (seq_len=4096) | 0.81 | +0.01 | +0.12 |
| + GRPO reinforcement learning | 0.85 | +0.04 | +0.16 |
| + DPO refinement | 0.87 | +0.02 | +0.18 |
| + Per-category error analysis + targeted fix | 0.90 | +0.03 | +0.21 |

### Optimistic Estimates (if data quality is exceptional)
| Step | Score | Notes |
|------|-------|-------|
| SFT on 21K quality examples | 0.80 | Data quality is the biggest lever |
| + GRPO with puzzle verifiers | 0.86 | RL excels when rewards are clear |
| + Iterative DPO | 0.88 | Refinement pass |
| + Category-specific fine-tuning | 0.90+ | Target weakest areas |

### Key Insight: Data Quality > Method
NemoSkills won AIMO-2 with **SFT alone** because the data was exceptional (5.5M curated samples, rejection-sampled, teacher-distilled). We may not need GRPO at all if our SFT data is good enough. Prioritize data preparation over method sophistication.

---

## 15. Sources

### Competition & Model
- [NVIDIA Nemotron Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
- [Nemotron-3-Nano-30B HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [Nemotron-3-Nano Chat Template](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/chat_template.jinja)
- [Nemotron-3-Nano Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)
- [vLLM Nemotron-3-Nano Guide](https://docs.vllm.ai/projects/recipes/en/latest/NVIDIA/Nemotron-3-Nano-30B-A3B.html)
- [vLLM Nemotron Cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano/vllm_cookbook.ipynb)

### Winning Solutions & Techniques
- [AIMO-2 1st Place (NemoSkills) - arXiv:2504.16891](https://arxiv.org/abs/2504.16891)
- [OpenMathReasoning Dataset](https://huggingface.co/datasets/nvidia/OpenMathReasoning)
- [Nemotron-Cascade 2 - arXiv:2603.19220](https://arxiv.org/abs/2603.19220)
- [Nemotron-Cascade 2 SFT Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data)
- [Nemotron-Cascade 2 Research Page](https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/)

### Training Methods
- [GRPO Trainer (TRL)](https://huggingface.co/docs/trl/grpo_trainer)
- [Implementing GRPO in TRL](https://huggingface.co/docs/course/en/chapter12/4)
- [HuggingFace GRPO Cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl)
- [Tina: Tiny Reasoning Models via LoRA - arXiv:2504.15777](https://arxiv.org/abs/2504.15777) (ICLR 2026)
- [Step-DPO - arXiv:2406.18629](https://arxiv.org/abs/2406.18629)
- [Evaluating GRPO and DPO for Reasoning](https://arxiv.org/abs/2512.22631)
- [Two-Stage SFT + GRPO Pipeline](https://langcopilot.com/posts/2025-09-05-grpo-training-pipeline-sft-rl-better)
- [Unsloth RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)

### LoRA on SSM/MoE Models
- [Parameter-Efficient Fine-Tuning of State Space Models](https://arxiv.org/abs/2410.09016) (ICML 2025)
- [MoLA: MoE LoRA with Layer-wise Expert Allocation](https://aclanthology.org/2025.findings-naacl.284/)
- [HELLoRA: Hot Experts Layer-level LoRA](https://openreview.net/forum?id=CsHahbRAFZ) (ICLR 2026)
- [Unsloth Nemotron-3 Guide](https://unsloth.ai/docs/models/nemotron-3)

### Datasets
- [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) - 306K math problems, 3.2M CoT solutions
- [nvidia/Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) - 30M+ samples
- [nvidia/Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) - EXACT base model training data

### Puzzle / Reasoning Benchmarks
- [Enigmata: Synthetic Verifiable Puzzles](https://seed-enigmata.github.io/) - arXiv:2505.19914
- [Reasoning Gym - arXiv:2505.24760](https://arxiv.org/abs/2505.24760)
- [GPQA Diamond](https://epoch.ai/benchmarks/gpqa-diamond)
- [IFEval - arXiv:2311.07911](https://arxiv.org/abs/2311.07911)

### Catastrophic Forgetting & Data Mix
- [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)
- [STaR: Self-Taught Reasoner](https://openreview.net/pdf?id=_3ELRdg2sgI)
- [AdaSTaR: Adaptive Data Sampling](https://arxiv.org/abs/2505.16322)

---

## Appendix A: Quick-Start Implementation Checklist

### Day 1: Data Preparation
- [ ] Generate teacher CoT for all 9,500 competition examples using DeepSeek-R1 API
- [ ] Apply rejection sampling (4 solutions per problem, keep correct ones)
- [ ] Download subsets from nvidia/Nemotron-Cascade-2-SFT-Data (math, science)
- [ ] Download 2000 examples from nvidia/OpenMathReasoning
- [ ] Format all data with proper chat template
- [ ] Total target: ~21,000 examples

### Day 2: SFT Training Run
- [ ] Update LoRA config: alpha=64, add QKV targets, add gate_proj
- [ ] Set MAX_SEQ_LEN=4096
- [ ] Set LR=1e-4, cosine schedule
- [ ] Apply Blackwell patches
- [ ] Run SFT on full dataset (3 epochs)
- [ ] Submit and score

### Day 3: Iterate Based on Score
- [ ] If score < 0.80: Focus on data quality, check which categories fail
- [ ] If score >= 0.80: Consider GRPO reinforcement learning
- [ ] Generate preference pairs from SFT model
- [ ] Run DPO refinement (1 epoch)
- [ ] Submit and score

### Day 4-7: GRPO + Category Analysis
- [ ] Set up GRPO with puzzle verifier rewards
- [ ] Train 300-500 steps
- [ ] Analyze per-category accuracy
- [ ] Generate targeted data for weakest categories
- [ ] Iterate until 0.90

---

## Appendix B: Nemotron-Cascade-2 as Direct Uplift

### Key Discovery (March 22, 2026)
Nemotron-Cascade-2-30B-A3B was released March 20, 2026 -- just 2 days ago. It uses the EXACT same base model as the competition (Nemotron-3-Nano-30B-A3B-Base). This means:

1. The Cascade-2 SFT data is PERFECTLY formatted for our base model
2. The Cascade-2 training recipe represents NVIDIA's own best practices for this architecture
3. The data covers math, science, coding, instruction following -- exactly the hidden test domains

### Strategy: Use Cascade-2 Data as Primary SFT Data
Instead of generating our own CoT from scratch, we could:
1. Use Cascade-2 SFT data for broad reasoning capability
2. Add competition-specific puzzle data with teacher CoT
3. This gives us the best of both worlds: broad generalization + puzzle specialization

### Estimated Impact
Using high-quality, model-specific training data from the same architecture family could provide +0.05-0.10 improvement over generic training data.
