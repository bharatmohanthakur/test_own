# Research: GRPO Alternatives for Mamba-Transformer MoE
# Date: 2026-03-23
# Problem: GRPO needs online generation, vLLM can't load mamba_ssm, generation is 100x too slow without vLLM

## Executive Summary

**FASTEST PATH from 0.69 to 0.75+ WITHOUT online generation:**

1. **Rejection Sampling Fine-Tuning (RFT/RAFT)** — Generate solutions OFFLINE with an API (DeepSeek-R1), filter correct ones, SFT on them. No generation during training. Expected: +0.04-0.06
2. **STaR/TPT (iterative SFT)** — Same idea but iterate: train model, generate with it, keep correct, retrain. Expected: +0.03-0.05 per iteration
3. **DPO/SimPO with self-generated pairs** — After RFT, generate pairs offline, run DPO. No online generation needed. Expected: +0.01-0.03
4. **KTO** — Even simpler than DPO, only needs "good/bad" labels, no pairs. Expected: +0.01-0.02

**Key insight from research**: A simple rejection sampling baseline (RAFT) yields *competitive performance with GRPO and PPO*. GRPO's main advantage is just filtering — which RAFT does explicitly. We don't need GRPO at all.

---

## Method-by-Method Analysis

### 1. Rejection Sampling Fine-Tuning (RFT/RAFT) — RECOMMENDED FIRST

**What it is:** Generate N solutions per problem, keep only correct ones, SFT on the correct solutions.

**Does it require generation during training?** NO. Generation is completely decoupled from training. Generate offline (via API or separate GPU), then train on filtered data.

**Memory requirements:** Same as SFT. No reference model, no reward model, no value function.

**Implementation:**
```python
# Step 1: OFFLINE — Generate solutions via API (NOT during training)
import openai  # or DeepSeek API
solutions = []
for problem in training_data:
    for _ in range(8):  # generate 8 solutions per problem
        response = api.generate(problem.prompt)
        if extract_boxed(response) == problem.answer:
            solutions.append({"prompt": problem.prompt, "response": response})

# Step 2: SFT on correct solutions only (standard HF Trainer)
from trl import SFTTrainer
trainer = SFTTrainer(model=model, train_dataset=solutions, ...)
trainer.train()
```

**Expected improvement over SFT:** +3-6% accuracy. Research shows RFT improvement is log-linear in the number of distinct correct CoTs per problem.

**How to combine with existing adapter:** Load v7 adapter, then continue SFT on RFT-filtered data. Or train fresh from base model on higher-quality data.

**Why this is the best option:**
- Paper "A Minimalist Approach to LLM Reasoning" (Apr 2025) showed RAFT is competitive with GRPO
- GRPO's main advantage = filtering prompts with all-incorrect responses. RAFT does this explicitly.
- DeepSeekMath used RFT to go from 35.9% to 49.3% on GSM8K (LLaMA-7B)
- PCL-Reasoner-V1.5 used offline RL (SFT then offline RL) to hit 90.9% on AIME 2024

**Practical plan for our competition:**
1. Use DeepSeek-R1 API ($5-10) to generate 8 solutions per problem for all 9500 training examples
2. Filter: keep only solutions where extracted \boxed{} matches ground truth
3. Deduplicate: keep diverse reasoning paths (not just repeated solutions)
4. SFT on ~15K-30K high-quality, verified correct CoT traces
5. This gives us MUCH better training data than hand-crafted regex CoTs

### 2. STaR / TPT (Iterative Self-Training) — RECOMMENDED SECOND

**What it is:** Generate solutions with current model, keep correct ones, retrain. Repeat.

**Does it require generation during training?** NO — generation happens BETWEEN training rounds, not during. But it does need the model to generate, which is slow without vLLM.

**Workaround for slow generation:** Use the trained model via HuggingFace generate() with batching. Even at 10 tok/s, generating 8 solutions x 9500 problems x ~500 tokens = ~100 hours. TOO SLOW on Kaggle.

**Better approach:** STaR with API-generated data. Use DeepSeek-R1 for round 1, then after training, use the NEW model for round 2 (if we can generate fast enough).

**TPT (Think, Prune, Train) specifics:**
- Stanford paper (Apr 2025) showed impressive results without RL
- Gemma2-9B: 41.9% -> 82% on GSM8K (matching LLaMA-3.1-70B!)
- Key: use ONE correct solution per problem per round (replace, don't accumulate)
- 3-5 iterations typically sufficient

**Memory requirements:** Same as SFT per round.

**Implementation:**
```python
# Round 1: Generate with current model
for problem in problems:
    outputs = model.generate(problem.prompt, num_return_sequences=8, temperature=0.7)
    correct = [o for o in outputs if extract_answer(o) == problem.answer]
    if correct:
        selected.append(random.choice(correct))  # keep 1 diverse correct solution
    else:
        # "Rationalization" — try again with answer hint
        hint_prompt = f"{problem.prompt}\nThe answer is {problem.answer}. Show your reasoning."
        rationalized = model.generate(hint_prompt, num_return_sequences=4)
        correct_r = [r for r in rationalized if extract_answer(r) == problem.answer]
        if correct_r:
            selected.append(random.choice(correct_r))

# Round 1: SFT on selected correct solutions
trainer = SFTTrainer(model=model, train_dataset=selected, ...)
trainer.train()
# Repeat for rounds 2, 3, ...
```

**Expected improvement:** +3-5% per iteration, diminishing returns after 3-5 rounds.

**Practical concern:** Generation speed. Without vLLM, each STaR iteration could take 10+ hours of GPU just for generation. Use API instead.

### 3. DPO (Direct Preference Optimization) — GOOD POST-SFT REFINEMENT

**What it is:** Train on preference pairs (chosen vs rejected) without a reward model.

**Does it require generation during training?** NO. Pairs are prepared offline.

**Memory requirements:** 2x model in memory (policy + reference model). With LoRA: policy is LoRA, reference is frozen base = ~1.2x base memory. FITS on RTX Pro 6000 (48GB).

**How to generate preference pairs for our task:**
```python
# OFFLINE: Generate multiple solutions per problem
for problem in training_data:
    solutions = model.generate(problem.prompt, num_return_sequences=8, temp=0.7)
    correct = [s for s in solutions if extract_boxed(s) == problem.answer]
    incorrect = [s for s in solutions if extract_boxed(s) != problem.answer]
    if correct and incorrect:
        pairs.append({
            "prompt": problem.prompt,
            "chosen": random.choice(correct),
            "rejected": random.choice(incorrect),
        })
```

**TRL Implementation:**
```python
from trl import DPOConfig, DPOTrainer
training_args = DPOConfig(
    output_dir="dpo_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,  # much lower LR for DPO
    num_train_epochs=1,
    bf16=True,
    loss_type="sigmoid",  # standard DPO
    # For SimPO variant (reference-free, saves memory):
    # loss_type="simpo", cpo_alpha=0.05
)
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # uses implicit reference with LoRA
    args=training_args,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)
trainer.train()
```

**Expected improvement over SFT:** +1-3% accuracy. AIMO-2 2nd place used DPO after SFT for +2-3%.

**Key concern for Mamba-Transformer:** DPO with LoRA uses implicit reference (frozen base weights). This should work fine — the DPO loss computes log-probs of chosen/rejected under policy and reference. No generation needed during training.

**Limitation:** DPO treats all tokens uniformly — bad for multi-step reasoning where early steps matter most. Step-DPO exists but is more complex to implement.

### 4. SimPO (Simple Preference Optimization) — BEST DPO VARIANT

**What it is:** Reference-free DPO variant. Uses average log probability as implicit reward. No reference model needed.

**Does it require generation during training?** NO.

**Memory requirements:** LESS than DPO — no reference model at all. Same memory as SFT.

**Implementation in TRL:**
```python
from trl import CPOConfig, CPOTrainer  # SimPO is implemented via CPOTrainer
training_args = CPOConfig(
    output_dir="simpo_output",
    loss_type="simpo",
    cpo_alpha=0.05,  # length penalty
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    bf16=True,
)
trainer = CPOTrainer(
    model=model,
    args=training_args,
    train_dataset=preference_dataset,  # same format as DPO
    tokenizer=tokenizer,
    peft_config=lora_config,
)
trainer.train()
```

**Expected improvement:** +1-3% over SFT. SimPO outperformed DPO by 6.4 points on AlpacaEval 2.

**Advantage:** No reference model = less memory, simpler, faster training. Perfect for our memory-constrained setup.

### 5. KTO (Kahneman-Tversky Optimization) — SIMPLEST PREFERENCE METHOD

**What it is:** Only needs binary labels (good/bad), not pairs. Based on prospect theory.

**Does it require generation during training?** NO.

**Memory requirements:** Needs reference model (like DPO), but with LoRA the base model serves as implicit reference. Same as DPO memory.

**Data format:**
```python
# KTO only needs: prompt + response + label (True/False)
dataset = [
    {"prompt": "...", "completion": "correct solution with boxed{}", "label": True},
    {"prompt": "...", "completion": "incorrect solution with wrong boxed{}", "label": False},
]
```

**Implementation in TRL:**
```python
from trl import KTOConfig, KTOTrainer
training_args = KTOConfig(
    output_dir="kto_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    bf16=True,
    desirable_weight=1.0,
    undesirable_weight=1.0,
)
trainer = KTOTrainer(
    model=model,
    ref_model=None,  # implicit with LoRA
    args=training_args,
    train_dataset=kto_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)
trainer.train()
```

**Expected improvement:** Comparable to DPO (+1-3%). Works well at scales from 1B to 30B.

**Advantage over DPO:** Don't need matched pairs. Can use any collection of good and bad responses, even from different prompts. Easier data collection.

### 6. OREO (Offline Reasoning Optimization) — MOST THEORETICALLY SOUND

**What it is:** Offline RL that jointly learns policy + value function using soft Bellman Equation. Better credit assignment for multi-step reasoning.

**Does it require generation during training?** NO. Fully offline.

**Memory requirements:** Higher — needs value function head in addition to policy. ~1.5x SFT.

**Expected improvement:** +5.2% on GSM8K, +10.5% on MATH over SFT (1.5B model).

**Implementation:** Not in TRL. Custom implementation needed (paper code available on GitHub).

**Practical concern:** More complex to implement, may not be worth it vs simpler methods given our time constraints.

---

## Comparison Table

| Method | Online Gen? | Memory vs SFT | TRL Support? | Expected Gain | Complexity |
|--------|-------------|---------------|--------------|---------------|------------|
| **RFT/RAFT** | NO (offline API) | 1.0x | SFTTrainer | +3-6% | LOW |
| **STaR/TPT** | Between rounds | 1.0x | SFTTrainer | +3-5%/iter | MEDIUM |
| **DPO** | NO | 1.2x (ref model) | DPOTrainer | +1-3% | LOW |
| **SimPO** | NO | 1.0x (no ref) | CPOTrainer | +1-3% | LOW |
| **KTO** | NO | 1.2x (ref model) | KTOTrainer | +1-3% | LOW |
| **OREO** | NO | 1.5x | Custom | +3-5% | HIGH |
| GRPO | YES (bottleneck) | 1.3x | GRPOTrainer | +3-8% | HIGH |
| RLOO | YES (bottleneck) | 1.1x | RLOOTrainer | +3-6% | MEDIUM |

---

## RECOMMENDED PIPELINE (Priority Order)

### Phase 1: Better Data via Rejection Sampling (Days 1-2)
**Expected: 0.69 -> 0.73-0.75**

1. Use DeepSeek-R1 API to generate 8 solutions per problem for all 9500 training examples
   - Cost: ~$5-10
   - Time: few hours via API
   - Filter: keep only solutions where \boxed{answer} matches ground truth
   - Also generate for broader reasoning data (math, code, science — 3000-5000 extra)
2. SFT on the filtered, high-quality CoT traces
   - Use ALL 9500 problems (not just 2000)
   - Include 3000-5000 broader reasoning examples
   - Total: ~12K-15K high-quality examples
3. Training config: rank=32, alpha=64, all projection targets, seq_len=4096, 3 epochs

### Phase 2: Preference Optimization (Day 3)
**Expected: 0.73-0.75 -> 0.76-0.78**

1. From Phase 1 model, generate 8 solutions per problem (use model.generate, batch size 4, temperature 0.7)
   - If too slow on Kaggle (>8 hrs), do on Vast.ai or use API
2. Create preference pairs:
   - Chosen: correct solutions (matching ground truth)
   - Rejected: incorrect solutions
   - Expect ~60-70% of problems to have both correct and incorrect solutions
3. Run SimPO (best option — no reference model, less memory):
   ```
   CPOTrainer with loss_type="simpo"
   LR: 5e-6, 1 epoch, batch=16
   ```
4. Alternative: If SimPO doesn't work with mamba_ssm, try KTO (only needs good/bad labels, no pairs)

### Phase 3: Iterative Refinement (Days 4+)
**Expected: 0.76-0.78 -> 0.80+**

1. Evaluate Phase 2 model on training set to find weak categories
2. Generate more data for weak categories (via API)
3. Second round of RFT + SimPO
4. Repeat until diminishing returns

---

## Critical Compatibility Notes for Nemotron-3-Nano (Mamba-Transformer MoE)

### What WORKS:
- SFT with LoRA (proven, our v5/v7 scores confirm)
- DPOTrainer with LoRA (implicit reference = frozen base, no extra model load)
- CPOTrainer/SimPO (reference-free, even simpler)
- KTOTrainer with LoRA (same mechanism as DPO)
- Offline data generation via API (completely decoupled)

### What MIGHT have issues:
- TRL trainers calling model internals that assume standard Transformer
- Log-probability computation for DPO/SimPO (need to verify mamba_ssm forward pass returns proper logits)
- Gradient computation through Mamba blocks for preference losses

### What DEFINITELY WON'T work:
- GRPO/RLOO/PPO (need online generation, which needs vLLM, which can't load mamba_ssm)
- Any method requiring vLLM inference during training
- Methods that need reward model inference during training (PPO)

### Mitigation for compatibility risks:
- Test DPO/SimPO on a small subset (100 examples) first before full run
- If DPO forward pass fails on Mamba blocks, fall back to pure RFT (which is just SFT on better data)
- All Blackwell fixes (causal_conv1d patch, rmsnorm_fn, ptxas) still apply

---

## Key Papers & Sources

- [A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce](https://arxiv.org/abs/2504.11343) — RAFT competitive with GRPO
- [Think, Prune, Train, Improve](https://arxiv.org/abs/2504.18116) — STaR without RL, Gemma2-9B matches 70B
- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — Original iterative self-training
- [SimPO: Simple Preference Optimization](https://arxiv.org/abs/2405.14734) — Reference-free DPO, NeurIPS 2024
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — Binary feedback alignment
- [OREO: Offline Reasoning Optimization](https://arxiv.org/abs/2412.16145) — Offline RL for multi-step reasoning, ACL 2025
- [PCL-Reasoner-V1.5](https://arxiv.org/abs/2601.14716) — Offline RL achieves 90.9% on AIME 2024
- [RAFT: Reward rAnked FineTuning](https://arxiv.org/abs/2304.06767) — Original rejection sampling framework
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — RFT for math reasoning
- [SFT Memorizes, RL Generalizes](https://arxiv.org/abs/2501.17161) — Why RL beats SFT
- [Self-Evolved Preference Optimization](https://arxiv.org/abs/2503.04813) — Self-generated DPO pairs for math
- [TRL DPOTrainer docs](https://huggingface.co/docs/trl/en/dpo_trainer)
- [TRL KTOTrainer](https://huggingface.co/docs/trl/index)
- [RLHFlow/Minimal-RL](https://github.com/RLHFlow/Minimal-RL) — Minimal RL implementations

---

## Bottom Line

**We do NOT need GRPO.** The research clearly shows:

1. **RAFT (rejection sampling + SFT) is competitive with GRPO** — the main benefit of GRPO is just filtering, which RAFT does explicitly
2. **The #1 lever is data quality** — better CoT traces from a stronger teacher model (DeepSeek-R1) will help more than any training algorithm change
3. **SimPO is the best post-SFT refinement** — no reference model, less memory, outperforms DPO
4. **Our bottleneck is generation speed, not training algorithm** — solve this by using APIs for data generation, keeping training purely offline

**Concrete next step:** Generate high-quality CoT traces for all 9500 problems using DeepSeek-R1 API, filter for correctness, SFT on the result. This alone should get us from 0.69 to 0.73-0.75.
