# RL Techniques for Reasoning Models — Research Report (2026-03-23)

## Executive Summary

Researched 10 categories of RL techniques for improving reasoning in LLMs. The field has evolved rapidly since DeepSeek-R1-Zero. Key finding: **for our Nemotron-3-Nano-30B (Mamba-Transformer MoE hybrid) with LoRA rank <= 32, the most practical path is either (1) offline/semi-online GRPO with Dr. GRPO fixes, or (2) REINFORCE++ which is simpler and more stable than GRPO.** vLLM support for Mamba/SSM is improving but still fragile. The M1 paper proves GRPO works on Mamba reasoning models via VeRL framework.

---

## 1. GRPO — Latest Implementations and Tricks (GRPO++)

### What's New
The community has identified critical fixes to vanilla GRPO that collectively form "GRPO++" or get folded into DAPO:

**Per-Token Loss (Critical Fix)**
- Original GRPO normalizes loss per-sequence-length, creating bias where shorter correct answers get disproportionately large gradients
- Fix: calculate policy gradient loss across ALL tokens in ALL samples, not averaging per sample
- Implementation: set `loss_type="per_token"` in TRL GRPOConfig (or `loss_agg_mode="seq-mean-token-sum-norm"` in VeRL)

**Dynamic Sampling**
- Skip prompts where ALL completions are correct (0% or 100% success rate) — these provide zero learning signal
- As training progresses, more prompts get solved perfectly, wasting compute
- Dynamically oversample harder prompts that have mixed success rates

**Decoupled Clipping (from DAPO)**
- Standard PPO/GRPO clips ratio in [1-eps, 1+eps] with eps=0.2
- DAPO decouples: eps_low=0.2, eps_high=0.28 — allows more exploration upward
- Prevents entropy collapse and improves performance

**Remove KL Penalty**
- Without a separate reward model to over-optimize against, KL divergence penalty is unnecessary
- Removing it helps learning — the model can explore more freely
- DeepSeek-R1-Zero already demonstrated this works

**Overlong Reward Shaping**
- Truncated responses (hit max_length) get masked from loss entirely
- Or apply soft length penalty to discourage excessively long reasoning chains
- Prevents training on incomplete/corrupted reasoning

### Assessment for Our Setup
- **Mamba/MoE compatible?** Yes, these are loss-level tricks, architecture-agnostic
- **Memory:** Same as vanilla GRPO
- **LoRA adapter?** Yes, works with any PEFT method
- **Expected improvement:** +2-5% over vanilla GRPO based on AIME benchmarks
- **Difficulty:** Low — mostly config changes in TRL or VeRL

Sources:
- [GRPO++ Tricks](https://cameronrwolfe.substack.com/p/grpo-tricks)
- [AReaL GRPO Algorithms](https://inclusionai.github.io/AReaL/en/algorithms/grpo_series.html)
- [DRA-GRPO](https://arxiv.org/html/2505.09655v4)

---

## 2. DAPO — Decoupled Clip and Dynamic sAmpling Policy Optimization

### What It Is
DAPO = "GRPO with all the lessons learned from running it at scale" (ByteDance Seed team).

### Key Results
- **50 points on AIME 2024** using Qwen2.5-32B base model (vs 30% with vanilla GRPO)
- Outperforms DeepSeek-R1-Zero-Qwen-32B (47 points) using **50% fewer training steps**
- DAPO is now supported in NeMo RL (the framework NVIDIA uses for Nemotron)

### Four Pillars
1. **Clip Higher** — Decoupled clipping with eps_low=0.2, eps_high=0.28
2. **Dynamic Sampling** — Filter prompts with uniform success/failure
3. **Token-Level Loss** — Per-token rather than per-sequence loss
4. **Overlong Reward Shaping** — Mask truncated completions

### Important Caveat
Best results achieved with Dynamic Sampling (DS) **disabled**. The other 3 pillars are the real winners.

### Assessment for Our Setup
- **Mamba/MoE compatible?** Yes
- **Memory:** Same as GRPO (critic-free)
- **LoRA adapter?** Yes
- **Expected improvement:** Significant over vanilla GRPO (+20% on AIME for 32B models)
- **Difficulty:** Medium — need TRL or NeMo RL with DAPO config
- **NeMo RL supports DAPO natively** — this is what NVIDIA used for Nemotron-3-Super

Sources:
- [DAPO Paper](https://arxiv.org/abs/2503.14476)
- [Post-Training in 2026](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [DAPO: Enhancing GRPO](https://aipapersacademy.com/dapo/)
- [Comparative Analysis PPO vs GRPO vs DAPO](https://arxiv.org/abs/2512.07611)

---

## 3. Dr. GRPO — Fixing Length Bias

### The Problem
GRPO has a fundamental bias: normalizing loss by sequence length causes longer incorrect responses to be under-penalized. This makes responses grow longer over training, especially wrong ones.

### The Fix (Two Changes)
1. **Normalize summed loss by a fixed constant** (max_completion_length) instead of actual sequence length
2. **Remove standard deviation from advantage normalization** — use only mean-centering

### Implementation
- TRL: Set `loss_type="dr_grpo"` in GRPOConfig
- VeRL: Set `loss_agg_mode="seq-mean-token-sum-norm"` and `norm_adv_by_std_in_grpo=False`

### Further Refinement: LOOP
- Dr. GRPO still has residual bias for small group sizes
- Fix: multiply advantage by correction term N/(N-1) — called "Leave-One-Out Proximal" (LOOP)
- Simple one-line fix on top of Dr. GRPO

### Assessment for Our Setup
- **Mamba/MoE compatible?** Yes, purely a loss computation change
- **Memory:** No additional memory
- **LoRA adapter?** Yes
- **Expected improvement:** Better token efficiency (shorter, more correct responses)
- **Difficulty:** Very low — single config flag in TRL

Sources:
- [Dr. GRPO Paper](https://arxiv.org/pdf/2503.20783)
- [Dr. GRPO Emergent Mind](https://www.emergentmind.com/topics/dr-grpo)
- [LOOP correction on X](https://x.com/leloykun/status/1903382502158500119)

---

## 4. Reinforcement Learning with Mamba Models

### M1 Paper — GRPO Works on Mamba! (Key Finding)
The M1 paper from TogetherAI/Cornell/Princeton demonstrated GRPO training on Mamba reasoning models:
- **Model:** Hybrid Mamba (similar architecture to Nemotron-3-Nano)
- **Framework:** VeRL with GRPO
- **Results:** 82 on MATH500, 22 on AIME25, matches DeepSeek-R1 distilled at same scale
- **Speed:** 3x faster inference than transformers
- **Key fix:** Resolved CUDA graph incompatibility with Mamba in VeRL, making it **5x faster** with CUDA graphs enabled

### NVIDIA Nemotron-3-Super Training
- Used **asynchronous GRPO** across 21 environment configurations
- Multi-environment RL: math, code, science, instruction following, tool use
- 1.2 million environment rollouts
- Uses NeMo RL (successor to NeMo Aligner)

### vLLM + Mamba Status
- vLLM V0 supports Mamba models (Jamba, FalconMamba, etc.)
- **vLLM V1 does NOT support SSM models yet** — RFC open (issue #17140)
- SSM models incompatible with prefix caching, KV cache offloading
- Recent PyTorch blog: "Hybrid Models as First-Class Citizens in vLLM" — improving support
- **Workaround:** Use `vllm_model_impl="transformers"` backend, or disable vLLM entirely

### Assessment for Our Setup
- **Mamba/MoE compatible?** YES — M1 proves it works. VeRL has fixes.
- **Memory:** Mamba is more memory efficient than pure transformer
- **LoRA adapter?** VeRL supports LoRA with GRPO (DTensor backend)
- **Expected improvement:** Significant — M1 matches transformer reasoning at same scale
- **Difficulty:** HIGH — need VeRL setup, CUDA graph fixes, custom integration

Sources:
- [M1 Paper](https://arxiv.org/abs/2504.10449)
- [Nemotron-3 Super Blog](https://developer.nvidia.com/blog/introducing-nemotron-3-super-an-open-hybrid-mamba-transformer-moe-for-agentic-reasoning/)
- [vLLM RFC for Mamba](https://github.com/vllm-project/vllm/issues/17140)
- [Hybrid Models in vLLM](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/)

---

## 5. Competition-Specific Approaches

### AIMO-2 Winning Solution (NemoSkills — NVIDIA Team)
- Used **Qwen2.5-14B-Base** with chain-of-thought reasoning
- **Knowledge distillation** from DeepSeek-R1 and QwQ-32B
- Fine-tuned on millions of synthetically generated math solutions
- **Parallel reasoning** — multiple long-thinking responses, majority vote
- Key insight: **data quality and distillation >> RL method**

### Our Competition Constraints
- Must submit LoRA adapter (rank <= 32)
- Eval uses vLLM with temp=0.0, top_p=1.0 (single deterministic output)
- Answer extracted from `\boxed{}` — binary correct/incorrect
- Hidden test may include broader tasks beyond the 6 puzzle types

### Implication
- Binary reward (correct/incorrect \boxed{}) is a natural fit for RLVR
- Single deterministic output means we can't do test-time majority voting
- Must maximize single-pass accuracy
- RL with verifiable rewards is the ideal paradigm for this task

Sources:
- [NVIDIA Kaggle Win](https://blogs.nvidia.com/blog/reasoning-ai-math-olympiad/)
- [OpenMath-Nemotron-14B-Kaggle](https://huggingface.co/nvidia/OpenMath-Nemotron-14B-Kaggle)

---

## 6. Reward Shaping for \boxed{} Tasks

### RLVR — Reinforcement Learning with Verifiable Rewards
- Binary reward: 1 if `\boxed{answer}` matches ground truth, 0 otherwise
- No reward model needed — rule-based verification
- Proven to implicitly incentivize correct reasoning even with outcome-only rewards
- This is exactly what DeepSeek-R1-Zero used

### Reward Design Options
| Reward Type | Description | Pros | Cons |
|------------|-------------|------|------|
| Binary (0/1) | Correct answer = 1, wrong = 0 | Simple, unambiguous | Sparse signal |
| Format bonus | +0.1 for using \boxed{}, +0.1 for <think> | Encourages format compliance | May over-optimize format |
| Length penalty | -0.001 per token beyond threshold | Prevents overthinking | May truncate reasoning |
| Partial credit | 0.5 for close answers | Richer signal | Hard to define "close" |

### Recommended Reward for Our Task
```python
def reward_fn(response, ground_truth):
    # Extract from \boxed{}
    predicted = extract_boxed(response)
    if predicted is None:
        return -0.5  # Penalize missing format

    # Exact match (case-insensitive) or numerical tolerance
    if matches(predicted, ground_truth, tol=1e-2):
        return 1.0

    # Partial: has \boxed{} but wrong answer
    return -0.25
```

### Advanced: Process Supervision (GRPO-VPS)
- Probe model's belief in correct answer at each reasoning step
- Segment generation into discrete steps
- Track conditional probability of correct answer at each boundary
- **+2.6% accuracy, -13.7% response length** vs outcome-only reward
- More complex to implement but very effective

Sources:
- [RLVR Paper](https://arxiv.org/abs/2506.14245)
- [GRPO-VPS](https://openreview.net/forum?id=Ise1xvtUF6)
- [Reward Modeling Survey](https://arxiv.org/html/2602.09305v1)
- [RLVR Explained](https://www.promptfoo.dev/blog/rlvr-explained/)

---

## 7. GRPO OOM Fixes for Generation Phase

### Three Memory Bottlenecks
1. **vLLM memory reservation** — reserves GPU upfront (often 60%+ of VRAM)
2. **Training activations** — scale with batch_size * seq_len * num_generations
3. **KV cache during generation** — grows with context length

### Concrete Fixes
| Fix | Impact | How |
|-----|--------|-----|
| Reduce num_generations | HIGH | 4 is minimum viable (down from 16) |
| Lower GPU_MEMORY_UTILIZATION | MEDIUM | 0.3-0.5 for colocate mode |
| Gradient checkpointing | MEDIUM | `gradient_checkpointing=True` |
| Reduce max_completion_length | MEDIUM | 2048 vs 4096 saves ~50% gen memory |
| bf16 generation | LOW | Ensure gen runs in bf16 not fp32 |
| Offload ref model | HIGH | Move reference model to CPU during gen |

### For Our 30B (3B Active) Model on 94GB GPU
- Model weights: ~6GB in bf16 (3B active params)
- With LoRA: ~6.5GB
- Generation KV cache: variable, but Mamba is more efficient than transformer
- **Should fit comfortably on H100 94GB** even with num_generations=8-16
- Key issue is CUDA compatibility, not raw memory

### Sparse-RL Alternative (Jan 2026)
- Compress KV cache during rollouts to break memory wall
- Uses Sparsity-Aware Rejection Sampling to fix policy mismatch
- Retains 96%+ of dense performance up to 7B scale
- Good for longer context RL training

Sources:
- [GRPO VRAM Guide](https://ghost.oxen.ai/grpo-vram-requirements-for-the-gpu-poor/)
- [OOM Fix Guide](https://home.mlops.community/public/blogs/stop-guessing-a-systematic-guide-to-fixing-cuda-out-of-memory-errors-in-grpo-training)
- [Sparse-RL Paper](https://arxiv.org/abs/2601.10079)
- [TRL OOM Issue](https://github.com/huggingface/trl/issues/3678)

---

## 8. Alternatives to vLLM for GRPO Generation

### Option A: Disable vLLM, Use Native Generation
```python
from trl import GRPOConfig
config = GRPOConfig(
    use_vllm=False,  # Falls back to model.generate()
    # ... other params
)
```
- **Pros:** Works with ANY model architecture including Mamba
- **Cons:** 5-10x slower generation than vLLM
- **Viable?** Yes for small datasets and few generations per prompt

### Option B: vLLM Transformers Backend
```python
config = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    vllm_model_impl="transformers",  # Uses HF transformers backend
)
```
- **Pros:** May work with Mamba if HF transformers supports the model
- **Cons:** Less optimized than native vLLM backend

### Option C: Offline/Pre-Generated Rollouts
- Generate completions separately (e.g., on inference-optimized setup)
- Train on pre-generated data using off-policy GRPO
- Use importance sampling to correct for off-policy bias
- **Most memory-efficient** — generation and training never co-exist on GPU

### Option D: VeRL Framework
- VeRL has proven Mamba GRPO support (M1 paper)
- CUDA graph fixes already implemented
- Supports LoRA with GRPO
- More complex setup but best option for our architecture

### Recommendation for Our Setup
**Option C (offline rollouts) or Option D (VeRL)** are most practical:
- Option C: Generate completions with the base model, then train LoRA with GRPO on pre-generated data
- Option D: Use VeRL which has native Mamba GRPO support

Sources:
- [TRL GRPOTrainer Docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [vLLM Colocate Blog](https://huggingface.co/blog/vllm-colocate)
- [VeRL GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html)

---

## 9. Unsloth GRPO for MoE Models

### 2026 Updates
- **12x faster MoE training** with >35% less VRAM
- Custom Triton grouped-GEMM + LoRA kernels
- 6x longer context for RL training
- **5GB VRAM** for DeepSeek-R1-style reasoning models (consumer hardware!)

### Concrete Numbers
| Model | VRAM (Unsloth) | Notes |
|-------|---------------|-------|
| gpt-oss-20b | 12.8 GB | MoE model |
| Qwen3-30B-A3B (16-bit LoRA) | 63 GB | Similar to our Nemotron |
| DeepSeek-R1 style reasoning | 5 GB | Consumer GPU possible |

### Nemotron Support
- Unsloth now supports all Nemotron models (Nano and Super)
- Fine-tuning guide available
- Known issues with dependency versions and mamba_ssm CUDA extensions
- **GRPO with Nemotron on Unsloth:** possible but may hit generation errors with Mamba layers

### Assessment for Our Setup
- **Mamba/MoE compatible?** Partially — MoE training works, Mamba generation may have issues
- **Memory:** Excellent — 35% VRAM savings
- **LoRA adapter?** Yes, native support
- **Expected improvement:** Faster training iterations = more experiments
- **Difficulty:** Medium — need to test Mamba compatibility

Sources:
- [Unsloth 2026 Update](https://unslothai.substack.com/p/unsloth-2026-update-faster-moe)
- [Unsloth MoE Docs](https://unsloth.ai/docs/new/faster-moe)
- [Nemotron Unsloth Guide](https://unsloth.ai/docs/models/tutorials/nemotron-3)

---

## 10. REINFORCE++ and RLOO — Simpler Alternatives

### REINFORCE++ (Recommended Simpler Alternative)
- **Critic-free** like GRPO, but uses **global** advantage normalization instead of per-prompt groups
- GRPO's per-prompt normalization has extremely high variance and is theoretically biased
- REINFORCE++ normalizes across the entire batch — more stable, less variance
- **Outperforms GRPO** in stability, prevents overfitting in low-data regimes
- Better in complex, long-horizon tool-use tasks

### RLOO (REINFORCE Leave-One-Out)
- For each prompt, generates K samples
- Baseline for sample i = mean of other K-1 samples
- Mathematically almost identical to GRPO but with unbiased advantage estimation
- Available in swift framework and VeRL

### RGRA (REINFORCE with Group Relative Advantage)
- Newest 2026 variant — simplifies GRPO further
- Retains group-relative advantage estimation
- **Removes PPO-style clipping and policy ratio terms entirely**
- Results: "simpler REINFORCE-based approaches can effectively enhance reasoning in LLMs"
- More transparent and efficient than GRPO

### Assessment for Our Setup
- **Mamba/MoE compatible?** Yes — architecture-agnostic
- **Memory:** Same or less than GRPO (no critic)
- **LoRA adapter?** Yes
- **Expected improvement:** Similar to GRPO but more stable training
- **Difficulty:** Low — available in VeRL, Minimal-RL, and other frameworks

Sources:
- [REINFORCE++ Paper](https://arxiv.org/html/2501.03262)
- [REINFORCE for LLMs](https://cameronrwolfe.substack.com/p/reinforce)
- [RGRA Paper](https://arxiv.org/html/2603.18756)
- [Minimal-RL GitHub](https://github.com/RLHFlow/Minimal-RL)
- [From REINFORCE to Dr. GRPO](https://lancelqf.github.io/note/llm_post_training/)

---

## 11. BONUS: LoRA-MoE GRPO (RO-GRPO)

### Problem
When fine-tuning MoE architectures with LoRA + GRPO, naive combination leads to **routing collapse** — experts become underutilized.

### Solution: RO-GRPO (Routing-Optimized GRPO)
- Uses internal expert routing statistics as a **reward signal**
- Curriculum-based scheduling: initially encourage confident routing, then promote load balance
- First demonstration that scalar reward in GRPO can come from model's own internal mechanics

### Results
- GSM8K: 91.51% (+1.37% over standard LoRA GRPO)
- SVAMP: 93.33% (+0.33%)

### Assessment for Our Setup
- **Directly relevant** — our model is MoE with LoRA
- **Implementation:** Would need custom reward function incorporating router statistics
- **Risk:** Must NOT include router layers in LoRA target_modules (known error #10)
- **Difficulty:** High — custom implementation needed

Sources:
- [RO-GRPO Paper](https://openreview.net/forum?id=rhD7ZuFAjU)

---

## Ranked Recommendations for Our Setup

### Priority 1: HIGHEST IMPACT, LOWEST RISK
**Improved SFT with Dr. GRPO loss fixes applied to future RL**
1. Continue improving SFT data quality (current 0.68 score)
2. When ready for RL, use Dr. GRPO loss (`loss_type="dr_grpo"`) with DAPO tricks
3. Binary RLVR reward on \boxed{} extraction
4. Start with num_generations=4-8

### Priority 2: BEST RL METHOD
**REINFORCE++ or RGRA instead of GRPO**
- Simpler, more stable, prevents overfitting
- Global advantage normalization > per-prompt normalization
- Available in VeRL and Minimal-RL frameworks
- Lower risk of training instability

### Priority 3: GENERATION WORKAROUND
**Offline rollout generation + online RL training**
1. Generate 4-8 completions per prompt using model.generate() in separate script
2. Score each completion with binary reward
3. Train LoRA with off-policy GRPO/REINFORCE++ on pre-generated data
4. Avoids the vLLM + Mamba incompatibility entirely
5. Most practical for Kaggle notebook constraints

### Priority 4: FRAMEWORK CHOICE
**VeRL > TRL for Mamba models**
- VeRL has proven Mamba GRPO support (M1 paper)
- CUDA graph fixes implemented
- Supports LoRA with GRPO
- But complex setup — save for when SFT improvements plateau

### Priority 5: ADVANCED (IF SCORE PLATEAUS)
**RO-GRPO for MoE-aware training**
- Custom routing-aware reward function
- Prevents expert collapse during RL
- +1.37% on GSM8K benchmarks
- Requires custom implementation

---

## Key Takeaways

1. **GRPO is not the only option** — REINFORCE++ and RGRA are simpler and often better
2. **Dr. GRPO fixes are essential** — always use per-token loss and remove std normalization
3. **DAPO tricks work** — especially clip-higher and overlong masking (skip dynamic sampling)
4. **Mamba + GRPO is proven** — M1 paper shows it works via VeRL framework
5. **Offline rollouts** are the most practical approach for our GPU constraints
6. **Binary RLVR reward** is perfect for our \boxed{} evaluation format
7. **Data quality still matters most** — AIMO-2 winners used distillation, not just RL
8. **RO-GRPO** is specifically designed for LoRA-MoE architectures like ours
9. **Unsloth 2026** gives 12x MoE speedup but Mamba generation may still be problematic
10. **NeMo RL** supports DAPO natively — this is what NVIDIA uses for Nemotron training

---

## Practical Implementation Plan

### Phase 1: Better SFT (Current)
- Improve training data with better CoT traces
- Target 0.75+ with pure SFT

### Phase 2: Simple RL (Next)
```python
# Offline rollout approach
# Step 1: Generate completions
for prompt in training_prompts:
    completions = model.generate(prompt, num_return_sequences=8, temperature=1.0)
    rewards = [1.0 if extract_boxed(c) == answer else 0.0 for c in completions]
    save(prompt, completions, rewards)

# Step 2: Train with REINFORCE++/GRPO on pre-generated data
# Use Dr. GRPO loss, binary rewards, LoRA only
```

### Phase 3: Online RL (If Needed)
- Set up VeRL with Mamba GRPO
- Use DAPO configuration
- RO-GRPO for MoE-aware optimization
- Target 0.85+

---

*Research conducted 2026-03-23. All sources verified as of search date.*
