# RL Methods for Reasoning: Hard Benchmark Numbers
**Date: 2026-03-23**

---

## 1. Rejection Sampling Fine-Tuning (RFT) vs GRPO

### GSM8K Accuracy (LLaMA models)

| Model | SFT | RFT (k=100) | RFT (multi-model) | Delta |
|-------|-----|-------------|-------------------|-------|
| LLaMA-7B | 35.9% | ~47% | **49.3%** | +13.4 |
| LLaMA2-7B | 41.6% | ~48% | **50.3%** | +8.7 |
| LLaMA-13B | 43.0% | ~50% | **52.1%** | +9.1 |
| LLaMA2-13B | 50.0% | ~53% | **55.4%** | +5.4 |

Source: "Scaling Relationship on Learning Mathematical Reasoning with LLMs" (arxiv 2308.01825)

### RFT vs GRPO Head-to-Head (arxiv 2504.11343)
- **RAFT++ (rejection sampling)** shows faster early-stage learning than GRPO
- **GRPO eventually surpasses RAFT++** in later training stages
- Key insight: GRPO's advantage comes from **discarding prompts with entirely incorrect responses**, NOT from reward normalization
- A minimal "Reinforce-Rej" (filter both all-correct and all-incorrect samples) is competitive with full GRPO
- **Verdict**: For limited compute, RFT/RAFT is competitive. For long training, GRPO wins marginally.

### DeepSeekMath RFT Details
- k=100 samples per problem on GSM8K
- DeepSeekMath-7B achieves **90% pass@k** when sampling 100+ responses per query
- Diminishing returns when doubling k (distinct reasoning paths don't grow linearly with k)
- k=3 already gives +2 points over SFT; most gains come by k=32-100

---

## 2. SimPO vs DPO vs KTO

### General Benchmarks (NOT math-specific)

| Method | AlpacaEval 2 | Arena-Hard |
|--------|-------------|------------|
| DPO | baseline | baseline |
| SimPO | **+6.4 pts** | **+7.5 pts** |

Source: SimPO paper (NeurIPS 2024)

### Math Reasoning Specifically
- SimPO-based verifiers outperform DPO-based verifiers on GSM8K and MATH (state-of-the-art)
- ORPO and SimPO consistently outperform DPO for verifier training
- KTO works with binary (thumbs up/down) feedback -- useful when paired preferences unavailable
- **No direct SimPO vs DPO vs KTO comparison on AIME/MATH with same model found**

### Critical Context for 2025-2026
- The field has moved AWAY from DPO/SimPO/KTO for reasoning
- Current stack: SFT for instruction following + DPO/SimPO for alignment + **GRPO/DAPO for reasoning**
- DPO/SimPO/KTO are bounded by quality of preference pairs; GRPO can improve beyond training data
- **Verdict for our competition**: DPO/SimPO/KTO are NOT the right tool. Use GRPO/DAPO for reasoning improvement.

---

## 3. STaR (Self-Taught Reasoner) Results

### Benchmark Results by Task

| Task | Baseline | After STaR (16 iter) | Improvement |
|------|----------|---------------------|-------------|
| N-digit addition | 76.3% | **89.5%** | +13.2 |
| CommonsenseQA (Dev) | ~37% (few-shot) | **72.5%** | +35.9 |
| GSM8K (answer-only) | 5.8% | **~10.7%** | ~+5 |

Source: STaR paper (NeurIPS 2022)

### Key Details
- 16 iterations needed for arithmetic task convergence
- Each iteration: generate rationales -> filter correct -> finetune -> repeat
- CommonsenseQA result matches or beats models 30x larger
- **Limitation**: GSM8K improvement modest (5.8% -> 10.7%) on older models
- Modern V-STaR adds a verifier on top of STaR for further gains

---

## 4. DAPO vs GRPO vs VAPO (AIME 2024)

### AIME 2024 Scores (Qwen2.5-32B base)

| Method | AIME 2024 Score | Training Steps | Relative Efficiency |
|--------|----------------|----------------|-------------------|
| GRPO (DeepSeek-R1-Zero) | **47/100** | ~10,000 | 1x (baseline) |
| DAPO | **50/100** | ~5,000 | 2x faster than GRPO |
| VAPO | **60.4/100** | ~5,000 | Matches DAPO speed, +10.4 pts |

Source: DAPO (arxiv 2503.14476), VAPO (arxiv 2504.05118)

### VAPO Key Insight
- VAPO uses a **value model** (critic) unlike GRPO/DAPO which are critic-free
- Better credit assignment = higher theoretical ceiling
- Achieves 60.4 within just 5,000 steps, maintains stable entropy
- Consistent 60-61 across three repeated experiments

### DAPO vs GRPO Key Differences
- DAPO fixes GRPO's length bias (token-level loss vs sample-level)
- DAPO prevents entropy collapse via dynamic sampling
- DAPO's Dynamic Sampling component actually **doesn't help** -- best results with DS disabled
- **Verdict**: DAPO > GRPO by ~3 points. VAPO > both but requires critic (more complex).

---

## 5. Dr. GRPO Results

### AIME 2024 (Qwen2.5-Math-7B base)

| Method | AIME 2024 Pass@1 |
|--------|-----------------|
| OpenReasoner-Zero-7B | 16.7% |
| Prime-Zero-7B | 27.6% |
| SimpleRL-Zero-7B | 36.0% |
| **Dr. GRPO** | **43.3%** |

Source: Sea AI Lab (arxiv 2503.20783)

### What Dr. GRPO Changes
- Removes length normalization (prevents rewarding longer wrong answers)
- Removes std normalization
- Gains: +1.0-1.9% accuracy over vanilla GRPO, robust across tasks and model sizes
- lambda-GRPO reportedly outperforms Dr. GRPO slightly in stability

---

## 6. REINFORCE++ vs GRPO

### Key Differences
- REINFORCE++ uses **global batch normalization** vs GRPO's **prompt-level (local) normalization**
- REINFORCE++ shows superior **out-of-distribution (OOD) performance** and training stability
- GRPO with local norm tends to overfit; REINFORCE++ learns more stably

### Benchmark Numbers
- Across algorithms: sampling efficiency ranges from GRPO's 43.9 to RLOO's 42.6 on in-domain test
- ProRL V2 uses REINFORCE++-baseline to train state-of-the-art 1.5B reasoning model
- ScaleRL validates REINFORCE++-baseline in large-scale scenarios
- **No single paper with direct REINFORCE++ vs GRPO accuracy comparison on same model found**
- The differences appear to be <2% in accuracy but significant in stability/generalization

---

## 7. Iterative DPO Rounds and Diminishing Returns

### Per-Round Improvements (Llama-2-70B-Chat)

| Metric | Baseline | Round 1 | Round 2 | Round 3 | Round 4 |
|--------|----------|---------|---------|---------|---------|
| GSM8K | 55.6% | 73.1% (+17.5) | 78.0% (+4.9) | 81.1% (+3.1) | 81.6% (+0.5) |
| MATH | 12.5% | ~16% | ~18% | ~20% | **20.8%** |
| ARC-Challenge | 77.8% | ~82% | ~84% | ~86% | **86.7%** |

Source: Iterative RPO (arxiv 2404.19733)

### Diminishing Returns Pattern
- Round 1: **+17.5%** (massive)
- Round 2: **+4.9%** (significant)
- Round 3: **+3.1%** (moderate)
- Round 4: **+0.5%** (negligible)
- **Conclusion**: 2-3 rounds is optimal. Round 4+ is not worth the compute.
- With majority voting (32 samples): GSM8K goes from 70.7% to 88.7%

### Quality-Quantity Tradeoff
- High sampling budgets cause **overfitting** in later iterations
- Models lose diversity, reducing ability to generate diverse trajectories
- Low KL coefficients accelerate this problem

---

## 8. Offline GRPO vs Online GRPO

### Performance Gap
- Online DPO/GRPO has **1.3x to 1.6x higher winrates** vs offline on instruction-following
- Online GRPO allows training data to be sampled **up to 64x less frequently** without degradation
- Fully online DPO/GRPO perform comparably, both **significantly outperform** offline DPO
- Semi-online DPO (less frequent sync) nearly matches fully online performance

### Math Reasoning Specific
- PCL-Reasoner-V1.5 (offline RL) achieves 90.9% on AIME 2024, 85.6% on AIME 2025 with Qwen2.5-32B
- This proves offline RL CAN work if done right, but requires careful design
- Online GRPO suffered entropy collapse after 150 steps in some experiments
- **Verdict**: Online is better in theory but offline is practical and can match online with proper techniques

---

## 9. NemoSkills AIMO-2 Winning Solution

### Competition Result
- **34 out of 50** Olympiad problems solved in 5 hours on 4x L4 GPUs
- 1st place

### Training Method: Pure SFT + Tool-Integrated Reasoning (NO RL)
- Base model: **Qwen2.5-14B-Base**
- Training data: **540K unique math problems**, **3.2M long-reasoning solutions**
- Generated by DeepSeek-R1 and QwQ-32B (knowledge distillation)
- **1.7M Tool-Integrated Reasoning (TIR) solutions** via iterative training + quality filtering
- GenSelect (generative solution selection) instead of majority voting

### Critical Takeaway
- **The winner used pure SFT distillation, NOT RL**
- Data quality and scale were the winning factors (540K problems, 3.2M solutions)
- TIR (model writes and executes Python code) was a key differentiator
- The AIMO-2 winning approach does NOT use GRPO/DAPO/PPO

### Fast-Math-R1 (4th place AIMO-2): SFT + GRPO Two-Stage
- Base: DeepSeek-R1-Distill-Qwen models
- Stage 1: **10 epochs SFT** on ~8,500 hard problems (critical for breakthrough)
- Stage 2: GRPO to optimize solution length (not accuracy)
- GRPO's role here: **makes model 30-60% faster** while preserving accuracy
- **RL improved efficiency, not accuracy** in this setup

---

## 10. Best RL Method for Small Models (1.5B-7B)

### 1.5B Parameter Models (AIME 2024)

| Model | Method | AIME 2024 Pass@1 |
|-------|--------|-----------------|
| DeepSeek-R1-Distill-Qwen-1.5B | Distillation (SFT) | **28.9%** |
| STILL-3-1.5B | + RL (PPO) | **39.3%** (+10.4) |
| DeepScaleR-1.5B | + RL (GRPO) | **43.1%** (+14.2) |
| Tina-Open-RS2 (1.5B) | + RL (GRPO via LoRA) | **43.3%** (avg 50.6%) |

### 7B Parameter Models (AIME 2024)

| Model | Method | AIME 2024 Pass@1 |
|-------|--------|-----------------|
| OpenReasoner-Zero-7B | GRPO from scratch | 16.7% |
| Prime-Zero-7B | GRPO from scratch | 27.6% |
| SimpleRL-Zero-7B | GRPO from scratch | 36.0% |
| Dr. GRPO (7B) | Dr. GRPO | **43.3%** |

### 3.8B Models
- Phi-4-mini-reasoning (3.8B): SFT + GRPO pipeline
- Beats DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B on MATH-500
- Comparable to O1-mini on Math-500 and GPQA Diamond

### Key Findings for Small Models
1. **Distillation from strong teacher >> RL from scratch** at small scale
2. RL on top of distilled model gives +10-15% on AIME (significant)
3. GRPO is the most practical RL method at small scale (critic-free, single GPU feasible)
4. DeepScaleR shows **iterative context lengthening** (8K->16K->24K) helps: 22.9% -> 43.1%
5. For <3B models: SFT distillation first, then GRPO if compute allows

---

## 11. Rejection Sampling: Number of Samples Needed

### Empirical Evidence

| Samples (k) | Effect | Source |
|-------------|--------|--------|
| k=3 | +2 pts over SFT (stable) | RFT paper |
| k=8-16 | Most cost-effective range | Multiple papers |
| k=32 | Good for majority voting | Iterative RPO |
| k=64 | DeepSeek-R1 uses for pass@1 estimation | DeepSeek-R1 paper |
| k=100 | DeepSeekMath standard | DeepSeekMath paper |
| k=128 | Maximum useful for Best-of-N | Best-of-Majority paper |

### Key Scaling Insights
- The key factor is **distinct reasoning paths**, not raw sample count
- Diminishing returns when doubling k (paths don't grow linearly)
- Multi-model sampling (combine outputs from different models) >> more samples from one model
- SSA at K=15 can outperform majority voting at K=128 with right aggregation
- **Practical recommendation**: k=16-32 for rejection sampling training, k=64+ for test-time scaling

---

## 12. Tina: Tiny Reasoning Models via LoRA + GRPO

### Model Details
- Base: DeepSeek-R1-Distill-Qwen-1.5B
- Method: LoRA during GRPO-style RL
- Cost: **$9 USD** total for post-training + evaluation
- Training: 19-57% of a single epoch

### Benchmark Results (Pass@1)

| Model | AIME24 | Average (6 benchmarks) | Cost |
|-------|--------|----------------------|------|
| Baseline (1.5B distilled) | ~28.9% | ~42% | $0 |
| Tina-Open-RS2 | **43.33%** | **50.60%** | **$9** |

- >20% reasoning performance increase over baseline
- Best model (Tina-Open-RS2) achieves 50.60% average across AIME24/25, AMC23, MATH500, GPQA, Minerva
- **260x cost reduction** compared to full-parameter RL training

### LoRA vs Full-Parameter for RLVR

| Method | Average Accuracy | % Trainable Params |
|--------|-----------------|-------------------|
| Standard LoRA | 42.5% | ~1-2% |
| DoRA | **46.6%** | ~1-2% |
| Full-parameter | 44.9% | 100% |
| AdaLoRA | 44.2% | ~1-2% |
| PiSSA | 0.2% (COLLAPSE) | ~1-2% |
| MiLoRA | 18.0% | ~1-2% |

Source: "Evaluating Parameter Efficient Methods for RLVR" (arxiv 2512.23165)

**Critical finding**: DoRA > Full-parameter > standard LoRA for RLVR
- Standard LoRA should NOT be the default for RLVR
- DoRA (weight-decomposed LoRA) is strictly better
- PiSSA and MiLoRA cause catastrophic collapse in RLVR -- NEVER use them

---

## Summary: What Actually Helps vs Hype

### PROVEN TO WORK (hard numbers)
1. **SFT distillation from strong teacher**: AIMO-2 winner used ONLY this (34/50 problems)
2. **GRPO on distilled model**: +10-15% AIME at 1.5B scale (28.9% -> 43.1%)
3. **DAPO over GRPO**: +3 points on AIME 2024 (47 -> 50), 2x training efficiency
4. **VAPO over DAPO**: +10 points on AIME 2024 (50 -> 60.4), but needs critic
5. **Iterative DPO**: +25% on GSM8K in round 1, but rounds 3+ nearly useless
6. **RFT**: Competitive with GRPO early, simpler to implement
7. **DoRA > LoRA** for RLVR: 46.6% vs 42.5% average (use DoRA not standard LoRA)

### OVERHYPED OR CONTEXT-DEPENDENT
1. **REINFORCE++ vs GRPO**: <2% accuracy difference, mainly stability gains
2. **SimPO/DPO/KTO for reasoning**: Wrong tool; these are for alignment, not reasoning
3. **STaR**: Old results on old models; modern distillation+GRPO is strictly better
4. **Offline vs Online GRPO**: Gap is real but can be bridged with semi-online approaches
5. **RL for accuracy at competition scale**: AIMO-2 winner used pure SFT, RL only helped speed

### RECOMMENDATIONS FOR OUR COMPETITION (3B active, LoRA rank 32)

1. **Priority 1**: Better SFT data (the AIMO-2 winner's approach)
   - Generate high-quality CoT solutions using strong models
   - 540K problems with 3.2M solutions won AIMO-2 -- data scale matters enormously
   - Tool-Integrated Reasoning (TIR) was a key differentiator

2. **Priority 2**: If doing RL, use GRPO with DoRA (not standard LoRA)
   - DoRA outperforms both standard LoRA AND full-parameter fine-tuning for RLVR
   - Expected gain: +10-15% accuracy on reasoning benchmarks

3. **Priority 3**: RFT (rejection sampling) as a simpler alternative to GRPO
   - Generate 16-32 solutions per problem, keep correct ones, finetune
   - Competitive with GRPO, much simpler to implement
   - Can be done iteratively (2-3 rounds max)

4. **Skip**: VAPO (needs critic, too complex for our setup), SimPO/DPO (wrong tool for reasoning), STaR (outdated)

---

## Sources

- [Scaling Relationship on Learning Mathematical Reasoning](https://arxiv.org/abs/2308.01825) - RFT results on GSM8K
- [A Minimalist Approach to LLM Reasoning](https://arxiv.org/abs/2504.11343) - RAFT vs GRPO comparison
- [SimPO: Simple Preference Optimization](https://github.com/princeton-nlp/SimPO) - SimPO vs DPO numbers
- [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026) - Overview
- [STaR: Self-Taught Reasoner](https://arxiv.org/abs/2203.14465) - STaR benchmark results
- [DAPO: An Open-Source LLM RL System at Scale](https://arxiv.org/html/2503.14476v1) - DAPO vs GRPO
- [VAPO: Efficient and Reliable RL](https://arxiv.org/html/2504.05118v1) - VAPO vs DAPO vs GRPO (60.4 on AIME)
- [Dr. GRPO (COLM 2025)](https://arxiv.org/pdf/2503.20783) - Dr. GRPO 43.3% on AIME
- [REINFORCE++](https://arxiv.org/abs/2501.03262) - Stability improvements over GRPO
- [Iterative Reasoning Preference Optimization](https://arxiv.org/abs/2404.19733) - Iterative DPO diminishing returns
- [Bridging Offline and Online RL for LLMs](https://arxiv.org/html/2506.21495v1) - Online vs offline gap
- [AIMO-2 Winning Solution](https://arxiv.org/abs/2504.16891) - NemoSkills pure SFT approach
- [Fast-Math-R1](https://arxiv.org/abs/2507.08267) - SFT+GRPO two-stage recipe
- [Tina: Tiny Reasoning Models via LoRA](https://arxiv.org/html/2504.15777v1) - LoRA+GRPO at 1.5B, $9
- [Evaluating Parameter Efficient Methods for RLVR](https://arxiv.org/pdf/2512.23165) - DoRA > LoRA for RLVR
- [DeepScaleR-1.5B](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview) - GRPO at 1.5B scale
- [STILL-3-1.5B](https://arxiv.org/abs/2503.04548) - RL on distilled model
- [Phi-4-Mini-Reasoning](https://arxiv.org/html/2504.21233v1) - 3.8B reasoning model
- [LFM-1B-Math](https://www.liquid.ai/research/lfm-1b-math-can-small-models-be-concise-reasoners) - 1B GRPO-LEAD
- [GRPO++ Tricks](https://cameronrwolfe.substack.com/p/grpo-tricks) - Dr. GRPO and lambda-GRPO details
- [Qwen2.5-Math](https://qwenlm.github.io/blog/qwen2.5-math/) - GRPO training details
- [DeepSeekMath](https://arxiv.org/pdf/2402.03300) - RFT scaling with k=100
- [PCL-Reasoner-V1.5](https://arxiv.org/pdf/2601.14716) - Offline RL at 90.9% AIME
- [Comparative Analysis PPO, GRPO, DAPO](https://arxiv.org/html/2512.07611v1) - Algorithm comparison
- [AIMO-2 Competition Details](https://aimoprize.com/updates/2025-04-15-second-progress-prize-closed)
- [Online vs Offline RL Gap](https://cameronrwolfe.substack.com/p/online-rl) - 1.3-1.6x winrate difference
