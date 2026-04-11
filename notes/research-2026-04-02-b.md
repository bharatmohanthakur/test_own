# Research Notes — 2026-04-02 (Deep Dive B)

## Status Check
- **Our score**: 0.69 (still waiting for v15 submission score)
- **Top score**: 0.82 (three teams tied: Rock, NEMOSPRAKS, Alice's Wonderland)
- **Gap**: 0.13 (need ~19% relative improvement)
- **Days to midpoint prize**: 7 (April 9, 2026)
- **Total competitors with 0.80+**: 9 teams
- **Total competitors with 0.75+**: ~35 teams

## Updated Leaderboard (April 2, 2026)
| Rank | Score | Team | Entries | Last |
|------|-------|------|---------|------|
| 1 | 0.82 | Rock | 11 | 20h |
| 2 | 0.82 | NEMOSPRAKS (yash bhaskar, Aneesh, Tony Li) | 66 | 2h |
| 3 | 0.82 | Alice's Wonderland (Yizhou XU, dong cookies, muyouqian4) | 15 | 1d |
| 4 | 0.81 | JK-Piece | 41 | 14h |
| 5 | 0.80 | toxu | 32 | 11h |
| 6 | 0.80 | Maciej Sypetkowski + Natalia | 17 | 13h |
| 7 | 0.80 | Saba Pivot | 38 | 2h |
| 8 | 0.80 | Sultan Algizani | 9 | 2d |
| 9 | 0.80 | Just a test | 16 | 1d |
| 10 | 0.79 | Genetron (Drowsy) | 15 | 10h |
| ... | 0.69 | **bharat (us)** | 7 | pending |

**Key observations**:
- NEMOSPRAKS: 66 entries -- brute-forcing variance from temp=1.0
- Top teams are very active (submitting multiple times per day)
- Many teams at 0.80 level suggest a technique plateau at that point

---

## CRITICAL FINDING 1: Temperature Changed to 1.0

### What happened
- Evaluation temperature changed from 0.0 to 1.0 (confirmed by JK-Piece, 4th place)
- MD Mushfirat Mohaimin: "evaluation is not running with temperature=0. Currently set to 1.0"
- Ryan Holbrook (Kaggle Staff): "looking into a rescore" but no fix yet
- No per-request seed set -- scoring is non-deterministic
- Same adapter gets different scores each submission

### Impact
- Everyone's scores dropped 0.03-0.04 on average
- muyouqian4 (3rd place): "resubmit my best submission, 0.82 -> 0.80"
- Some dropped much more: one user went from 0.80+ to 0.64
- Multiple teams now submit 5x/day hoping for high variance rolls

### Strategy implications
1. **Submit same adapter multiple times** to capture high-variance outcomes
2. **Train for deterministic outputs** -- model should converge on one answer even at temp=1.0
3. **Higher confidence = less variance** -- if model assigns high probability to correct answer, temp=1.0 matters less
4. **Short, decisive answers** reduce variance more than long reasoning chains

---

## CRITICAL FINDING 2: Per-Category Error Analysis (GOLD DATA)

### EnDream's analysis (0.63 LB, 1200 training examples, LoRA rank 32, 2 epochs, NO thinking)
| Category | Accuracy | Error Pattern |
|----------|----------|---------------|
| numeral | 100% | Solved |
| unit_conv | 100% | Solved |
| bit_ops | 30% | Model guesses plausible but wrong bit patterns |
| gravity | 12% | Numerical errors of 10-25%, doesn't compute g correctly |
| cipher | 0% | Outputs random plausible words, no actual decryption |
| symbol | 6% | Completely wrong outputs |

Key observations from EnDream:
- Format is NOT the issue -- all 176 errors had properly formatted \boxed{} output
- cipher at 0% is most striking -- model generates plausible text but no actual letter substitution
- gravity errors are systematic: model "estimates" instead of computing g = 2d/t^2
- symbol data quality may be a factor -- some problems may not be uniquely solvable

### James Day's analysis (0.68 LB, DIFFERENT results, uses Qwen3.5 27B distillation)
| Category | 8K tokens | 16K tokens |
|----------|-----------|------------|
| bit_manipulation | 9.9% | 18.8% |
| numeral_system | 100% | 100% |
| physics_gravity | 98.8% | 98.8% |
| symbol_transform | 17.3% | 20.7% |
| text_cipher | 75.5% | 76.1% |
| unit_conversion | 100% | 100% |
| **overall** | **67.1%** | **69.3%** |

**CRUCIAL insight from James Day (0.68 LB):**
- He uses Qwen3.5 27B for distillation WITH thinking enabled
- Gets 75.5% on cipher (vs EnDream's 0%) and 98.8% on gravity (vs EnDream's 12%)
- BUT gets 9.9% on bit_manipulation -- much worse
- "Qwen3.5 27B can solve 49% of bit problems within 16K tokens, but drops to 10% at 8K"
- Suspects "poisonous examples" from distillation that yap too much

**TAKEAWAY**: The training data quality (especially CoT traces) determines per-category performance dramatically. Using a strong distillation model with proper thinking AND enforcing the model to think (not just output answers) is critical.

### Gap analysis: What we need to get to 0.82
Assuming 6 categories with ~equal weight (~16.7% each):
- numeral + unit_conv: already 100% = 33.3% of score
- If we get cipher to 75%: +12.5% (from 0%)
- If we get gravity to 99%: +16.5% (from 0%)  
- If we get bit_ops to 50%: +8.3%
- If we get symbol to 50%: +8.3%
- Total: 33.3 + 12.5 + 16.5 + 8.3 + 8.3 = ~79%
- To hit 82%: need cipher ~80%, gravity ~99%, bit_ops ~60%, symbol ~55%

---

## CRITICAL FINDING 3: Distillation is the Key Technique

### What top teams are doing (inferred)
1. **Distilling from strong reasoning models** (Qwen3.5 27B, DeepSeek-R1) to generate CoT traces
2. **Filtering aggressively** -- bad CoT traces (too long, wrong answer) are poison
3. **16K+ token CoT traces** improve bit_manipulation by 2x vs 8K traces
4. **SFT on high-quality distilled data** is the primary technique -- NOT GRPO
5. **Multiple submissions per day** to exploit temp=1.0 variance

### Why GRPO is failing for most teams
From discussion thread "RL/GRPO difficulty":
- Unsloth does NOT support fast_inference for Mamba hybrid models (confirmed bug)
- vLLM has issues loading LoRA adapters for Nemotron (conv1d BaseLayerWithLoRA error)
- Need 2 copies of model in bf16 -- impossible on single GPU
- No KV cache in transformers inference engine -- painfully slow generation
- One user reports "vLLM + sleep mode worked" but most can't get it running
- **Most competitors are sticking with SFT due to GRPO complexity**

---

## CRITICAL FINDING 4: Nemotron-Cascade 2 Training Recipe (Open-Sourced)

### Pipeline (directly applicable to our competition)
1. **SFT first**: 15.9M samples packed into 256K token sequences
2. **Sequential RL domains**: IF-RL -> Multi-domain RL -> MOPD -> RLHF -> Long-context RL -> Code RL -> SWE RL
3. **GRPO config**: strict on-policy (importance ratio = 1.0), no KL divergence term
4. **Reduces to REINFORCE** with group-normalized rewards and token-level loss

### Key data volumes
- Math: 1.8M tool-calling + 2.6M non-tool = 4.4M samples
- Code: 1.9M Python + 1.0M C++ + 1.3M Python tool-calling = 4.2M samples
- Chat: 4.9M reasoning-enabled + 372K reasoning-disabled = 5.3M samples
- Total SFT: ~15.9M samples

### What's relevant for us
- The SFT stage alone produces very strong results
- Domain-specific data quality matters more than RL tricks
- Tool-integrated reasoning (code execution) adds significant lift
- Available via NeMo-RL repository

---

## CRITICAL FINDING 5: Available Datasets (Prioritized for Competition)

### Tier 1: Most Directly Useful
1. **nvidia/Nemotron-Cascade-RL-Math** (14,476 problems)
   - Math problems with short answers from OpenMathReasoning + NuminaMath + DeepScaleR + AceReason
   - Decontaminated against benchmark test sets
   - CC-BY-4.0 license
   - Perfect for RL training on math reasoning

2. **nvidia/Nemotron-RL-math-OpenMathReasoning** (112,867 Q&A pairs)
   - AoPS forum problems with expected answers
   - Format: user message with "solve... put answer in \boxed{}" + expected_answer
   - Ready for RL/RLVR training
   - Covers algebra, geometry, number theory, combinatorics

3. **nvidia/Nemotron-Post-Training-Dataset-v2** (6.2M samples)
   - Math: 239K, Code: 175K, STEM: 355K, Chat: 628K (English only ~1.4M)
   - Generated by DeepSeek-R1-0528, Qwen2.5-14/32B, Qwen3-30B-A3B, Qwen3-235B
   - Has reasoning traces from DeepSeek-R1
   - Can extract math + STEM subsets for SFT

### Tier 2: Supplementary
4. **nvidia/Puzzle-KD-Nemotron-Post-Training-Dataset-v2** (851K samples)
   - English-only subset of Post-Training-v2
   - Math + Code + STEM + Chat
   - BUT reasoning="off" (no thinking traces) -- less useful for us
   
5. **nvidia/Llama-Nemotron-Post-Training-Dataset** (30M+ samples)
   - Math: 2.2M, Code: 500K reasoning data
   - Massive scale but may need heavy filtering

### Tier 3: For broader generalization
6. **nvidia/OpenMathReasoning** (540K problems, 3.2M solutions)
   - Full AIMO-2 winning dataset
   - Includes tool-integrated reasoning solutions
   - Very high quality but enormous

---

## CRITICAL FINDING 6: What 0.82 Teams Are Likely Doing

Based on all evidence, the 0.82 teams likely:

1. **Training data**: Distilled CoT traces from Qwen3.5 27B or DeepSeek-R1 for ALL 6 puzzle types
2. **CoT quality**: Long thinking traces (16K+) with proper step-by-step reasoning
3. **Data volume**: 5000-20000 high-quality examples (not just 1000-2000)
4. **Format**: Proper `<think>...</think>\n\n\boxed{answer}` format
5. **Per-category specialization**: Different CoT strategies for different puzzle types:
   - cipher: letter-by-letter substitution mapping
   - gravity: explicit g = 2d/t^2 computation from examples
   - bit_ops: step-by-step binary operation tracing (needs 16K+ tokens)
   - symbol: operation identification from examples
   - numeral/unit_conv: already easy, basic computation
6. **Multiple submissions**: 5+/day to exploit temp=1.0 variance
7. **SFT only**: Most are NOT using GRPO (too hard with Mamba)
8. **Possibly external compute**: Some teams likely train outside Kaggle

---

## ACTIONABLE NEXT STEPS (For April 2-9 Sprint)

### Priority 1: Data Quality Revolution (BIGGEST LEVER)
- [ ] Generate CoT traces using MinMax 2.7 API for ALL 6 puzzle types
- [ ] Focus on cipher (0% -> 75%), gravity (12% -> 99%), bit_ops (30% -> 50%)
- [ ] Use programmatic CoT for cipher: write out the actual letter mapping step-by-step
- [ ] Use programmatic CoT for gravity: compute g from examples, then apply d=0.5*g*t^2
- [ ] For bit_ops: generate detailed binary operation traces, ensure < 8K tokens to avoid truncation
- [ ] Aim for 5000-10000 high-quality examples total

### Priority 2: Training Optimization
- [ ] Increase max_seq_len to 4096 or higher (bit_ops needs more tokens)
- [ ] Use 3-5 epochs with 5000+ examples
- [ ] Keep LoRA rank 32, alpha 64
- [ ] Include broader reasoning data (math from Nemotron-Cascade-RL-Math)
- [ ] Mix: 75% reasoning / 25% non-reasoning to preserve model capabilities

### Priority 3: Exploit Variance
- [ ] Submit same adapter 3-5 times to capture high-variance outcomes
- [ ] Save best-performing submissions, don't waste slots on untested adapters
- [ ] Coordinate submissions: submit early in day, track scores, resubmit if low

### Priority 4: GRPO (If Time Permits)
- [ ] Use offline rollout approach: generate completions with model.generate(), then train
- [ ] Binary reward on \boxed{} extraction
- [ ] Skip vLLM -- use native transformers generation (slow but works)
- [ ] Focus GRPO on hardest categories (cipher, bit_ops, symbol)

---

## Key Resources
- Nemotron-Cascade 2 paper: https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/
- Nemotron-Cascade 2 blog (Maxime Labonne): https://maximelabonne.substack.com/p/nemotron-cascade-2-on-policy-distillation
- Nemotron 3 Nano training recipe: https://docs.nvidia.com/nemotron/latest/nemotron/nano3/README.html
- AIMO-2 winning solution: https://arxiv.org/abs/2504.16891
- OpenMathReasoning dataset: https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning
- Nemotron-Cascade-RL-Math: https://huggingface.co/datasets/nvidia/Nemotron-Cascade-RL-Math
- Nemotron-Post-Training-v2: https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2
- OpenReasoning-Nemotron (distilled models): https://huggingface.co/blog/nvidia/openreasoning-nemotron
- Score drop discussion: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/685920
- Error analysis discussion: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/686069
- GRPO difficulty discussion: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/686794
- Unsloth Nemotron guide: https://docs.unsloth.ai/models/nemotron-3
- Unsloth Mamba issue: https://github.com/unslothai/unsloth/issues/4073

---

## Critical Insight Summary

**The gap from 0.69 to 0.82 is almost entirely about training data quality**, specifically:
1. Using a strong teacher model (Qwen3.5 27B or DeepSeek-R1) to generate CoT traces
2. Ensuring traces actually solve the problem step-by-step (not just output a plausible answer)
3. Per-category specialized strategies (cipher needs letter mapping, gravity needs g calculation)
4. Sufficient token length for complex problems (16K for bit_ops)
5. Volume: 5000-10000 examples, not 1000-2000

GRPO is a distraction for now -- most top teams are pure SFT with better data. Focus 100% on data quality.
