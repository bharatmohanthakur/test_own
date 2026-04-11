# Research Notes - April 2, 2026 (Afternoon)

## Leaderboard Status (as of ~5pm IST)

| Rank | Team | Score | Entries | Last Active |
|------|------|-------|---------|-------------|
| 1 | Rock | 0.82 | 15 | 1h |
| 2 | NEMOSPRAKS (yash/Aneesh/Tony) | 0.82 | 67 | 4h |
| 3 | Alice's Wonderland (Yizhou/dong/muyouqian4) | 0.82 | 15 | 2d |
| 4 | JK-Piece | 0.81 | 43 | 9h |
| 5 | toxu | 0.80 | 37 | 1h |
| 6 | Maciej Sypetkowski | 0.80 | 18 | 3h |
| 7 | Saba Pivot | 0.80 | 40 | 12h |
| 8 | Sultan Algizani | 0.80 | 9 | 2d |
| 9 | Just a test | 0.80 | 16 | 2d |
| 10 | Genetron (Drowsy) | 0.79 | 15 | 1d |

**Our rank: #122, Score: 0.69**
**Gap to #1: 0.13**
**Pending submission: v7 resub #2 (variance roll)**

## Key Discussion Thread Findings

### 1. Score Drop (16 upvotes, 25 comments) - CRITICAL
- JK-Piece (4th place, 0.81): "My notebook of 0.80+ now scores 0.77"
- Everyone's score fell by 0.03-0.04 due to evaluation metric change
- muyouqian4 (3rd place): "I resubmit my best submission.csv, 0.82 -> 0.8"
- **Non-determinism confirmed**: Same submission yields different scores (0.66-0.67-0.66-0.62 variance)
- Ryan Holbrook (host): "We are currently assessing a rescore"
- Inference is non-deterministic even with temp=0

### 2. Per-Category Error Analysis (9 upvotes) - VERY ACTIONABLE
**From EnDream (0.63 LB, SFT with 1200 samples):**
| Category | Accuracy | Error Pattern |
|----------|----------|---------------|
| numeral | 100% | -- |
| unit_conv | 100% | -- |
| bit_ops | 30% | Guesses plausible but wrong bit patterns |
| gravity | 12% | Numerical errors 10-25%, doesn't compute g correctly |
| cipher | 0% | Random plausible words, no actual decryption |
| symbol | 6% | Completely wrong outputs |

**From James Day (0.68 LB, distillation-based, 1K out-of-distribution eval):**
| Category | 8K tokens | 16K tokens |
|----------|-----------|------------|
| bit_manipulation | 9.9% | 18.8% |
| numeral_system | 100.0% | 100.0% |
| physics_gravity | 98.8% | 98.8% |
| symbol_transform | 17.3% | 20.7% |
| text_cipher | 75.5% | 76.1% |
| unit_conversion | 100.0% | 100.0% |
| overall | 67.1% | 69.3% |

**Key insight**: James Day gets 75% on cipher (we likely get 0%), but only 10-19% on bit_manipulation. Longer tokens help bit_manipulation (8K->16K: 10%->19%).
- Bit manipulation failures stem from "distilling models which yap too much" - Qwen3.5 27B solves 49% at 16K but only 10% at 8K
- "Poisonous examples" in training data cause problems

### 3. Training Data Quality (from Ashutosh Kumar, 15th place)
**MAJOR finding - training data is POISONED:**
- **Bit manipulation: 50.5% of training traces are WRONG** (764/1513 mismatches between computed Result and answer)
- **Equation transformation: 49% have I/O length mismatches** making char_map fundamentally broken
- Another 44% have unknown '?' characters when applying char_map
- "bit_manipulation is poisoning the model with wrong answers in half its examples"
- Script available to find these bugs

### 4. RL/GRPO Difficulty (2 upvotes, 8 comments)
- RL is hard on this model: need 2 copies of model, bf16 training impossible
- Unsloth: "debugging is super hard and very..."
- HuggingFace GRPO trainer supports same vLLM + PyTorch RAM sharing as Unsloth
- "vLLM + sleep mode worked for me" (ImperfectKitto, 314th)
- Some LoRA layers not supported by vLLM for adapter loading

### 5. More Data ≠ Better Score
- **600 samples >> 9500 samples** for training
- ForcewithMe: "Sampling 600 data is much better than 9500 data"
- QianYuu: "0.62 for 9500 data, 0.67 for 600 data, 0.66 for 1200"
- Noizersam (222nd): "significant distributional shifts between training and test sets, as well as noisy data"
- Another user: "4800 CoT samples = 0.26 score, ~3500 CoT at ~0.60 epochs = 0.52"

### 6. Scoring Tolerance
- Numerical tolerance: 1% (1e-2)
- String matching: case-insensitive exact match

### 7. No Public Notebooks
- The Code tab shows "No notebooks found" -- either not shared or competition rules prevent it

## Techniques from Web Search

### Nemotron-CrossThink (NVIDIA, April 2025)
- Framework for scaling RL-based self-learning beyond math to diverse reasoning
- Multi-domain corpora (STEM, humanities, social sciences) + structured templates
- +30.1% on MATH-500, +15.1% on AGIEVAL, +12.8% on MMLU-Pro
- Uses 28% fewer tokens for correct answers
- **Relevance**: Cross-domain RL training could help with the diverse puzzle types

### Nemotron-Cascade RL
- Cascaded domain-wise GRPO with strict on-policy training
- Surpasses DeepSeek-R1-0528 on coding benchmarks

## Actionable Items for Next Training Run

### Priority 1: Data Quality (BIGGEST lever)
1. **Clean the training data** - Remove the 50.5% poisoned bit_manipulation traces and 49% broken equation_transformation examples
2. **Generate verified synthetic data** for weakest categories (cipher, bit_manipulation, symbol_transform)
3. **Use 600-1200 high-quality examples, NOT 9500** - Multiple data points confirm less data beats more
4. **Longer context for bit_manipulation** - James Day's data shows 8K->16K nearly doubles accuracy (10%->19%)

### Priority 2: Per-Category Training Strategy
- numeral_system + unit_conversion: Already 100% -- minimal effort needed
- physics_gravity: James Day gets 98.8% -- invest in good programmatic CoT (compute g=2d/t^2)
- text_cipher: James Day gets 75% -- need letter-by-letter substitution CoT traces
- bit_manipulation: Hardest category -- need clean training data, concise traces (not verbose), 16K token budget
- symbol_transform: ~20% -- needs better char_map reasoning traces
- equation_transform: Data is fundamentally broken -- may need completely different approach

### Priority 3: Training Configuration
- Stay with SFT for now (GRPO is too memory-intensive and unstable)
- Use distilled CoT from strong models (but verify correctness!)
- Keep max_seq_len at 4096+ (longer helps bit manipulation)
- Small high-quality dataset > large noisy dataset

### Priority 4: Exploit Non-Determinism
- Same adapter gives different scores on resubmission (0.03-0.04 variance)
- Submit best adapter multiple times to get lucky on the stochastic evaluation
- Keep 2 submission slots for variance rolling on best adapter
