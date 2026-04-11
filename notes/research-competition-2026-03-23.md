# Competition Research: NVIDIA Nemotron Reasoning Challenge
**Date:** 2026-03-23
**Current Score:** 0.69 (best), 0.67 (latest) | **Rank:** 22nd | **Gap to #1:** 0.10

---

## 1. LEADERBOARD STATUS (Live from Kaggle, March 23)

| Rank | Team | Score | Entries | Last Sub |
|------|------|-------|---------|----------|
| 1 | Genetron (Drowsy) | **0.79** | 2 | 6h ago |
| 2 | Alice's Wonderland (3-person team) | **0.79** | 2 | 4h ago |
| 3 | ThienAnHoangNguyen | **0.78** | 7 | 11h ago |
| 4 | Saba Pivot | **0.78** | 4 | 15h ago |
| 5 | Maciej Sypetkowski | **0.78** | 8 | 13h ago |
| 6 | JK-Piece | **0.76** | 10 | 6h ago |
| 7 | Q (K Q J) | **0.76** | 3 | 3d ago |
| 8 | Aneesh (2-person) | **0.76** | 8 | 15h ago |
| 9 | toxu | **0.74** | 16 | 14h ago |
| 10 | Gabriel Cha | **0.74** | 4 | 12h ago |
| 11 | Jiazhen Dong (2-person) | **0.73** | 5 | 17h ago |
| 12 | Kota Morimoto | **0.72** | 4 | 9h ago |
| 13 | Roschild.Rui | **0.71** | 5 | 1d ago |
| 14 | SFT 54% (Josephus) | **0.71** | 19 | 20h ago |
| 15 | Thank Arunodhayan | **0.70** | 9 | 3h ago |
| 16 | Gendaijin | **0.70** | 7 | 3h ago |
| 17-21 | Various | **0.69** | - | - |
| **22** | **bharat (US)** | **0.69** | 7 | 13h ago |

**Total teams:** 717
**Competition timeline:** 3 months remaining, deadline June 15, 2026
**50% public / 50% private** leaderboard split

### Key Observations
- **#1 jumped from 0.76 to 0.79** since our last check (was "Q" at 0.76 on Mar 21)
- **Top 5 are all at 0.78-0.79** — very tight cluster
- "SFT 54%" team name suggests they got 0.54 with SFT alone and improved further
- Top entries have very few submissions (2-8), suggesting external training + confident submissions
- Competition is heating up fast — many active submissions in last 24h
- Our gap to #1 is now **0.10** (was 0.08 two days ago — leaders are pulling away)

---

## 2. CRITICAL DISCUSSION FINDINGS

### A. Test Set Composition (CONFIRMED by Kaggle Staff)
**Thread:** "Are problem types the same for train and test?" (16 upvotes)
**Ryan Holbrook (Kaggle Staff):** "Yes, they are the same. The distribution should be roughly similar as well."

**IMPLICATION:** The hidden test uses the SAME 6 puzzle types with similar distribution. We do NOT need to train for broader reasoning (math, coding, science). Focus entirely on the 6 puzzle types.

### B. Distillation is ALLOWED (CONFIRMED by Competition Host)
**Thread:** "Are we allowed to distil larger models?" (lucian kucera, 21st place)
**CPMP (Competition Host):** "Your submission must be a LORA adapter to Nemotron v3 Nano. The way you train this LORA adapter is open."

**IMPLICATION:** We can generate training data using any model (Claude, DeepSeek-R1, GPT-4, Nemotron-340B, Qwen3, etc.) and use it to SFT train the LoRA. This is likely what top teams are doing.

### C. Zero-Shot Performance by Category (lucian kucera, 21st place)
- **numeral_system (number conversion):** 100% accuracy zero-shot
- **gravity + unit_conversion:** ~33% accuracy zero-shot
- **bit_manipulation, equation_transform, cipher:** Very poor zero-shot

**IMPLICATION:** Focus training data on the 4 weak categories. The model already handles numeral conversion perfectly. Training more on it may be wasted effort or even harmful.

### D. LoRA Target Module Warning
**Thread:** "Lora FInetuning TIP to save you compute time" (lucian kucera)
**Key finding:** "You can't just fine tune `up_proj` without `down_proj` on MOE layers — this will cause submission to fail" (vLLM compatibility issue)

**IMPLICATION:** Always include both up_proj and down_proj when targeting MoE layers. Our current config already does this correctly.

### E. RL for Bit Problems: Likely Ineffective
**Thread:** "Why RL for bit problems will not work" (lucian kucera, posted 5h ago)
**Finding:** Rejection sampling on bit manipulation problems very rarely succeeds, suggesting RL (GRPO) won't produce meaningful gradients for this category.

**IMPLICATION:** For bit_manipulation, SFT with high-quality distilled CoT traces is the better approach. Don't waste GPU time on RL for bit problems.

### F. Competition Metric Bug
**Thread:** "Competition Metric Bug: verify method fails for Binary String Problem" (6 upvotes)
**Finding:** There may be edge cases in the verification method for binary string answers. Monitor for updates.

### G. Other Active Discussions
- **RTX PRO 6000 Blackwell CUDA incompatibility** — still being discussed, 11 comments, 10 upvotes
- **Kaggle CLI develop locally + run on RTX Pro 6000** — 30 upvotes, most popular non-pinned thread
- **Midpoint Leaderboard Clarification** — new thread from 6h ago
- **Can we train externally?** — yes, as confirmed by competition host (LoRA must be the submission)
- **Prompting strategies vs sequence length constraints** — 8 upvotes, clarifies that only LoRA is submitted

---

## 3. WHAT TOP TEAMS ARE LIKELY DOING

### Gap Analysis: 0.69 (us) vs 0.79 (top)
The 0.10 gap represents ~100 more correct answers out of ~1000 test questions. Given:
- Test has same 6 types, ~167 each
- We already get numeral_system right (~100% = 167 correct)
- Top teams likely get 130+ correct on each type

### Likely Approach of 0.79 Teams
1. **High-quality distilled CoT training data** — Generated puzzle solutions using a powerful teacher model (DeepSeek-R1, Claude, Qwen3-235B, or Nemotron-340B)
2. **Per-category specialized training** — Different reasoning traces for each puzzle type
3. **Large training dataset** — Likely 50K-100K+ examples (not just the 9,500 provided)
4. **Augmented puzzles** — Programmatically generated new puzzles with known answers
5. **External GPU training** — Training off Kaggle (allowed), uploading only the LoRA adapter
6. **Multiple SFT rounds or longer training** — More epochs, better hyperparameters

### What Separates 0.68 from 0.79
- **0.68 = basic SFT on competition data** (many teams cluster here)
- **0.71-0.74 = better CoT traces + more data**
- **0.76-0.79 = distilled data from powerful models + augmented puzzles + careful hypertuning**

---

## 4. TECHNIQUE RESEARCH

### A. Nemotron-Cascade 2 (NVIDIA, released March 16, 2026)
- Same architecture as Nano 30B (30B MoE, 3B active)
- Uses **Cascade RL**: domain-wise RL training stages with GRPO
- **Multi-domain on-policy distillation**: distills from best-performing intermediate checkpoints
- Achieves IMO 2025 Gold and IOI 2025 Gold
- Available on HuggingFace: `nvidia/Nemotron-Cascade-2-30B-A3B`
- **Idea:** Could we extract the LoRA difference between Cascade-2 and base Nano? Probably not directly since the competition requires LoRA on the specific competition model.

### B. Nemotron-CrossThink (NVIDIA, April 2025)
- Dataset for multi-domain RL training
- 10M-100M examples across STEM, humanities, law, social science
- Covers QA, math, physics, chemistry, business, history
- Available: `nvidia/Nemotron-CrossThink`
- Has `train_qa` and `train_math` splits
- **NOT directly useful** since test set is only the 6 puzzle types (confirmed by staff)
- Could still help with general reasoning capability

### C. Nemotron-Post-Training-v3
- Major NVIDIA post-training dataset collection
- Includes synthetic code, math, science data
- Data cutoff: February 2026
- **Potentially useful** for improving general reasoning before puzzle-specific training

### D. Unsloth for Nemotron Fine-Tuning
- 16-bit LoRA needs ~60GB VRAM
- Don't fine-tune router layer (disabled by default)
- GRPO learning rate: 5e-6 recommended
- Maintain reasoning: use 75% reasoning + 25% non-reasoning data mix
- Available notebooks: LoRA SFT + NeMo Gym GRPO

### E. AIMO-2 Winning Approach (Relevant Lessons)
- Used OpenMath-Nemotron-14B-Kaggle
- Fine-tuned on strategically selected subset (not all data)
- Synthetic solutions from DeepSeek-R1 and QwQ-32B
- Combined natural language reasoning + Python code execution
- Multiple parallel reasoning responses, then majority vote
- **Key lesson:** Data quality and selection > data quantity

---

## 5. ACTIONABLE NEXT STEPS (Priority Order)

### IMMEDIATE (Next 24-48 hours)
1. **Generate distilled training data using a powerful model**
   - Use DeepSeek-R1 API (or any available API) to solve ALL 9,500 train puzzles
   - Get step-by-step reasoning traces for each puzzle type
   - This is the single biggest lever — top teams are doing this

2. **Programmatically generate MORE puzzles**
   - For each puzzle type, write code to generate new examples with known answers
   - For unit_conversion: random conversion factors + random values
   - For gravity: random g values + compute d=0.5*g*t^2
   - For cipher: random substitution ciphers
   - For bit_manipulation: random bit operations
   - For equation_transform: random symbol mappings
   - For numeral_system: already 100% — skip or minimal data
   - Target: 10K-50K additional training examples

3. **Focus training on weak categories**
   - Weight bit_manipulation, cipher, equation_transform heavily (model is worst here)
   - Reduce/skip numeral_system in training (already 100% zero-shot)
   - Balance gravity and unit_conversion (model gets ~33% already)

### MEDIUM-TERM (Next week)
4. **Try external training** (if GPU quota is tight)
   - Train on Vast.ai or other cloud GPU
   - Upload only the LoRA adapter file to Kaggle
   - This is explicitly allowed by competition host

5. **Experiment with data augmentation strategies**
   - Vary the number of examples given in each puzzle prompt
   - Create harder variants (more complex operations, longer sequences)
   - Add "distractor" information to make model more robust

6. **Hyperparameter sweep**
   - Current: LR=2e-4, epochs=3, rank=32, alpha=32
   - Try: LR=1e-4 or 5e-5, epochs=5-10, alpha=64
   - Try longer sequences (4096 or 8192 to match eval)

### LONGER-TERM
7. **Multi-stage training**
   - Stage 1: SFT on distilled data (general puzzle-solving)
   - Stage 2: SFT on hardest examples (curriculum learning)
   - Stage 3: GRPO on verifiable puzzles (gravity, unit_conversion, numeral — where we can check answers)

---

## 6. KEY RESOURCES

### Datasets
- `nvidia/Nemotron-CrossThink` — Multi-domain reasoning (10M-100M examples)
- `nvidia/Llama-Nemotron-Post-Training-Dataset` — 18M+ training samples
- `nvidia/Nemotron-Post-Training-v3` — Latest post-training collection
- `nvidia/Nemotron-RL-Super-Training-Blends` — RL training blends

### Models for Distillation
- `nvidia/Nemotron-Cascade-2-30B-A3B` — Same arch, better trained (March 2026)
- DeepSeek-R1 — Best open reasoning model
- Qwen3-235B — Strong reasoning
- Any API-accessible model for generating training traces

### Public Notebooks to Study
- lucian kucera's "Zero shot inference exploration" notebook
- lucian kucera's "Bit rejection sampling" notebook
- barkataliarbab's "Structured Reasoning for NVIDIA NeMoTron"

### Papers
- Nemotron-Cascade 2 (arxiv:2603.19220, March 2026)
- Nemotron-CrossThink (arxiv:2504.13941)
- Nemotron 3 Nano Technical Report (Dec 2025)

---

## 7. COMPETITION TIMELINE
- **Now:** March 23, 2026
- **Midpoint prize:** April 9, 2026 (17 DAYS AWAY)
- **Final deadline:** June 15, 2026 (3 months)
- Leaderboard: 50% public, 50% private

### Midpoint Strategy
- 17 days to midpoint prize — need to maximize score by April 9
- Current gap to top: 0.10 (0.69 vs 0.79)
- Realistic target by midpoint: 0.75-0.78 (requires distilled data + augmented puzzles)
- Key bottleneck: training data quality, not model architecture

---

## 8. SUMMARY OF CHANGES TO TRY

| Priority | Change | Expected Impact | Effort |
|----------|--------|----------------|--------|
| 1 | Distill CoT from powerful model | +0.05-0.08 | Medium |
| 2 | Generate augmented puzzles | +0.03-0.05 | Medium |
| 3 | Skip numeral_system in training | +0.01-0.02 | Low |
| 4 | Weight weak categories higher | +0.01-0.03 | Low |
| 5 | External GPU for longer training | +0.01-0.03 | Medium |
| 6 | Hyperparameter tuning | +0.01-0.02 | Low |
| 7 | Multi-stage training pipeline | +0.02-0.04 | High |

**Combined realistic improvement:** 0.69 -> 0.76-0.80
