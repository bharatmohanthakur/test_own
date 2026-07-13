---
name: data_quality_findings
description: Critical findings about training data quality — what works and what doesn't
type: feedback
---

## Data Quality is THE Lever (confirmed by multiple experiments)

### Score vs Data Size
- 504 verified examples = 0.69 (v7, best under OLD metric, ~0.65 under new strict metric)
- 1699 mixed examples = 0.53-0.58 (v19, v20)
- 9500 raw examples = 0.53 (v15, v18)
- **500 Donald-templated examples alone = 0.52** (v29, Apr 7) — dropped 0.13 from v22
- **Less curated > more raw** — confirmed 5+ times

### What Kills Score
- "Other" category data (math/general): 498 examples in v18 data diluted LoRA capacity
- Poisoned data: 50.5% of bit_manipulation, 49% of equation_transform traces are WRONG
- Short/shallow CoT: v18 cipher CoT was 585 chars vs v7's 1307 chars
- **Rigid hard-labeled steps alone** (SOLVE/VER/APPLY/ANS, Donald style) without mixing verbose CoT → crowds out general reasoning. v29 dropped to 0.52.
- **Missing a puzzle type entirely**: v29 omitted equation_transform (~1/6 of eval) → ceiling 0.69 instantly
- **Templates that only cover a subset of a type**: binary template handling only 1+2 input gates means 42% of binary rows get untrained-base behavior

### What Works
- 504 puzzle-only examples, ~85 per category, balanced
- Detailed step-by-step CoT (1000+ chars) showing full derivation
- Verified against ground truth answers
- **All 6 puzzle types represented** — dropping any type tanks the score immediately
- **MIX styles** — structured templates + verbose CoT together, don't replace one with the other (hypothesis for v30)

### Teacher Model Quality
- Grok 4.20-reasoning: ~70% accuracy, solves bit_manipulation (hardest category)
- DeepSeek R1: ~20% accuracy, fails most hard puzzles
- Hand-crafted solvers: variable quality, some categories wrong
- **Grok is best teacher discovered so far**

### SFT Ceiling
- Community confirmed: ~0.76 with SFT alone (tamura_aicon)
- Our best: 0.69 — room for +0.07 with better data
- Beyond 0.76 needs GRPO/RL

**Why:** Data quality > data quantity > training method > hyperparams. This hierarchy held throughout all experiments.
**How to apply:** Always verify training data against ground truth. Weight toward weak categories. Use Grok 4.20 for distillation.
