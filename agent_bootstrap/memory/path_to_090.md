---
name: Path to 0.90 — research synthesis
description: Deep research findings on pushing past 0.85 (top) to 0.90 on Nemotron Reasoning Challenge. Sources: Kaggle discussions, NemoSkills, AIMO winners, THK writeup.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
# Path to 0.90 (researched Apr 11, 2026)

## Scoring math
- **LB has 250 puzzles** — each worth 0.004
- 0.02 gap = 10 puzzles
- 0.85 cluster = ~213/250 correct
- 0.90 = 225/250 correct — needs fixing 12 more puzzles than current top

## Tier 1: Known levers (additive to single adapter)

### 1. THK 354-combo bit_manipulation (+0.10-0.13)
- Discussion 690307 — posted 9 hrs ago, 35 votes
- Pure Python algorithm, no GPU needed for data gen
- 18 unary + 336 binary combos × ~15 tokens = 5130 token CoT
- Solves 85.1% of 1602 train rows (vs Kh0a 21.88%, our v38 70.6%)
- **My v38 solver missing 3-input gates**: MAJ, CHO, PAR3, AO/OA/AX/OX/XA/XO — adds ~15% coverage
- Full CoT saved: `notes/thk-bit-cot-full.txt` (845 lines)

### 2. m4nocha's eval config (+0.05)
- Discussion 689792/689840
- 0.72 LB / 81.34% local with Unsloth 16-bit
- Uses `max_model_len=7000, max_num_seqs=128, gpu_mem=0.9`
- Better than our default eval params

### 3. NemoSkills SFT hyperparams (unverified — Kh0a 0.73 uses different)
- NVIDIA-NeMo/Skills/recipes/openmathreasoning/scripts/simplified_recipe.py
- LR=1e-5, 2 epochs, batch=32, max_seq=8192, teacher=QwQ-32B temp=0.6
- **CONFLICT**: Kh0a's 0.73 uses LR=1e-4 (we matched in v37/v38). NemoSkills is multi-node config with big batch.
- **Decision**: trust Kh0a's proven 0.73 over NemoSkills speculative. Keep LR=1e-4.
- Revisit ONLY if v37 underperforms <0.70.

## Tier 2: Nemotron-specific

### GRPO speed fix (20x)
- Discussion 690161 (Komil Parmar, 17 votes)
- Root cause: `prepare_inputs_for_generation` passes cache as `past_key_values` but forward expects `cache_params` → cache silently dropped → 2 tok/s
- Fix: `transformers>=5.3.0` + **drop trust_remote_code=True** + `gradient_checkpointing=False`
- Result: 2→38 tok/s, 500-step GRPO now fits Kaggle session
- **v40 target**: 50-200 step GRPO on binary+symbol_transform with Donald's per-bit reward

### 3-input gates (needed for THK-level binary)
- MAJ(a,b,c) = (a+b+c)≥2
- CHO(a,b,c) = a ? b : c
- PAR3(a,b,c) = a ^ b ^ c
- AO, OA, AX, OX, XA, XO — 2-level compositions
- 4-input: AOA, OAO, PAR4, XX, AXA

## Tier 3: Uncrackable (hard ceiling)

### symbol_transform (cipher-digit equation) — nobody solves
- Kh0a 0/82, m4nocha 0/82
- Requires cracking symbol→digit map + operation scan simultaneously
- ~10% of test puzzles → caps max at ~0.85 without breakthrough

### Hidden test distribution unknown
- May include math/GPQA/IFEval beyond 6 puzzle types
- Base model capabilities set the floor

## Blockers to 0.90
1. **7680 token budget** — THK bit scan alone uses 5130 tokens, no room for second solver in same trace
2. **symbol_transform unsolved** by anyone public
3. **250 puzzles total** = small-sample high-variance regime

## Recommended plan (executing Apr 11)

### Immediate (tonight)
- v35 (running, equation fix) — fallback baseline
- v37 (running, Kh0a data + config) — Kh0a-level baseline, ETA ~05:00 IST
- v38 (staged, THK-style binary + Kh0a others) — push after v35 frees slot

### Sunday Apr 12 UTC (~12 hrs from research)
- **Wait for THK full solution publication** (disc 689915)
- Fork his notebook, adapt config to our constraints
- Expected: 0.82-0.85 single adapter

### Post-THK
- v40: GRPO with Komil's fix on THK base — target +0.02-0.04
- v41: ensemble via adapter fusion (train one adapter on multiple teacher traces)

## Key URLs
- THK bit algorithm: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307
- THK solution drop placeholder: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915
- Kh0a public notebook: https://www.kaggle.com/code/llkh0a/nemotron-unsloth-sft-training-3-30-2
- GRPO fix: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690161
- Full research report: `notes/research-2026-04-11-to-090.md`
- THK bit CoT sample: `notes/thk-bit-cot-full.txt`
