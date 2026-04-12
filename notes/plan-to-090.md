# Plan to 0.90 — Nemotron Reasoning Challenge

**Date**: 2026-04-11 evening IST
**Current**: v34c 0.67 Kaggle (best)
**Target**: 0.90 on public LB
**Top**: Tong Hui Kang 0.85 (Apr 10)
**Deadline**: June 15, 2026 (final), Apr 9 (midpoint prize, already locked)

## Scoring math
- 250 puzzles in public LB, each worth 0.004
- 0.67 → 0.90 = +0.23 = 58 more correct puzzles
- 0.85 (top) → 0.90 = +0.05 = 12.5 more correct puzzles

## Current stack
- **v35** (running, ~22:50 completion): v34c data - hallucinated equation + 13 verifier-derived equation, control baseline
- **v37** (running, ~05:00 IST completion): Kh0a's 2450 subsampled training data + Kh0a exact config (alpha=16 dropout=0.1 11 targets incl embed+lm_head LR=1e-4 2 epochs). Expected ≈Kh0a 0.73
- **v38** (staged, pushing after v35 frees slot): THK-sequence binary solver at 70.6% (1131 traces) + Kh0a others (1550). Expected 0.74-0.78
- **Research done**: `memory/path_to_090.md`, `notes/research-2026-04-11-to-090.md`, `notes/thk-bit-cot-full.txt`

## Critical blockers to 0.90
1. **symbol_transform (cipher-digit equation)** — nobody public solves this (Kh0a 0%, m4nocha 0%) — ~10% of test, caps public ceiling at ~0.85
2. **7680 token budget** — THK's bit scan alone uses 5130 tokens → no room for second algorithmic solver in same CoT
3. **Hidden test distribution** — may include math/GPQA/IFEval not matching puzzle training
4. **250-puzzle LB** — high variance regime; 2-3 puzzle swings normal

## Iteration plan (priority order)

### Step 1 — Submit v35 (imminent, ~22:50)
- Download adapter from notebook output
- Skip local validation (no RunPod budget)
- Submit via Playwright UI → Output tab → Submit to Competition
- Score: control baseline. Expected ~0.65-0.68.
- **Cost**: 1/5 daily submissions

### Step 2 — Wait for v37 (~05:00 IST)
- Download, submit
- Expected 0.71-0.73 (Kh0a-level)
- **Cost**: 1/5 daily submissions

### Step 3 — Push v38 while waiting
- As soon as v35 frees a GPU slot, push v38
- v38 trains ~7 hrs, completes ~06:00 IST
- Download, submit
- Expected 0.74-0.78 (Kh0a + THK binary partial)
- **Cost**: 1/5 daily submissions

### Step 4 — WAIT for THK Sunday drop (~12 hrs)
- Discussion 689915: THK publishes full solution Sunday Apr 12 UTC (~Mon Apr 13 05:30 IST)
- This IS the 0.85 code. Fork it, adapt config, run on our quota, submit.
- **Do NOT burn compute on v39 reimplementation** — his code will be strictly better
- Expected after fork: 0.82-0.85

### Step 5 (v40) — Improve on THK
Options, in order of expected delta:

**A. GRPO with Komil's fix (+0.02-0.04)**
- transformers≥5.3.0, drop trust_remote_code=True, gradient_checkpointing=False
- 20x speedup: 2 tok/s → 38 tok/s (500 steps now fits Kaggle 12hr)
- Rewards: per-bit correctness for binary + format check (`\boxed{}`) + strict string match
- Reference: discussion 690161 (Komil Parmar, 17 votes)
- Run on THK base, not v22

**B. 3-input gates for binary (+0.02-0.03)**
- THK's 85% doesn't cover MAJ/CHO/PAR3/AO/OA/AX/OX/XA/XO (Donald's list)
- Extend my v38 solver with these → target 90%+ binary coverage
- Still fits 5130-token budget

**C. symbol_transform breakthrough (+0.05-0.08)**
- Nobody public has solved cipher-digit equation
- Approach: simultaneous symbol→digit map cracking + 47-combo op scan
- Use constraint satisfaction (Z3 or simple search)
- High risk, high reward — would leapfrog THK

**D. Ensemble-by-distillation (+0.01-0.02)**
- Train single LoRA on traces from multiple teachers (Grok + DeepSeek-R1 + THK algo)
- Diversity helps generalization
- Needs API access + cost

### Step 6 (v41) — Beyond 0.85 stretch
- Teacher distillation pipeline: generate traces for v38's 470 unsolved binary rows via Grok-4 API, filter by `\boxed{}` correctness
- Hidden test prep: add ~500 GPQA/math/IFEval examples (Nemotron-Math-v2 from HuggingFace)
- Curriculum: sort training by solver difficulty, easy→hard

## Submission budget (daily resets ~18-24 hrs)
- Today used: 0
- Budget: v35 + v37 + v38 + THK fork + 1 buffer = 5/5 for Sunday
- If we run out: can re-submit same adapter for variance roll (0.65→0.69 seen historically)

## GPU budget
- Fresh 30hr quota today
- v35: ~1hr, v37: ~7hr, v38: ~7hr = 15hrs used
- Remaining for Sunday: 15hrs → fits THK fork + v40 GRPO

## Risk assessment
| Risk | Likelihood | Mitigation |
|---|---|---|
| THK doesn't publish Sunday | 20% | v37/v38 are fallback at 0.74-0.78 |
| Our v37 LR=1e-4 wrong (NemoSkills says 1e-5) | 30% | Kh0a proved 1e-4 works at 0.73, trust precedent |
| Concurrent runs contend for resources | 20% | Observed v35 slow due to v37 parallel — schedule sequentially when possible |
| symbol_transform unsolvable | 80% | Accept 0.85 ceiling until breakthrough |
| Kaggle submission bug / metric change | 10% | Keep buffer submission daily |

## Definition of done
- Minimum: **0.73** this session (matching Kh0a)
- Expected: **0.78-0.82** by Sunday (THK fork)
- Stretch: **0.85+** by Monday (THK + GRPO)
- Dream: **0.90** by deadline (symbol_transform breakthrough + all levers)

## Next concrete actions
1. ⏳ Wait for v35 save+zip (~5 min)
2. ⏳ Submit v35
3. ⏳ Push v38 (frees GPU slot after v35)
4. ⏳ Wait for v37/v38 completion
5. ⏳ Sunday Apr 12 UTC: check Kaggle discussion 689915 for THK drop
6. ⏳ Fork + adapt + run THK notebook
