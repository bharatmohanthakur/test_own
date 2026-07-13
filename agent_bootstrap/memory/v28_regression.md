---
name: v28 regression to 0.67
description: v28 = THK v26 + 538 crypt corrections + full 16720 SFT at LR=1e-4. Scored 0.67 (−0.18 vs v26). Packing without flash_attn_2 + LR too high.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Date:** 2026-04-18
**Score:** **0.67** (vs THK v26 = 0.85, regression **-0.18**)

## Config
- Base: THK v26 transformed adapter (loaded via PEFT with backbone.lm_head keys silently mis-matched)
- Data: thk_training_v3.jsonl (16720 examples, 538 crypt traces replaced with correct-answer 【】-format from `reasoning_cryptarithm_legacy`)
- Training: SFT on B200 192GB
  - r=32, alpha=32, dropout=0, 9 targets incl lm_head
  - LR=1e-4, cosine, warmup 3%
  - batch=8 × grad_accum=2 = 16 effective
  - max_seq=4096, **packing=True** (no flash_attention_2)
  - 638 packed steps = 1 epoch
  - 5h 55m on B200 Oregon, $20 cost
- Wandb: g59gsx9f
- Loss: 1.06 → 0.42 (clean convergence, 60% drop)
- Token accuracy: 91% → 94.7%

## Why 0.67 (hypotheses, ranked)

### #1 — LR too aggressive for refinement (MOST LIKELY)
- THK's LR=2e-4 was for FROM-SCRATCH training
- v27 used 2e-5 (too low, narrow data regressed anyway)
- v28 used 1e-4 (too high for refine on top of v26)
- Correct refinement LR is ~2e-5 to 5e-5 for adapters with full data
- 1e-4 × 638 steps moved weights far enough to drift from v26's learned distribution

### #2 — Packing without flash_attention_2
TRL warning was explicit:
```
Packing flattens batches into a single sequence, and Flash Attention is
the only known attention mechanism that reliably supports this. Using
other implementations may lead to cross-contamination between batches.
```
Using eager attention with packing → tokens from different examples attend to each other within the packed sequence → gradients corrupt across examples.

### #3 — lm_head LoRA reset then retrained
- THK's v26 has lm_head LoRA at `backbone.lm_head.*` (wrong HF path, silently ignored by vLLM but loaded as MISSING by PEFT)
- PEFT initialized lm_head LoRA to zero
- Trained 638 steps on 16720 examples at LR=1e-4 → lm_head LoRA diverged from what THK's data assumed
- This is the SAME lm_head issue as v27 but amplified 167× by more data + 5× by higher LR

### #4 — Legacy trace format mismatch
- 538 new crypt traces in `reasoning_cryptarithm_legacy` 【】 format with correct answers
- 1976 unchanged crypt traces in OLD 【】 format with WRONG "unknown" answers
- Model sees CONFLICTING training signals for crypt (correct vs wrong in same format)
- Not the primary cause but compounds the damage

## Score timeline
| Adapter | Score | Notes |
|---|---|---|
| THK v26 as-is | 0.85 | ✅ Still best |
| v27 (100 bit_manip, LR=2e-5) | 0.82 | Narrow data regression |
| **v28 (16720 crypt, LR=1e-4, packing)** | **0.67** | **Multiple compound issues** |
| v38a (our Kh0a) | 0.72 | Pre-THK best |

## Lessons

1. **For refinement LR ≤ 5e-5.** Never use THK's 2e-4 on top of an already-trained adapter.
2. **Don't use packing without flash_attention_2.** The TRL warning is real — not informational.
3. **lm_head mismatch requires explicit fix OR removal from target_modules.** Ignoring the warning and relying on data volume to compensate doesn't work.
4. **Replace ALL bad crypt traces, not just solvable ones.** Leaving the 1976 "unknown" wrong-answer traces creates contradictory training signal with the 538 correct ones.
5. **NEVER submit without local eval.** Would have caught this at step 100 (loss 0.55 but per-type accuracy crashed).

## Submissions used today
1. (none this UTC day yet)

## Path forward
v26 stays our best at 0.85. To beat it:

### Option A — v29 with conservative refinement
- Load v26, LR=2e-5 (NOT 1e-4)
- Disable packing OR enable flash_attention_2
- Remove lm_head from target_modules to avoid warning
- Train only 100-200 steps on the 538 crypt corrections + 200 of each other category (balanced minibatch)
- LOCAL EVAL at step 50 and 100 on cheap H100

### Option B — Accept 0.85 as final
- 0.85 ties THK's public score
- Top of leaderboard is 0.86-0.87 (minimal gap)
- Multiple retries didn't beat it; remaining effort may be better spent elsewhere

### Option C — Generate better traces
- Fix the 1976 remaining wrong-answer crypt traces
- Need either THK's new answer-guided solver (already tried, gets 293) OR a REAL CSP + template solver that cracks more
- Build training data v4 with ALL 823 crypt traces correct (if possible)
- Then retrain with v4 + conservative LR

## Budget used (Apr 17-18)
- Vast.ai: $24 → $0.81 ($23.19 on training + some overhead)
- Kaggle: free
- Nothing to show for the $23 spend — v26 remains our best

## Key files
- `runpod_v40/train_refine_thk_v28.py` — the training script (config preserved for post-mortem)
- `data/thk_training_v3.jsonl` — the mixed training set (538 corrections)
- `tracking/run_bench_fast.py`, `compare.py`, `submit_gate.py` — validation system built but never run
- `notebooks/submit_v28/` — CPU submit kernel
- `bharatmohan/nemotron-v28-crypt-adapter` — Kaggle dataset of v28 adapter
