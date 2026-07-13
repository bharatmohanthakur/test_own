---
name: Research Apr 11 2026 — top scorer intel
description: Critical competition intel gathered Apr 11 — Tong Hui Kang drops full solution Apr 12 UTC; his bit_manipulation algo public; Kh0a 0.73 public notebook config; GRPO speed fix.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
# Research Findings — April 11, 2026

## 🔥 IMMEDIATE ACTION: Tong Hui Kang solution drops Sunday Apr 12 UTC
- Discussion 689915 (59 votes): THK (1st place, 0.85) promised full notebook + data this Sunday (~Mon Apr 13 05:30 IST)
- **DO NOT burn GPU before this** — adapt immediately when it drops
- Until then: implement his already-public bit_manipulation algorithm

## THK bit_manipulation algorithm (already public)
- **Discussion 690307** (35 votes) + gist `tonghuikang/6312fb04a0149c9a3e28bd7c2a844e9a`
- Claims **85% binary solve rate** vs our 25% (v34c)
- Pure Python, no GPU needed
- ~5130 token CoT (long — budget ~8192 max_seq_len if using full trace)
- Approach: expanded 2-input gate search:
  - Identity(j), NOT(j), Constant 0/1
  - AND, OR, XOR (commutative pairs)
  - **AND-NOT, OR-NOT, XOR-NOT (NON-commutative pairs)** ← we were missing
- Per-bit derivation: for each output bit k, try all ~270 candidate rules, pick first match
- My implementation hit 60% solve rate (954/1602) with this gate set — gap to 85% likely from 3-input gates (MAJ, choice) I haven't added

**Why:** Binary alone is 1/6 of the benchmark. Going from 25%→85% on binary = +0.10 score. 0.67→0.77 closes most of the gap.
**How to apply:** Generate 50-100 THK-style binary traces as v36 training data. Keep binary ratio < 15% total to avoid v34's cipher poisoning threshold.

## Kh0a (0.73 public notebook) config
- URL: https://www.kaggle.com/code/llkh0a/nemotron-unsloth-sft-training-3-30-2
- MAX_SEQ_LEN=3500
- LR=1e-4
- 2 epochs
- batch_size=2
- **11 target_modules including `embed_tokens` and `lm_head`** ← WE'VE BEEN MISSING THESE
- Unsloth auto-adds MoE expert LoRA when target_modules includes `gate_proj` ← we miss this too
- **5x weighted cross-entropy on tokens AFTER the last `\boxed{` marker** via custom compute_loss_func
- 888M trainable params = 2.74% of 32B

**Why:** Adding embed_tokens+lm_head + MoE expert LoRA = ~10x more trainable params. The 5x weighted loss forces the model to nail answer format.
**How to apply:** v36 train.py should use 11 target modules: `["q_proj","k_proj","v_proj","o_proj","in_proj","out_proj","up_proj","down_proj","gate_proj","embed_tokens","lm_head"]`. Adopt LR=1e-4, 2 epochs. Defer weighted loss until validated simpler changes work.

## GRPO speed fix (Komil Parmar, discussion 690161, 16 votes)
- Nemotron has a param-name bug: `prepare_inputs_for_generation` passes `past_key_values` but `forward()` expects `cache_params`
- Slowdown: 2 tok/s vs 38 tok/s (~20x)
- **Fix:** `transformers>=5.3.0` + **drop `trust_remote_code=True`** + `gradient_checkpointing=False`
- Upload transformers 5.3.0 wheel as Kaggle dataset

**Why:** Our GRPO attempts (v28, v31) were agonizingly slow; this could be the unlock.
**How to apply:** Defer until v36 SFT base proves improved. If we retry GRPO, use this fix.

## Negative findings (rule out these pivots)
- **Metric rescore is COMPLETE** — old scores are final; no free lift from rescore
- **Puzzle-KD-v2 has NO puzzle split** — just code/math/stem/chat (808k rows). v36's prior "puzzle-heavy Puzzle-KD mix" idea is misguided — there's no puzzle data in that dataset to upweight.
- **NeMo-RL nano-v3 recipe is MULTI-NODE (32×8 GPUs)**, not single-GPU. The memory `nemo_rl_single_gpu_recipe.md` mislabels it — don't try it on single RunPod.
- **NVIDIA Nemotron-Post-Training-v3** collection has no April updates.

## Metric edge cases to check
- Answers starting with `}` break the eval regex (discussion 689580)
- `\boxed{}` cannot contain `}` (discussion 689284)
- Check any v35/v36 output for these formats

## Session plan
1. ✅ v35 running (incremental equation fix) — let it finish, use as control baseline
2. 🔄 v36 prep: v34c saturated + 13 equation + 50 THK-binary + Kh0a 11-target config + LR=1e-4
3. ⏸ Wait for THK Sunday drop
4. ⏸ v37 from THK's full solution
