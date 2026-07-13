---
name: THK v26 baseline verified
description: Submitted THK's raw v26 adapter with SVD transformation — scored 0.85 (tied THK). Our new floor.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Date:** 2026-04-16
**Score:** **0.85** (matched THK's public 0.85 exactly)
**Adapter:** bharatmohan/thk-nemotron-v26-raw (dataset) → bharatmohan/nemotron-submit-thk-v26 (kernel) → submission.zip

## How we got here

1. Downloaded THK's raw v26 adapter from `huikang/nemotron-adapter/transformers/default/26` via Kaggle API URL (CLI model-hub download 0-bytes bug avoided via direct URL trick)
2. Uploaded as our own Kaggle dataset `bharatmohan/thk-nemotron-v26-raw` (model_sources of other users' models don't auto-mount for us)
3. Built CPU-only submit kernel `notebooks/submit_thk_v26/` that applies THK's tinker-submission-notebook transformation:
   - Load 1.54 GB raw adapter (418 trained tensors, "all-linear" scope)
   - Read base-model shard shapes for in_proj dim lookup
   - Rename `base_model.model.model.*` → `base_model.model.backbone.*`
   - Unfuse MoE: w1 → per-expert up_proj, w2 → per-expert down_proj (×128 experts × 20 layers)
   - SVD-merge Mamba `gate_proj` + `x_proj` LoRA → `in_proj` LoRA (23 Mamba layers, 75-78% singular value retention)
   - Drop 23 empty `.experts.w3` LoRAs
   - Rewrite target_modules to 9 names INCLUDING `lm_head`
   - Zip: 3.26 GB submission.zip (3535 MB adapter + 761 B config)
4. `kaggle competitions submit` via CLI worked (3 min upload)
5. Scored 0.85 in ~20 min

## Key numbers from the transformation
- 418 trained tensors → 12,010 output tensors after MoE unfuse
- Base-model key count: 6,243
- Output size: 3.53 GB safetensors (2.3× raw because MoE unfuse duplicates experts)
- SVD retention per layer: 75-78% of Frobenius mass (acceptable; THK accepts this loss)

## Score timeline
| Date | Attempt | Score |
|---|---|---|
| Apr 12 | v38a SFT (Kh0a) | 0.72 (our prev best) |
| Apr 13 | GRPO Phase1+2 | 0.47 |
| Apr 14 | THK Unsloth refine | 0.66 |
| Apr 14 | B200 full SFT | 0.26 |
| **Apr 16** | **THK v26 as-is (transformed)** | **0.85** ✅ |

## New floor + path forward

- **New floor:** 0.85. We never submit anything worse than this again.
- **Target:** 0.87+ via bit_manip refinement (next step)
- **Stretch:** 0.93+ via cryptarithm CSP solver (758 unsolved problems)

## Reusable artifacts

- `/Users/bharat/Downloads/kaggle/adapters/thk_v26/` — THK's raw adapter (1.54 GB)
- `bharatmohan/thk-nemotron-v26-raw` — Kaggle dataset
- `bharatmohan/nemotron-submit-thk-v26` — CPU submit kernel (applies transformation)
- `/tmp/thk_v26_out/submission.zip` — final 3.26 GB submission.zip
- `notes/peft-nemotron-config-reference.md` — doc explaining every config field

## Submission log
Used 3/5 today (0.66, 0.26, **0.85**). 2 remaining.
