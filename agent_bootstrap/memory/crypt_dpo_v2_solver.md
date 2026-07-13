---
name: Crypt-DPO v2 solver pairs
description: Dead end. 242 real solver-verified positive DPO pairs from CSP solver + 100 steps DPO from step100 → crypt 2/10 on ALL 4 checkpoints. Same as v26 and step100. Matches prior failure pattern.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Run date**: 2026-04-22. Total cost ~$7 (pod setup + model download + training + bench).

## What we tried
- Init from dpo_step100 (not v26)
- Chosen pairs: **292 crypt prompts** where CSP solver got correct answer (from `thk_nemotron/bench_full_crypt_results_t30.jsonl`, `crypt_traces.jsonl`)
- Rejected pairs: step100 greedy on same 292 prompts (50 dropped where step100 happened to get it right)
- Final: 242 crypt pairs + 31 preserve pairs
- DPO: LR 2e-6, beta 0.05, 100 steps, save every 25, max_length 2048

## Training metrics
- Step 10: loss 0.17, acc 1.0 (already saturated)
- Step 25: loss 0.05
- Step 50: loss 0.01
- Step 100: loss 0.005

## Bench results (60 prompts)

| type | step100 | v2_ckpt25 | v2_ckpt50 | v2_ckpt75 | v2_ckpt100 |
|---|---|---|---|---|---|
| binary | 9 | 9 | 10 | 9 | 9 |
| cipher | 10 | 10 | 10 | 10 | 10 |
| **crypt** | **2** | **2** | **2** | **2** | **2** |
| gravity | 7 | 8 | 7 | 8 | 8 |
| roman | 10 | 10 | 10 | 10 | 10 |
| unit | 5 | 5 | 5 | 5 | 5 |
| **total** | 43 | 44 | 44 | 44 | 44 |

## The definitive crypt-DPO dead end

Crypt stays at 2/10 regardless of:
- DPO from v26 with sampled-v26 pairs (v1 Apr 20 → crypt 2)
- DPO from step100 with 5 sample-positive + 251 oracle pairs (v1-v2 Apr 20 → crypt 2)
- **DPO from step100 with 242 SOLVER-VERIFIED positive pairs (v2 Apr 22 → crypt 2)**

Even with the cleanest possible training signal (correct CoTs + actual greedy rejections), the inference path doesn't shift.

**Why:** Model's crypt failure is a deterministic policy attractor. Teacher-forced training (DPO on pair likelihoods) teaches the model to *rate* correct sequences higher, but greedy decoding still lands in the "concat-default on unknown operator" well. Token-level preference optimization doesn't move the deterministic decoding path for this specific failure mode.

## What to try next (future sessions)

- **Inference-time tool use**: Load CSP solver at Kaggle eval time, detect crypt prompts, inject solver answer directly into \boxed{}. Sidesteps the model entirely for crypt.
- **SFT on solver CoTs with token-priority loss** (instead of DPO): force the model to *produce* these tokens, not just *rate* them higher.
- **Ensemble scoring**: run the model + run CSP solver in parallel, pick higher-confidence answer.

## How to apply
- Stop attempting DPO/RL to fix crypt. It won't work at this scale.
- v26 (0.85 Kaggle) remains our best. Path to 0.86+ requires a non-training intervention.
