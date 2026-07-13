---
name: v26 vs v28 per-type bench
description: 60-prompt vLLM bench on Kaggle RTX 6000 — v28 lost 9/10 on BINARY (not crypt) despite only adding crypt traces. Points to LR/packing/lm_head, not data.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Date:** 2026-04-18
**Script:** `notebooks/bench_v28_v26_vllm/train.py` (Kaggle GPU, vLLM, 60 prompts, temp=0, max_tokens=7680)

## Results (local 60-prompt bench)

| Type | v26 | v28 |
|---|---|---|
| binary | **9/10 → 0/10** | catastrophic regression |
| cipher | 10/10 | 10/10 |
| cryptarithm | 2/10 | 0/10 |
| gravity | 7/10 | 8/10 |
| roman | 10/10 | 10/10 |
| unit | 6/10 | 5/10 |
| **total** | 44/60 (0.733) | 33/60 (0.550) |

Kaggle scores: v26=0.85, v28=0.67. Local ↔ Kaggle delta is consistent (~+0.12 offset, within ±0.01 noise).

## Key finding
v28's regression was NOT primarily on crypt (the category we "fixed"). Binary went from **9/10 → 0/10** while we didn't touch binary data at all. This rules out "data was bad" and points to **training dynamics**:

- **LR=1e-4 too high** for refinement from v26 (THK used 2e-4 from-scratch; refinement needs ≤5e-5)
- **Packing without flash_attention_2** — TRL warns this cross-contaminates; gradients from cipher/roman examples likely polluted binary-reasoning weights
- **lm_head LoRA reset + retrained** — v26 had wrong-path lm_head (`backbone.lm_head.*`); PEFT silently initialized to zero then trained 638 steps at LR=1e-4

## For v29 (from base, not refine)
- LR=2e-4 is OK (from-scratch, matches THK)
- **Packing risk remains** — if v29 regresses binary similarly, the root cause is packing
- lm_head is freshly initialized (from base), so the v26-specific lm_head bug doesn't apply

## Bench workflow (validated)
- `bench_v28_v26_vllm/train.py` applies inline SVD/rename transform for v26-raw → HF-compat before LoRARequest
- Uses `VLLM_WORKER_MULTIPROC_METHOD=fork` + `VLLM_ENABLE_V1_MULTIPROCESSING=0` (Kaggle scripts can't spawn)
- mayukh18 wheels for offline install
- `--accelerator NvidiaRtxPro6000` CLI flag locks GPU (bypasses UI pain)
- Full eval completes in ~20 min on RTX 6000 Blackwell 102GB
