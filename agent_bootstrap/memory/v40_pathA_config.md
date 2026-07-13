---
name: v40-patha-config
description: "v40 Path A training config — fresh LoRA from base with embed_tokens, used to attempt 0.85→0.88+ Kaggle"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---

v40 = "Path A big structural change" attempt at beating v26's 0.85 Kaggle.

**Config that fit on RTX Pro 6000 Blackwell (95 GB):**
- Fresh LoRA from base NemotronH-3-Nano-30B (NOT init from v26).
- r=64, lora_alpha=128.
- target_modules = mamba (in_proj/out_proj) + attention (q/k/v/o) + MLP (up/down/gate) + **embed_tokens**.
- lm_head DROPPED from LoRA targets — `embed_tokens + lm_head` together OOM'd at max_seq=2048.
- max_seq=1536 (2048 OOM'd even with batch=1).
- per_device_batch=1, grad_accum=16 → effective batch 16.
- gradient_checkpointing=False (NemotronH does not support it — raises ValueError).
- 2 epochs, lr=2e-5, warmup_ratio=0.03, cosine.
- Training data: `data/v40_training_corpus.jsonl` — 2,677 rows, 100% boxed-after-think.
- Trainable params: 55.4M (0.175% of 31.6B).
- ~12 s/step → 335 steps × 12s ≈ 67 min on RTX Pro 6000.

**Why:** Diag 2 (prior session) showed v26 has rare-symbol embedding bottleneck on crypt. embed_tokens LoRA is the architectural lever to fix it. v33 (same data tweak path) already failed to beat v26 (0.85 on Kaggle 2026-05-10), so we needed a structural change.

**How to apply:** If v40 wins, the recipe = embed_tokens-in-LoRA at max_seq=1536, batch=1×accum=16 on a 95 GB GPU. If it loses, drop the embed_tokens hypothesis and look at lm_head LoRA at a smaller max_seq, or move to B200 to bring max_seq=2048+lm_head back.

Run logs: wandb project `nemotron-v40`. Pod: 38935791 (ssh7:15790).
