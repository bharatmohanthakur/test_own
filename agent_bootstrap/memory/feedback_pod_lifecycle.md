---
name: Pod lifecycle — stop never destroy, reuse, split train/eval GPUs
description: Consolidated GPU-pod rules for Vast.ai / RunPod — violated repeatedly; every violation wastes $ and 15-30 min of re-setup. Replaces 6 separate feedback files.
type: feedback
---

Enforced by `.claude/hooks/block_destroy.sh`. This memory exists so the judgment/why is preserved.

## Rules

### 1. Stop, never destroy — unless user explicitly says so
- `vastai stop instance <id>` preserves data + packages at zero cost.
- `vastai destroy instance <id>` loses everything. Re-setup = 60 GB model + 3.5 GB adapter + mamba-ssm/causal-conv1d wheels + Blackwell patches = **15–30 min wasted** per destroy.
- Trigger phrases that mean destroy: "destroy it", "kill the pod", "shut it down", "terminate", "we're done with this pod", explicit approval after asking.
- **Default after training finishes**: stop, offer options. Don't destroy.
- **Default after eval finishes**: stop. User may want to iterate on top.
- **Low balance ($2-3 remaining)**: flag to user, don't auto-destroy.

### 2. Reuse pods between runs
- Don't destroy after each run. Upload new script, restart training on same pod.
- Saves both setup time AND money (you pay for time, not per-run).

### 3. `vastai copy` between instances, not re-upload from laptop
- `vastai copy OLD_ID:/workspace/adapters/v<N>/ NEW_ID:/workspace/adapters/` works on stopped instances too.
- 3.3 GB adapter transfer: 5 min via `vastai copy` vs 30 min via laptop re-upload.

### 4. Split train GPU vs eval GPU
- **Training (30B + LoRA + grad ckpt)**: B200 192G / H200 / RTX Pro 6000 Blackwell. $0.92–3.44/hr. Pay for VRAM headroom.
- **Eval (load + vLLM batch)**: H100 SXM / A100 80G. $0.80–1.50/hr. CUDA 12.6 vLLM works (Hopper sm_90).
- **NEVER eval on B200**: sm_100a flashinfer broken, falls back to HF `.generate()` 10-100× slower. Burned $2+ on v27 this way.
- Sequential workflow: train on B200 → save adapter to Kaggle dataset → stop B200 → spin up cheap H100 → run vLLM eval → scp results → stop H100.
- **Exception**: if train + eval together fit in ~45 min on one pod AND CUDA version is right for vLLM, one-pod OK. Verify CUDA 12.8 before starting.

## Why (incidents)
- **2026-04-16 v27**: trained on B200, kept same B200 for eval. Image had CUDA 12.6, vLLM flashinfer failed to JIT-compile for compute_100a, fell back to `.generate()` (10-100× slower), eval stalled, burned $2+, wasted a submit slot deciding blind.
- **Multiple sessions**: claude destroyed pods right after training, forcing 30-min re-setup when user wanted one more iteration with tweaked config.
- **mamba-ssm from source**: 15-20 min compile every fresh pod. Reusing pods avoids this entirely.

## How to apply
- After training: `vastai stop`, then ask user what's next.
- Between runs on same workload: upload new script via `vastai copy` or scp, restart on same pod.
- Poisoned / broken pod: create new, `vastai copy` data over, then destroy old only after copy confirms.
- Monthly housekeeping: ask user "can I destroy stopped pods over 7 days old?" — don't assume.

## Bypass for legitimate destroy
- User says destroy → OK
- Or: set `KAGGLE_ALLOW_DESTROY=1` env before running vastai destroy (tells the hook to allow).

## Merged from (now in archive/)
Originally 6 files. Content merged here; originals preserved at:
- feedback_pod_destroy.md
- feedback_stop_not_destroy.md
- feedback_reuse_gpu.md
- feedback_split_train_eval_gpus.md
- feedback_vastai_copy.md
- pod_workflow_feedback.md
