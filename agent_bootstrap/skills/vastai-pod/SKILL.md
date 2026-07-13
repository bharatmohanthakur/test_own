---
name: vastai-pod
description: Vast.ai / RunPod pod lifecycle rules. Use when creating, using, or ending a GPU rental. Enforces stop-never-destroy, reuse between runs, split train/eval GPU selection, and vastai copy between pods instead of re-upload.
---

# vastai-pod — GPU pod lifecycle

## THE RULE: stop, never destroy (unless user explicitly says so)

Re-setup cost: 15–30 min (base model 60 GB + adapter 3.5 GB + wheels + Blackwell patches). Every unnecessary destroy burns time AND money.

| User says | Action |
|---|---|
| "done for now" / "finished training" | `vastai stop instance <id>` |
| "destroy it" / "kill pod" / "terminate" / explicit approval | `vastai destroy instance <id>` |
| Ambiguous / nothing said | **KEEP RUNNING**, report status, ask before stopping |

Between training runs on the same workload: **reuse the pod**. Upload new script via `vastai copy`, restart training. Don't destroy.

## Split train vs eval GPUs

Train and eval are different workloads. Don't burn premium GPU on eval.

| Workload | GPU | $/hr | Why |
|---|---|---|---|
| Train 30B + LoRA + grad ckpt | B200 192G / H200 / RTX Pro 6000 Blackwell | $0.92–3.44 | Need VRAM headroom |
| Eval (load base + LoRA, vLLM batch) | H100 SXM 80G / A100 80G | $0.80–1.50 | Eval fits in 40–60 G, CUDA 12.6 vLLM works |
| **NEVER eval on B200** | — | — | sm_100a flashinfer broken, falls back to HF .generate() 10-100× slower |

Sequential workflow: train on B200 → save adapter to Kaggle dataset → `vastai stop` the B200 → spin up cheap H100 → download adapter → run vLLM eval → produce bench JSON → scp locally → stop H100.

## Data transfer: use `vastai copy`

Don't destroy + re-upload. `vastai copy` works between instances (incl. stopped ones):
```bash
vastai copy OLD_ID:/workspace/adapters/v<N>/ NEW_ID:/workspace/adapters/
```
Takes ~5 min for 3.3 GB vs 30 min re-upload from laptop.

## Startup checklist (new pod)

1. Image: `pytorch/pytorch:2.5.1-cuda12.8-cudnn9-devel` (Blackwell needs CUDA 12.8)
2. `pip install` from `mayukh18` offline wheels (see `runpod_workflow.md`) — don't build mamba-ssm from source every time (15 min waste)
3. Apply Blackwell patches (see `blackwell_training_fix.md`) BEFORE any training import
4. `wandb login` → verify success
5. `vastai show instance <id>` — confirm GPU is what you paid for (UI sometimes mismatches)

## Cost guardrails

- **Flag at $5/hr total spend**, ask before continuing
- **Flag at $2-3 pod balance remaining** (Vast.ai) — don't auto-destroy, just tell user
- **Never leave B200 idle** — $3.44/hr is $82/day. Stop immediately after training completes.

## Common mistakes (don't repeat)

- ❌ `vastai destroy` after first training run → had to re-setup for next run (30 min wasted)
- ❌ Using B200 for eval → flashinfer failed, eval stalled, burned a submit slot deciding blind (v27)
- ❌ Re-uploading 3.3 GB adapter from laptop → 30 min vs `vastai copy` 5 min
- ❌ Skipping wandb because "quick test" → run crashed at step 200 with no recovery possible

## Memory pointers
- `runpod_workflow.md` — validated RunPod GRPO pipeline
- `vastai_h200_workflow.md` — H200 setup (uv py3.12, Komil override)
- `cloud_gpu_strategy.md` — B200=Blackwell=Kaggle fixes apply directly
- `feedback_pod_destroy.md`, `feedback_stop_not_destroy.md`, `feedback_reuse_gpu.md`, `feedback_vastai_copy.md`, `feedback_split_train_eval_gpus.md`
