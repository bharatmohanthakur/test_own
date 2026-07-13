---
name: v31_v32_results
description: v31 GRPO and v32 GSPO training run pointers — where the adapters live, exact configs used, what was submitted. Apr 8, 2026.
type: project
---

# v31 (GRPO) and v32 (GSPO) — Apr 8, 2026

Both trained on RunPod RTX Pro 6000 Blackwell ($1.69/hr) when Kaggle weekly quota was exhausted. See `notes/runpod-grpo-workflow.md` for the full reproducible workflow.

## v31 — GRPO over v22 SFT adapter

**Where it lives:**
- Kaggle dataset: `bharatmohan/nemotron-v31-grpo-adapter` (3.3 GB)
- Submit notebook: `bharatmohan/nemotron-submit-v31` (CPU-only, no GPU quota)
- Local script: `notebooks/grpo_v31/train.py` and the RunPod variant at `runpod_v31/train_runpod.py`

**Training config:**
- Base: v22 SFT adapter (`bharatmohan/nemotron-sft-v22`, 0.65 score)
- Data: 179 Goldilocks prompts (166 Grok-distilled + 60 R1-distilled, deduped) — `bharatmohan/nemotron-training-v31`
- LoRA: r=32, alpha=64, target_modules=in_proj/out_proj/up_proj/down_proj (matches v22)
- GRPO: max_steps=10, save_steps=2, num_generations=2, batch=1, grad_accum=4, max_completion_length=512
- Reward: simple correctness (+2 if `\boxed{}` matches via new strict metric) + format (+0.5 if `\boxed{` present)
- LR=5e-6, temp=0.9, max_grad_norm=0.1
- Patches: v28 chunked-log-softmax monkey-patch + Mamba `is_fast_path_available=False` on imported modules AND on `model.modules()`
- Runtime: 10 steps × ~12 min = 121 min total. VRAM peaked at 94.5GB and stabilized.

**Submission:**
- Submitted to competition at 10:39 UTC, Apr 8 2026
- Status: PENDING (queue backed up)

## v32 — GSPO polish on top of v31 GRPO

**Where it lives:**
- Kaggle dataset: `bharatmohan/nemotron-v32-gspo-adapter` (3.3 GB)
- Submit notebook: `bharatmohan/nemotron-submit-v32` (CPU-only)
- Local script: `runpod_v31/train_v32_gspo.py` (single-flag diff from v31's runpod script)

**Training config:**
- Base: v31 GRPO adapter (loaded from `/workspace/output/grpo_output/checkpoint-10` on the pod)
- Data: 50 balanced samples from the 179 Goldilocks pool (28 binary + 12 cipher + 4 gravity + 3 numeral + 2 unit + 1 equation; sources: 17 hardest types + 17 longest + 16 random) — local at `data/training_v32/gspo_50.jsonl`
- Same LoRA architecture, same patches
- GSPO: **`importance_sampling_level="sequence"`** ← single-line change from GRPO
- max_steps=2, save_steps=1, num_generations=2, max_completion_length=512
- Same LR/temp/grad_norm as v31
- Runtime: 2 steps × ~12 min = 24 min total on the SAME pod (no model re-download)
- VRAM: peaked ~95GB, lower than GRPO baseline (sequence-level GSPO is leaner)

**Submission:**
- Submitted to competition at 11:12 UTC, Apr 8 2026
- Status: PENDING

## Cost
- Total RunPod spend for both runs + setup overhead + 2 dead pod attempts: **~$7 of $10 budget**
- RunPod balance after termination: $2.93

## Recovery notes (don't repeat)
- v31's post-train `model.save_pretrained(ADAPTER_OUT)` left `adapter_config.json` as 0 bytes despite `adapter_model.safetensors` being saved correctly. **Recovered from `output/grpo_output/checkpoint-10/`** which had the full valid adapter saved by the trainer's auto-save.
- I then accidentally truncated `checkpoint-10/adapter_config.json` to 0 bytes by running `with open(p, "w") as f: json.dump(...)` while the MooseFS quota was full (the open-for-write truncates BEFORE writing, and the json.dump crashed). **Recovered by copying `adapter_config.json` from `checkpoint-8/`** — they're identical for the same LoRA architecture.
- v32 saved cleanly because by then I had freed disk space and used `/root/work` for the staging.

## What we learned (regardless of submission scores)
- **RunPod RTX Pro 6000 Blackwell is a drop-in for Kaggle's RTX Pro 6000** — same architecture, same wheels (mayukh18 dataset), same patches, same VRAM profile. Cost ~$1.69/hr.
- **GSPO via single-line `importance_sampling_level="sequence"` flag** works on Mamba-30B without any code changes beyond what GRPO already needs.
- **v28 chunked-log-softmax monkey-patch is the unlock for Nemotron GRPO/GSPO on Unsloth.** Without it, training dies at the very first generation step.
- **Direct pod → Kaggle dataset push** (3.3GB in 27 sec) is the fastest way to land an externally-trained adapter.
- **CPU-only Kaggle submission kernel** (`enable_gpu: false`) burns zero GPU quota and runs in ~30 sec.
