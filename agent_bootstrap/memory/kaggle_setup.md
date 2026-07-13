---
name: kaggle_setup
description: Kaggle notebook constraints - GPU quota, submissions, setup requirements, timing
type: feedback
---

## GPU Quota
- RTX Pro 6000: 30 hrs/week, resets Saturday 05:30 IST
- Model load: ~12 min. Baseline run: ~13 min. SFT 1 epoch: ~30-45 min.
- Budget: ~15-25 runs per week depending on training length
- CLI push with `--accelerator NvidiaRtxPro6000` runs directly on RTX Pro 6000! No UI switch needed. (Note: `gpu_rtx_pro_6000` does NOT work, must use `NvidiaRtxPro6000`)

## Submissions
- 5 per day, resets every ~18-24 hrs
- Scoring takes 15-30+ min (loads 30B model + runs inference)
- Don't waste on untested changes

## Required Notebook Setup
1. GPU: RTX Pro 6000 (UI only)
2. Environment: "Always use latest"
3. Inputs: competition + model + ryanholbrook/nvidia-utility-script
4. `import mamba_ssm` before model load
5. `offload_folder="/kaggle/tmp/offload"` in from_pretrained
6. `gradient_checkpointing=True` for training
7. `model.enable_input_require_grads()` with LoRA training
8. Data path: use glob (`/kaggle/input/*/train.csv` + deeper paths)

## Workflow
Edit locally → `kaggle kernels push -p <dir> --accelerator gpu_rtx_pro_6000` → Monitor → Submit via UI
No Playwright needed for deploy! The --accelerator flag sets GPU directly from CLI.

**Why:** 6+ failed runs learning these constraints. Every shortcut wastes GPU quota.
**How to apply:** Follow this checklist for EVERY notebook run. Plan runs to maximize score per GPU hour.
