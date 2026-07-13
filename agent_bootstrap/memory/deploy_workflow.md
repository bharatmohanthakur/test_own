---
name: deploy_workflow
description: Exact step-by-step workflow to deploy a Kaggle notebook on RTX Pro 6000 — follow every time
type: feedback
---

## Deploy Workflow (MANDATORY — follow exactly)

### 1. Before writing code
- **NO pip install** in the script for any library not pre-installed on Kaggle
- If you need trl, datasets, wandb, etc.: add `dennisfong/nvidia-nemotron-offline-packages` as dataset source
- Install from offline wheel: `pip install --no-index --find-links=/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages/ trl`
- All 5 Blackwell fixes must be present (causal_conv1d, ptxas, mamba3, is_fast_path, rmsnorm)

### 2. kernel-metadata.json must have ALL sources
- competition: `nvidia-nemotron-model-reasoning-challenge`
- model: `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1`
- kernel_sources: `ryanholbrook/nvidia-utility-script`
- dataset_sources: training data + `dennisfong/nvidia-nemotron-offline-packages`

### 3. Data path — use comprehensive glob
- Kaggle mounts datasets at `/kaggle/input/datasets/USERNAME/SLUG/` NOT `/kaggle/input/SLUG/`
- Always glob both patterns
- NEVER fallback to raw train.csv — `sys.exit(1)` if data not found

### 4. Push + GPU switch
- **BEST**: `kaggle kernels push -p <dir> --accelerator NvidiaRtxPro6000` → runs directly on RTX Pro 6000!
  - This preserves ALL metadata (dataset sources, inputs) from kernel-metadata.json
  - No Playwright needed, no UI save that drops inputs
- **FALLBACK** (if CLI accelerator fails): Push via CLI, wait for P100 error, then Playwright UI switch
  - WARNING: UI save creates a new version that may DROP dataset inputs added via CLI metadata
  - If using Playwright, verify all inputs are still present before saving

### 5. Verify
- `kaggle kernels status` must show RUNNING
- Check again after 5 min — if still running, past model load = good
- Monitor every 5 min in background

**Why:** Wasted 5+ runs and hours of GPU quota from: trl pip install failing (no internet on RTX Pro 6000), wrong data path, forgetting GPU switch, using custom loop when SFTTrainer was requested.
**How to apply:** Follow this checklist for EVERY deploy. No shortcuts.
