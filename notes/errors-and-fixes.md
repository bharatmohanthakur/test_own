# Errors Encountered & Fixes

## Error 1: ModuleNotFoundError: No module named 'mamba_ssm'
- **When**: v1, v3 — loading Nemotron-3-Nano-30B
- **Cause**: Model uses Mamba-Transformer hybrid architecture, needs mamba_ssm package
- **Fix**: Add `ryanholbrook/nvidia-utility-script` as a kernel source (Utility Scripts input). Then `import mamba_ssm` before loading model.
- **Wrong fix**: `pip install mamba-ssm` — fails in no-internet GPU environment

## Error 2: ValueError: device_map had weights offloaded to disk
- **When**: v4 — model loaded on RTX Pro 6000 but couldn't offload
- **Cause**: 30B model in bf16 is ~60GB, GPU has 48GB VRAM, needs disk offload
- **Fix**: Add `offload_folder="/kaggle/tmp/offload"` to `from_pretrained()`

## Error 3: CUDA / PyTorch architecture mismatch on P100
- **When**: v5 and later CLI-started runs — model loaded on `Tesla P100-PCIE-16GB`
- **Cause**: P100 is `sm_60`, but the current PyTorch build only supports `sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`. CLI `kaggle kernels push` starts an automatic P100 run unless the notebook is re-saved in the Kaggle UI on `RTX Pro 6000`.
- **Typical log**:
  - `Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation`
  - `The current PyTorch install supports CUDA capabilities sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`
- **Fix**: ALWAYS use GPU RTX Pro 6000. After every CLI push, open the notebook in the Kaggle UI, switch GPU to RTX Pro 6000, then Save & Run.

## Error 4: FileNotFoundError: train.csv not found
- **When**: SFT v1-v3 — hardcoded wrong data path
- **Cause**: Competition data path varies depending on how inputs are mounted. Demo uses `/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv`, public notebooks use `/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv`
- **Fix**: Use glob to find: `glob.glob("/kaggle/input/*/train.csv") + glob.glob("/kaggle/input/*/*/train.csv") + glob.glob("/kaggle/input/*/*/*/train.csv")`

## Error 5: CLI push resets GPU and environment settings
- **When**: Every CLI push
- **Cause**: `kaggle kernels push` doesn't support `--accelerator` flag in current version (1.7.4.5). It resets GPU to P100 and environment to pinned.
- **Fix**: After every CLI push, open notebook in Kaggle UI (Playwright), switch GPU to RTX Pro 6000, switch environment to "Always use latest", then Save & Run from UI.

## Error 6: CLI push doesn't preserve kernel_sources
- **When**: Early pushes lost utility script input
- **Cause**: Actually it DOES preserve them if specified in kernel-metadata.json. The issue was the v3 UI save didn't carry over the CLI-set sources.
- **Fix**: Always include in kernel-metadata.json:
  ```json
  "kernel_sources": ["ryanholbrook/nvidia-utility-script"],
  "competition_sources": ["nvidia-nemotron-model-reasoning-challenge"],
  "model_sources": ["metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"]
  ```
  If missing after UI save, re-add via Add Input → Utility Scripts → NVIDIA Utility Script.

## Error 7: Kernel source adapter path mismatch
- **When**: `grpo_only_v27` version 1 — trying to reuse `bharatmohan/nemotron-sft-v22` as a `kernel_source`
- **Cause**: Kaggle mounted the notebook output under `/kaggle/input/notebooks/bharatmohan/nemotron-sft-v22/nemotron-lora-adapter`, but the script only searched shallower `/kaggle/input/...` paths.
- **Fix**: Add the explicit notebook path and deeper glob depths like `/kaggle/input/*/*/nemotron-lora-adapter` and `/kaggle/input/*/*/*/nemotron-lora-adapter`.

## Error 8: GRPO matmul shape mismatch after loading SFT adapter
- **When**: `grpo_only_v27` after loading the v22 adapter with `PeftModel.from_pretrained(...)`
- **Cause**: Mamba fast-path remained active on modules inside the adapter-wrapped model, leading to GRPO generation/scoring failure with:
  - `RuntimeError: mat1 and mat2 shapes cannot be multiplied (...x131072 and 2688x131072)`
- **Fix**:
  - After `PeftModel.from_pretrained(...)`, iterate through `model.modules()` and set `is_fast_path_available = False` wherever present
  - Keep `gradient_checkpointing=True` with `gradient_checkpointing_kwargs={"use_reentrant": False}`
  - Keep `use_vllm=False` and `remove_unused_columns=False` in `GRPOConfig`
