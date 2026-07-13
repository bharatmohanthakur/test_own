---
name: nemotron-deploy
description: Deploy a Kaggle notebook for training or bench on RTX Pro 6000. Use when pushing a notebook to Kaggle. Covers 7 code preflight checks + 5 UI checks (GPU switch, internet toggle, inputs, save). Prevents the CLI-defaults-to-P100 silent failure.
---

# nemotron-deploy — Kaggle notebook push + UI checklist

## THE TRAP: `kaggle kernels push` silently defaults to P100 (16 GB) — fails silently on 30B model

Every push MUST be followed by UI GPU switch. No exceptions.

## 7 code preflight checks

Before `kaggle kernels push`, grep the notebook's `train.py`:

1. ✅ `import mamba_ssm` has offload fix applied (`blackwell_training_fix.md` step 1)
2. ✅ `offload_folder` arg passed to `from_pretrained` (fixes error #4 "offload_folder required")
3. ✅ `gradient_checkpointing=True` in TrainingArguments (30B OOMs without)
4. ✅ `max_seq_length >= 4096` (shorter truncates `\boxed{}`)
5. ✅ `target_modules="all-linear"` (NOT just QKV — history: v8 regressed)
6. ✅ `report_to="wandb"` with `os.environ["WANDB_API_KEY"]` loaded
7. ✅ Adapter save path = `/kaggle/working/adapter_v<N>/` (NOT `/tmp/`)

## kernel-metadata.json template

```json
{
  "id": "bharatmohan/nemotron-sft-v<N>",
  "title": "Nemotron SFT v<N>",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "false",
  "enable_gpu": "true",
  "enable_internet": "true",
  "dataset_sources": [
    "bharatmohan/nemotron-training-data-v<N>",
    "bharatmohan/nemotron-base-model"
  ],
  "competition_sources": ["nvidia-nemotron-model-reasoning-challenge"],
  "kernel_sources": []
}
```

## Push command

```bash
cd notebooks/sft_v<N>/
kaggle kernels push -p .
```

## 5 UI checks (MANDATORY — via Playwright or manual)

After push, open `https://kaggle.com/code/bharatmohan/nemotron-sft-v<N>/edit`:

1. **Accelerator → NvidiaRtxPro6000** (NOT P100, NOT T4). This is the #1 silent failure.
2. **Internet ON** (wandb needs it; without, trains blind)
3. **Inputs show all 3 sources** (base model, training data, competition). If any missing, re-push after fixing metadata.
4. **Save & Run All** (NOT "Save Version — Quick Save"). Quick Save doesn't execute.
5. **Verify status = "Running"** after save click. If "Failed" in < 2 min, open logs.

Playwright sequence: see `CLAUDE.md` → "Playwright UI Automation Reference" (21-step checklist).

## CLI shortcut (bypasses UI for some fields, NOT GPU type on all regions)

```bash
kaggle kernels push -p . --accelerator NvidiaRtxPro6000
```
Works on some account regions; verify via UI anyway. Never trust CLI-only for 30B.

## Post-push monitoring

```bash
watch -n 60 'kaggle kernels status bharatmohan/nemotron-sft-v<N>'
```
- Running: wait (typically 2–4 hrs for SFT, 3+ hrs for GRPO)
- Complete: fetch output, move adapter to `adapters/v<N>/`
- Failed: `kaggle kernels output bharatmohan/nemotron-sft-v<N>` → read logs

## Common errors + fixes (short list; full list in `errors_fixes.md`)

| Error | Fix |
|---|---|
| mamba_ssm import fails | Blackwell patch step 1 (`is_fast_path_available = False`) |
| "offload_folder required" | Pass `offload_folder="/kaggle/working/offload"` to `from_pretrained` |
| `causal_conv1d` CUDA mismatch | Patch same as mamba: `is_fast_path_available = False` |
| Adapter saved as 0 bytes | Don't use `trainer.save_model()` — use `model.save_pretrained()` directly |
| "Cannot find adapter" after submit | Search `/kaggle/input/notebooks/<owner>/<slug>/adapter_v<N>/` — 4 depth levels |
| GRPO matmul error | Disable Unsloth's text fast-path OR use native TRL stack |

## After training completes

1. Adapter ends up in `/kaggle/working/adapter_v<N>/`
2. Kaggle autosaves as notebook output → available at `kaggle kernels output`
3. Push adapter as a separate dataset for submission use: `kaggle datasets version -p adapter_v<N>/ -m "v<N> adapter"`
4. Run local bench via `kaggle-submit` skill BEFORE using submit slot

## Memory pointers
- `deploy_workflow.md` — full 12-step checklist
- `blackwell_training_fix.md` — 5-step patch
- `kaggle_setup.md` — GPU quota, metadata
- `kaggle_gpu_quirk.md` — CLI defaults to P100 trap
- `errors_fixes.md` — 28 documented bugs + fixes
- `logging_policy.md` — wandb mandatory
