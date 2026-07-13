---
name: kaggle_submit_workflow
description: Submit a LoRA adapter to the Nemotron competition without burning Kaggle GPU quota — push adapter as Kaggle dataset, run a CPU-only kernel that wraps it as submission.zip, click Submit via Playwright UI. Validated Apr 8, 2026 with v31 and v32.
type: reference
---

# Submitting an externally-trained adapter to Nemotron Reasoning Challenge

**Why this matters**: every "Submit to Competition" button click on a Kaggle kernel uses GPU quota by default. For a kernel that does nothing but `cp + zip`, that's a waste. This workflow uses zero GPU quota by setting `enable_gpu: false` in the kernel metadata.

## Step 1: Push adapter as Kaggle dataset (from external GPU pod or local)

```bash
mkdir -p staging
cp adapter_config.json staging/
cp adapter_model.safetensors staging/
cat > staging/dataset-metadata.json << EOF
{"title":"Nemotron VXX Adapter","id":"bharatmohan/nemotron-vXX-adapter","licenses":[{"name":"CC0-1.0"}]}
EOF
cd staging && kaggle datasets create -p .
# Upload speed: ~3.3GB in 27 sec from RunPod, ~3 min from local
```

**Caveat**: After `kaggle datasets create`, the dataset takes ~60-90 sec to be queryable as a kernel `dataset_sources`. The first `kaggle kernels push` will warn `not valid dataset sources` and run without the input. Wait 90 sec and re-push (`Kernel version 2`).

## Step 2: CPU-only submit kernel

`notebooks/submit_vXX/kernel-metadata.json`:
```json
{
  "id": "bharatmohan/nemotron-submit-vXX",
  "title": "Nemotron Submit vXX",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_internet": "false",
  "dataset_sources": ["bharatmohan/nemotron-vXX-adapter"],
  "competition_sources": ["nvidia-nemotron-model-reasoning-challenge"]
}
```

`notebooks/submit_vXX/train.py`:
```python
import os, glob, json, zipfile, shutil

OUT = "/kaggle/working"
ADP = os.path.join(OUT, "nemotron-lora-adapter")
os.makedirs(ADP, exist_ok=True)

# Walk /kaggle/input to find adapter (paths vary by mount style)
src = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "adapter_model.safetensors" in files and "adapter_config.json" in files:
        src = root
        break
assert src, "no adapter found in /kaggle/input"

for fname in ["adapter_config.json", "adapter_model.safetensors"]:
    shutil.copy(os.path.join(src, fname), os.path.join(ADP, fname))

# Patch config for inference
cfg_path = os.path.join(ADP, "adapter_config.json")
with open(cfg_path) as f: cfg = json.load(f)
cfg["inference_mode"] = True
cfg["lora_dropout"] = 0.0
with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)

# Zip
zip_path = os.path.join(OUT, "submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in ["adapter_config.json", "adapter_model.safetensors"]:
        zf.write(os.path.join(ADP, fname), fname)
print("DONE")
```

```bash
kaggle kernels push -p notebooks/submit_vXX
# Wait for status COMPLETE (usually 30 sec to 2 min for CPU)
kaggle kernels status bharatmohan/nemotron-submit-vXX
```

## Step 3: Click "Submit to Competition" via Playwright (kaggle CLI can't do this)

`kaggle competitions submit -f file.zip` requires a LOCAL file. Our submission.zip is on Kaggle's side. The kaggle CLI has no submit-from-kernel command. We MUST use the UI:

```javascript
// Navigate to /code/USER/SLUG/output
// Click "Submit to Competition" button
// Click "Submit" in the modal dialog
// Page redirects to /competitions/.../submissions
```

Verified working with Playwright via DOM selectors. See `notes/runpod-grpo-workflow.md` step 12.

## Why not download submission.zip to local and `kaggle competitions submit`?

Because `kaggle kernels output` returns 0 bytes for large `.safetensors` files (errors_fixes #21, persistent bug). The submission.zip on Kaggle's side is fine — it's only the local download that's broken. So we have to submit from Kaggle's side directly.

## Daily limit
5 submissions per day. Resets at midnight UTC. Use the `kaggle competitions submissions` CLI to check today's count.
