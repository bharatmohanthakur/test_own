"""
CPU-only submission wrapper for v31 GRPO adapter.
- Takes Kaggle dataset bharatmohan/nemotron-v34c-sft-adapter as input
- Copies adapter files
- Builds submission.zip
- Zero GPU quota used (enable_gpu: false in metadata)
"""
import os, glob, json, zipfile, shutil

OUTPUT_DIR = "/kaggle/working"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "nemotron-lora-adapter")
os.makedirs(ADAPTER_DIR, exist_ok=True)

# Find adapter files in any input dataset
print("Searching for adapter files...")
candidates = []
for pat in [
    "/kaggle/input/nemotron-v34c-sft-adapter",
    "/kaggle/input/datasets/bharatmohan/nemotron-v34c-sft-adapter",
    "/kaggle/input/*/nemotron-v34c-sft-adapter",
    "/kaggle/input/*",
    "/kaggle/input/*/*",
]:
    for d in glob.glob(pat):
        if os.path.isdir(d):
            cfg = os.path.join(d, "adapter_config.json")
            sft = os.path.join(d, "adapter_model.safetensors")
            if os.path.exists(cfg) and os.path.exists(sft):
                candidates.append(d)
                print(f"  Found candidate: {d}")

if not candidates:
    # Walk all of /kaggle/input
    print("No direct match — walking /kaggle/input/")
    for root, dirs, files in os.walk("/kaggle/input"):
        if "adapter_model.safetensors" in files and "adapter_config.json" in files:
            candidates.append(root)
            print(f"  Found via walk: {root}")

if not candidates:
    print("ERROR: no adapter files found in /kaggle/input")
    raise SystemExit(1)

src_dir = candidates[0]
print(f"\nUsing adapter from: {src_dir}")

# Copy adapter files
for fname in ["adapter_config.json", "adapter_model.safetensors"]:
    src = os.path.join(src_dir, fname)
    dst = os.path.join(ADAPTER_DIR, fname)
    shutil.copy(src, dst)
    print(f"  Copied {fname}: {os.path.getsize(dst)/1024/1024:.1f} MB")

# Patch adapter_config.json for inference
config_path = os.path.join(ADAPTER_DIR, "adapter_config.json")
with open(config_path) as f:
    cfg = json.load(f)
cfg["inference_mode"] = True
cfg["lora_dropout"] = 0.0
with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"  Patched adapter_config.json (inference_mode=True)")

# Build submission.zip
zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in ["adapter_config.json", "adapter_model.safetensors"]:
        fpath = os.path.join(ADAPTER_DIR, fname)
        if os.path.exists(fpath):
            zf.write(fpath, fname)
            print(f"  Added to zip: {fname}")

print(f"\nsubmission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
print("DONE")
