"""
Submit pre-trained LoRA adapter for Nemotron reasoning challenge.
Adapter trained on Vast.ai H100 NVL: 3 epochs, 2000 quality CoT examples.
"""
import glob, os, shutil, subprocess, zipfile

# Find adapter files from dataset input
adapter_paths = glob.glob("/kaggle/input/nemotron-sft-v5-adapter/adapter_*")
print(f"Found adapter files: {adapter_paths}")

OUTPUT_DIR = "/kaggle/working"

# Copy adapter files to working directory
for p in adapter_paths:
    fname = os.path.basename(p)
    shutil.copy2(p, os.path.join(OUTPUT_DIR, fname))
    print(f"Copied {fname} ({os.path.getsize(p)/1024/1024:.1f} MB)")

# Create submission.zip
zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
    for fname in ["adapter_config.json", "adapter_model.safetensors"]:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            zf.write(fpath, fname)
            print(f"Added {fname} to zip")

print(f"\nsubmission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
print("DONE - ready to submit!")
