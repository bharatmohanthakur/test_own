#!/bin/bash
set -e
echo "=== v34 setup at $(date) ==="

# 1. Torch upgrade
echo "--- torch upgrade ---"
pip install -q --upgrade --break-system-packages "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3

# 2. Kaggle credentials (private dataset access)
mkdir -p /root/.kaggle
cat > /root/.kaggle/kaggle.json << 'KAGEOF'
{"username":"bharatmohan","key":"1f8a45c8e928ba8ef9929855d233a32f"}
KAGEOF
chmod 600 /root/.kaggle/kaggle.json

# 3. Install kaggle CLI + wandb
pip install -q --break-system-packages kaggle wandb 2>&1 | tail -3

# 4. Download mayukh18 wheels (the proven Nemotron toolchain)
echo "--- mayukh18 wheels ---"
kaggle datasets download mayukh18/nemotron-packages -p /workspace/nemotron-pkgs --unzip 2>&1 | tail -3

# 5. Install deps from wheels
echo "--- install deps from wheels ---"
PKG=/workspace/nemotron-pkgs
pip install -q --break-system-packages --no-index --find-links $PKG/packages \
    unsloth trl peft transformers datasets accelerate bitsandbytes 2>&1 | tail -3
pip install -q --break-system-packages $PKG/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl 2>&1 | tail -3
pip install -q --break-system-packages $PKG/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl 2>&1 | tail -3

# 6. vLLM for inference (eval phase)
echo "--- vllm ---"
pip install -q --break-system-packages vllm 2>&1 | tail -5

# 7. Verify imports
echo "--- verify imports ---"
python3 << 'PYEOF'
import torch, mamba_ssm, causal_conv1d, unsloth, trl
print(f"torch {torch.__version__} cuda {torch.version.cuda}")
print(f"unsloth {unsloth.__version__}, trl {trl.__version__}")
try:
    import vllm
    print(f"vllm {vllm.__version__} OK")
except Exception as e:
    print(f"vllm FAILED: {e}")
PYEOF

# 8. Download v34 training data
echo "--- v34 training data ---"
mkdir -p /workspace/data
kaggle datasets download bharatmohan/nemotron-training-v34 -p /workspace/data --unzip 2>&1 | tail -3
ls -lh /workspace/data/

# 9. Download train.csv (for eval) — competition data
echo "--- train.csv for eval ---"
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -f train.csv -p /workspace/data 2>&1 | tail -3
ls -lh /workspace/data/train.csv 2>/dev/null

# 10. Download v31 adapter (for side-by-side eval comparison)
echo "--- v31 adapter for eval ---"
mkdir -p /workspace/adapter_v31
kaggle datasets download bharatmohan/nemotron-v31-grpo-adapter -p /workspace/adapter_v31 --unzip 2>&1 | tail -3
ls -lh /workspace/adapter_v31/

# 11. Download Nemotron base model
echo "--- model download ---"
mkdir -p /workspace/model
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 << 'PYEOF'
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    local_dir="/workspace/model",
    max_workers=8,
)
print("model download done")
PYEOF

echo "=== setup done at $(date) ==="
du -sh /workspace/model /workspace/data /workspace/nemotron-pkgs /workspace/adapter_v31 2>/dev/null
