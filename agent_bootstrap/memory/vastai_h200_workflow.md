---
name: Vast.ai H200 GRPO workflow
description: Complete Vast.ai H200 setup for GRPO training — uv Python 3.12 venv, mayukh18 wheels, Komil override, step-by-step validated Apr 12 2026
type: reference
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
# Vast.ai H200 Setup (validated Apr 12, 2026)

## Instance details
- Contract: 34754786, SSH: `ssh -p 34786 root@ssh6.vast.ai`
- GPU: NVIDIA H200 143771 MiB, $3.50/hr
- Image: pytorch (default Vast.ai image, Python 3.11 system)

## Setup steps (in order — each depends on previous)

### Step 1: Install uv + Create Python 3.12 venv
Vast.ai image has Python 3.11, but mayukh18 wheels are cp312.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv /workspace/venv --python 3.12
source /workspace/venv/bin/activate
```

### Step 2: Install PyTorch 2.10+ (cu128)
TRL 1.1.0 requires `FSDPModule` from torch >= 2.10. cu124 only has up to 2.6.
```bash
uv pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
```

### Step 3: Install mayukh18 wheels FIRST
These bring mamba_ssm, causal_conv1d, unsloth, and OLD versions of transformers/trl/peft.
```bash
kaggle datasets download mayukh18/nemotron-packages -p /workspace/nemotron-pkgs --unzip
PKG=/workspace/nemotron-pkgs
uv pip install --no-index --find-links $PKG/packages unsloth trl peft transformers datasets accelerate bitsandbytes
uv pip install $PKG/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
uv pip install $PKG/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

### Step 4: Override with Komil-compatible versions LAST
mayukh18 installs old transformers 4.57 / trl 0.24 — override to Komil fix versions:
```bash
uv pip install "transformers>=5.5" "trl>=1.1.0" "peft>=0.18.1" "huggingface_hub>=1.5"
```

### Step 5: Download model + data directly on instance
```bash
# Kaggle credentials
mkdir -p ~/.kaggle
echo '{"username":"bharatmohan","key":"<REDACTED-set-via-env>"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Model from HuggingFace
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', local_dir='/workspace/model')"

# train.csv
kaggle competitions download nvidia-nemotron-model-reasoning-challenge -f train.csv -p /workspace/data/

# v38a adapter (URL trick — kaggle CLI has 0-byte bug for kernel outputs)
# Use Python requests with /api/v1/kernels/output endpoint
```

## Verified versions
- torch 2.10.0+cu128
- transformers 5.5.3
- trl 1.1.0
- peft 0.18.1
- mamba_ssm 2.3.1
- causal_conv1d 1.6.1
- Python 3.12.13

## Performance
- Model load: ~15 sec
- VRAM after model + adapter: 63.3 GB
- VRAM during training: ~99.5 GB (of 143 GB)
- Step time: ~3.5 min/step (4096 max_completion, 2 generations)
- First step slower (4:30) due to CUDA compilation

## Key differences from RunPod
- RunPod used Python 3.12 system → no uv needed
- RunPod used RTX Pro 6000 Blackwell → needed is_fast_path_available patch
- H200 is Hopper → no Blackwell patches needed
- H200 has 143GB vs 94GB → 4096 max_completion fits without OOM

## Cost
- $3.50/hr for H200
- 200 steps × 3.5 min = ~12 hrs = ~$42
- 50 steps × 3.5 min = ~3 hrs = ~$10.50
