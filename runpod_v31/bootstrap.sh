#!/bin/bash
# Bootstrap script for RunPod v31 GRPO training
# Run this on the pod after SCP'ing the runpod_v31/ directory to /workspace/
set -e

cd /workspace
echo "=== Bootstrap starting at $(date) ==="

# 1. Install Python deps
echo "=== Installing Python packages ==="
pip install -q --upgrade pip
pip install -q "huggingface_hub[hf_transfer]"
pip install -q unsloth trl peft transformers datasets accelerate bitsandbytes
pip install -q causal_conv1d mamba_ssm 2>&1 || echo "  (mamba_ssm/causal_conv1d may already be present)"

# 2. Download Nemotron-3-Nano-30B from HuggingFace
echo "=== Downloading Nemotron-3-Nano-30B-A3B-BF16 (~60GB) ==="
mkdir -p /workspace/model
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --local-dir /workspace/model \
    --local-dir-use-symlinks False

ls -lh /workspace/model | head -20

# 3. Verify everything is in place
echo "=== Verifying inputs ==="
test -f /workspace/adapter_v22/adapter_model.safetensors && \
    echo "  v22 adapter: $(du -h /workspace/adapter_v22/adapter_model.safetensors | cut -f1)" || \
    echo "  ERROR: v22 adapter missing!"
test -f /workspace/data/grpo_goldilocks.jsonl && \
    echo "  Goldilocks data: $(wc -l < /workspace/data/grpo_goldilocks.jsonl) lines" || \
    echo "  ERROR: Goldilocks data missing!"
test -f /workspace/train_runpod.py && echo "  train_runpod.py: present" || echo "  ERROR: train_runpod.py missing!"

# 4. Run training
echo ""
echo "=== Starting GRPO training at $(date) ==="
cd /workspace
python3 train_runpod.py 2>&1 | tee /workspace/train.log

echo ""
echo "=== Done at $(date) ==="
ls -lh /workspace/output/
