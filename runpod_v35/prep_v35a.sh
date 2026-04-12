#!/bin/bash
# Prep v35a (narrow GRPO) on the SAME pod as v34. Assumes setup.sh ran already and
# v34 SFT is complete (output_v34/nemotron-lora-adapter exists).
set -e
echo "=== v35a prep at $(date) ==="

# 1. Download goldilocks GRPO dataset (179 prompts, used by v31)
if [ ! -f /workspace/data/grpo_goldilocks.jsonl ]; then
    echo "--- downloading goldilocks data ---"
    kaggle datasets download bharatmohan/nemotron-training-v31 -p /workspace/data --unzip 2>&1 | tail -3
fi
ls -lh /workspace/data/grpo_goldilocks.jsonl

# 2. Verify v34 adapter exists
ls -lh /workspace/output_v34/nemotron-lora-adapter/

echo "=== prep done at $(date) ==="
