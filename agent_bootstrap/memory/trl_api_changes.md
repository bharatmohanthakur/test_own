---
name: TRL API version differences
description: Critical API changes between TRL 0.24 (mayukh18 wheels) and TRL 1.1.0 (latest pip) that cause silent crashes.
type: feedback
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
# TRL API Changes (0.24 → 1.1.0)

When using `pip install trl` (gets 1.1.0) instead of mayukh18 wheels (pins 0.24.0):

| TRL 0.24 | TRL 1.1.0 | Notes |
|---|---|---|
| `GRPOConfig(max_new_tokens=...)` | `GRPOConfig(max_completion_length=...)` | Renamed |
| `GRPOConfig(kl_coeff=...)` | `GRPOConfig(beta=...)` | Renamed |
| `GRPOTrainer(tokenizer=...)` | `GRPOTrainer(processing_class=...)` | Renamed |
| `GRPOTrainer(config=...)` | `GRPOTrainer(args=...)` | Renamed |
| `from trl import GRPOTrainer` works | Needs `mergekit`, `llm_blender` installed | New deps |

## Environment setup for GRPO with Komil's speed fix

Must use **clean venv** — mixing mayukh18 wheels with pip transformers 5.x breaks everything.

```bash
python3 -m venv /workspace/grpo_env
source /workspace/grpo_env/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=5.3" trl peft datasets accelerate
# Install mamba/causal_conv1d wheels separately
```

**Why:** Spent 4 fix-relaunch cycles on API mismatches. TRL 1.1.0 is a major rewrite.
**How to apply:** Always check TRL version. Use `python3 -c "import trl; print(trl.__version__)"` before writing GRPO code.

## TRANSFORMERS_CACHE fix for transformers 5.x
```python
import transformers.utils.hub as _hub
if not hasattr(_hub, 'TRANSFORMERS_CACHE'):
    from huggingface_hub import constants as _hf_c
    _hub.TRANSFORMERS_CACHE = getattr(_hf_c, 'HF_HUB_CACHE', '/tmp/hf_cache')
```

## RunPod file transfer rule
**NEVER pass Kaggle credentials via SSH heredoc** — user prefers all downloads/uploads done directly on RunPod. SCP local files instead.
