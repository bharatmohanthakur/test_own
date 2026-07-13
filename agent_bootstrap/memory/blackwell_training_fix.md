---
name: blackwell_training_fix
description: Complete fix for training Nemotron-3-Nano-30B on RTX Pro 6000 Blackwell GPU - solved after 20+ failed attempts
type: feedback
---

## The 5 fixes needed to train on RTX Pro 6000 Blackwell:

1. **Hide causal_conv1d via find_spec patch** (BEFORE any model imports)
```python
_orig_find_spec = importlib.util.find_spec
def _patched_find_spec(name, *args, **kwargs):
    if 'causal_conv1d' in str(name): return None
    return _orig_find_spec(name, *args, **kwargs)
importlib.util.find_spec = _patched_find_spec
```

2. **Triton ptxas-blackwell permission fix** (use UNDERSCORE in path)
```python
src = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/triton/backends/nvidia/bin/ptxas-blackwell"
# NOT nvidia-utility-script (hyphen) — must be nvidia_utility_script (underscore)
```

3. **Patch is_fast_path_available = False AFTER model load**
```python
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        mod.is_fast_path_available = False
```

4. **Replace Triton rmsnorm_fn with pure PyTorch**
```python
def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5, ...):
    # pure PyTorch implementation
```

5. **gradient_checkpointing with use_reentrant=False**
```python
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```

## Also:
- NO offload_folder (model fits in 48GB bf16)
- NO internet on RTX Pro 6000 — wandb offline only
- NO pip install — use only pre-installed packages
- Path uses UNDERSCORE: nvidia_utility_script (not hyphen)
- Reference: kaggle.com/code/johnnyhyland/nvidia-nemotron-tuning-basic-approach

**Why:** 20+ failed runs learning these. Cost ~3hrs GPU quota.
**How to apply:** Use sft_v3/train.py as the template for ALL future training runs.
