# Reference: Working Training Notebook by johnnyhyland
# https://www.kaggle.com/code/johnnyhyland/nvidia-nemotron-tuning-basic-approach

## Key Fixes for RTX Pro 6000 Blackwell:

### 1. Patch is_fast_path_available AFTER model load
```python
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        mod.is_fast_path_available = False
```

### 2. Replace Triton rmsnorm_fn with pure PyTorch
```python
def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5, ...):
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()
    if bias is not None: out = out + bias.float()
    if z is not None: out = out * F.silu(z.float())
    return out.to(dtype)

for name, mod in list(sys.modules.items()):
    if hasattr(mod, 'rmsnorm_fn'):
        mod.rmsnorm_fn = _pure_rmsnorm_fn
```

### 3. Fix Triton ptxas permission
```python
shutil.copy2(src_ptxas_blackwell, "/tmp/ptxas-blackwell")
os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC)
os.environ["TRITON_PTXAS_PATH"] = dst
```

### 4. Gradient checkpointing with use_reentrant=False
```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

### 5. No offload_folder needed (model fits in 48GB with bf16)

### 6. lora_alpha=8 (not 2x rank)
