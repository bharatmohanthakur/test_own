---
name: GRPO working recipe — Komil fix + full bf16 + LoRA
description: PROVEN GRPO recipe on RunPod RTX Pro 6000 Blackwell. 27s/step (26x faster than v31's 12 min/step). Full bf16 + native NemotronH + LoRA + no grad ckpt.
type: reference
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
# Working GRPO Recipe (Apr 12, 2026)

## Key ingredients
1. **Clean venv** (NOT mayukh18 wheels): `pip install transformers>=5.3 trl peft datasets accelerate`
2. **Native NemotronH**: `AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")` — NO trust_remote_code
3. **LoRA on top**: `get_peft_model(model, LoraConfig(r=32, alpha=16, ...))` — required for training, also keeps optimizer small
4. **gradient_checkpointing=False**: native NemotronH doesn't support it, but LoRA keeps optimizer states to ~0.2GB (27M params only)
5. **TRL 1.1.0 API**: `max_completion_length` not `max_new_tokens`, `processing_class` not `tokenizer`, `args` not `config`, `beta` not `kl_coeff`
6. **TRANSFORMERS_CACHE fix** + mock `llm_blender`/`mergekit` for import chain

## Performance
- **27s/step** (vs v31's ~720s/step = 26x faster!)
- VRAM: 80.2 GB / 97.9 GB (17 GB headroom)
- Model: 63 GB, LoRA + optimizer: ~1 GB, generation cache: ~16 GB
- 50 steps = ~22 min. 200 steps = ~90 min.

## Config that works
```python
GRPOConfig(
    max_steps=50, per_device_train_batch_size=1, gradient_accumulation_steps=4,
    learning_rate=5e-6, num_generations=2, max_completion_length=512,
    temperature=0.9, beta=0.02, use_vllm=False, bf16=True,
    gradient_checkpointing=False, remove_unused_columns=False,
)
```

## What DOESN'T work
- gradient_checkpointing=True → ValueError (native NemotronH doesn't declare support)
- 4-bit quantization → Mamba SSM shape mismatch in F.linear
- Unsloth + transformers 5.x → version conflict
- TRL 0.24 + transformers 5.x → import errors

## Exact pip install sequence (clean venv, RunPod or Kaggle)
```bash
python3 -m venv /workspace/grpo_env
source /workspace/grpo_env/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=5.3" trl peft datasets accelerate
# Do NOT install mamba_ssm/causal_conv1d — native fallback is fine and avoids conflicts
# Do NOT install Unsloth — incompatible with transformers 5.x
```

## Import fixes (add BEFORE importing trl)
```python
import transformers.utils.hub as _hub
if not hasattr(_hub, 'TRANSFORMERS_CACHE'):
    from huggingface_hub import constants as _hf_c
    _hub.TRANSFORMERS_CACHE = getattr(_hf_c, 'HF_HUB_CACHE', '/tmp/hf_cache')
for _mod in ['llm_blender', 'mergekit', 'mergekit.config']:
    if _mod not in sys.modules:
        _fake = type(sys)(_mod)
        _fake.__spec__ = type(sys)(_mod)
        _fake.__path__ = []
        if '.' not in _mod: _fake.MergeConfiguration = None
        sys.modules[_mod] = _fake
```

## Model load (NO trust_remote_code)
```python
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
# NOT trust_remote_code=True — that pulls old broken modeling_nemotron_h.py
```

## LoRA (REQUIRED — can't train full model, and optimizer stays small)
```python
lora_config = LoraConfig(r=32, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q_proj","k_proj","v_proj","o_proj","in_proj","out_proj",
                     "up_proj","down_proj","gate_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
```

## Reward functions
```python
def correctness_reward(prompts, completions, answer, **kwargs):
    # Returns 2.0 for correct, 0.0 for wrong
def format_reward(completions, **kwargs):
    # Returns 0.5 if \boxed{} present, 0.0 if not
```

## For Kaggle: use mayukh18 offline wheels dataset for packages
On Kaggle RTX Pro 6000, internet is blocked in batch mode for this competition.
Upload transformers 5.3+ wheel as a dataset, install from offline:
```bash
pip install --no-index --find-links /kaggle/input/your-wheel-dataset transformers trl peft
```

**Why:** Komil's fix (native NemotronH in transformers 5.3+) fixes the cache param bug that caused 2 tok/s. Combined with LoRA (small optimizer footprint), fits in 96GB without grad ckpt.
**How to apply:** Use this recipe for any future GRPO runs. Budget 27s/step for planning. Script at `runpod_v40/train_grpo_v2.py`.
