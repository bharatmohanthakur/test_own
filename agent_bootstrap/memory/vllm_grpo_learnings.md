---
name: vLLM GRPO learnings
description: Complete vLLM + GRPO + Nemotron compatibility findings — what works, what doesn't, exact errors and fixes. Critical for B200 setup.
type: feedback
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
# vLLM + GRPO + Nemotron-3-Nano-30B (Apr 12, 2026)

## What we tried and what happened

### 1. vLLM colocate + PeftModel (FAILS)
- **Error**: `KeyError: 'layers.1.mixer.experts.up_proj'` in vllm/model_executor/models/nemotron_h.py
- **Why**: vLLM colocate syncs weights from training model. PeftModel has `base_model.model.model.layers.X` names, but vLLM expects vanilla `layers.X` names.
- **Fix**: `merge_and_unload()` the adapter into base model BEFORE GRPOTrainer, then pass fresh `peft_config` to GRPOTrainer.
- **GitHub**: huggingface/trl#2698 (closed/fixed)

### 2. vLLM colocate + merged model + same GPU (FAILS without sleep mode)
- **Error**: `ValueError: Free memory on device cuda:0 (19.55/139.81 GiB) on startup is less than desired GPU memory utilization`
- **Why**: Training model (~62GB) already on GPU. vLLM tries to claim its share ON TOP of that. 62GB model + 0.5*140=70GB vLLM = 132GB > 140GB available.
- **Fix**: Enable `vllm_enable_sleep_mode=True` — training model offloads during generation.

### 3. vLLM colocate + sleep mode + expandable_segments (FAILS)
- **Error**: `AssertionError: Expandable segments are not compatible with memory pool`
- **Why**: vLLM sleep mode uses PyTorch memory pool. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` conflicts.
- **Fix**: Remove `expandable_segments:True` env var when using vLLM sleep mode.

### 4. vLLM colocate + sleep mode + correct memory (TESTED on B200 — FAILS)
- Even with all fixes (merge_and_unload, sleep mode, correct mem, enforce_eager), the fundamental issue remains:
  GRPOTrainer re-wraps the merged model with fresh `peft_config` → PeftModel → weight sync fails
- `KeyError: 'layers.1.mixer.experts.up_proj'` during training_step weight sync
- **This is UNFIXABLE without TRL code changes** (they need to handle PeftModel→vLLM name mapping for NemotronH)

### 8. B200 Blackwell flashinfer JIT failure
- **Error**: `subprocess.CalledProcessError` during `ninja build` of flashinfer fmha_gen for sm_100a
- **Fix**: Patch `enforce_eager=True` into TRL's `vllm_generation.py` line 342 (in the `LLM()` call)
- This skips CUDA graph capture which triggers the JIT
- B200 actual VRAM: 178.35 GiB usable (advertised 183GB)
- gpu_memory_utilization=0.65 fits (0.65 × 178 = 116GB, with 118GB free after 60GB model)

### 5. vLLM + transformers version conflict
- vLLM 0.17.1 pins `transformers==4.57.6` via dependencies
- Komil fix needs `transformers>=5.5.3` for native NemotronH
- **Fix**: Install vLLM first, then `uv pip install "transformers>=5.5" --no-deps` to override without pulling vLLM deps
- Both import fine after override

### 6. vLLM + NemotronH model registry
- vLLM 0.17.1 DOES have `NemotronHForCausalLM` in its native registry (found via grep)
- BUT `vllm_model_impl="transformers"` backend explicitly REJECTS NemotronH: `'NemotronHForCausalLM' is not compatible with vLLM`
- **Fix**: Do NOT set `vllm_model_impl`. Let vLLM use its native implementation (default).
- Also must strip `auto_map` from model config.json and remove custom `.py` files

### 7. NemotronH `_no_split_modules` has nested set (accelerate bug)
- **Error**: `TypeError: unhashable type: 'set'` in `accelerate/utils/modeling.py:1019`
- **Why**: NemotronH config has a `set` inside `_no_split_modules` list. `set(no_split_module_classes)` fails.
- **Fix**: Flatten before loading adapter:
```python
if hasattr(model, '_no_split_modules') and model._no_split_modules:
    flat = []
    for item in model._no_split_modules:
        if isinstance(item, (set, frozenset)):
            flat.extend(list(item))
        else:
            flat.append(item)
    model._no_split_modules = flat
```

## Without vLLM — native generation speed
- Step 1: 270s (CUDA compile overhead)
- Step 2: 234s
- Step 3: 219s
- Step 4: 188s
- Step 5: 153s
- Step 6: 104s
- Step 7: 123s
- Settling ~100-150s/step after warmup with max_completion=4096, gen=2
- VRAM: 99.5GB on H200 (no OOM)

## B200 result (Apr 13, 2026) — DOES NOT WORK
- B200 192GB VRAM fits model (60GB) + vLLM (65% × 178 = 116GB) = 176GB ✓
- BUT vLLM colocate weight sync ALWAYS fails with PeftModel:
  `KeyError: 'layers.1.mixer.experts.up_proj'`
  Even after merge_and_unload, GRPOTrainer re-wraps with peft_config → PeftModel again → weight names mismatch
- flashinfer JIT also fails on Blackwell sm_100a → need `enforce_eager=True` patch in TRL
- **CONCLUSION: vLLM colocate is INCOMPATIBLE with LoRA/PEFT on NemotronH. Period.**
  This is a TRL limitation, not a VRAM issue. Weight sync between PeftModel and vLLM uses different parameter name schemes.
- **Workaround**: Use vLLM **server mode** with 2+ GPUs (separate GPU for vLLM inference, separate for training). Requires multi-GPU instance.
- **Alternative**: Just use native generation (no vLLM). H200 achieved ~2 min/step after warmup. That's the practical path.
- **KEY INSIGHT**: B200 is Blackwell (sm_120) = SAME as Kaggle RTX Pro 6000 Blackwell. Just use v2 GRPO script (no vLLM) with Kaggle Blackwell fixes (is_fast_path_available=False, Triton rmsnorm, ptxas). No need to experiment with vLLM — proven approach works directly. STOP overengineering, use what's proven.

## GRPOConfig params for vLLM (TRL 1.1.0)
```python
use_vllm=True,                    # Enable vLLM
# vllm_mode="colocate",           # Default, don't set explicitly
vllm_enable_sleep_mode=True,      # Training/gen take turns on GPU
vllm_gpu_memory_utilization=0.7,  # For B200 179GB
vllm_max_model_length=4096,       # Match max_completion_length
# Do NOT set vllm_model_impl — let vLLM use native backend
```

## Vast.ai H200 setup steps (validated)
1. `uv venv --python 3.12` (mayukh18 wheels are cp312)
2. `uv pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128`
3. `kaggle datasets download mayukh18/nemotron-packages` + install wheels
4. `uv pip install "transformers>=5.5" "trl>=1.1.0" "peft>=0.18.1" --no-deps` (override mayukh18's old versions)
5. `uv pip install "vllm>=0.11,<=0.17.1"` then override transformers again with `--no-deps`
6. Strip `auto_map` from model config.json, remove `modeling_*.py` and `configuration_*.py`
7. Download model, adapter, train.csv via Kaggle API on instance

**Why:** Wasted ~2 hours of H200 time ($7) debugging vLLM compatibility. These learnings will save time on B200.
**How to apply:** Follow this exact sequence when setting up B200 tomorrow. Test vLLM sleep mode first before long training run.
