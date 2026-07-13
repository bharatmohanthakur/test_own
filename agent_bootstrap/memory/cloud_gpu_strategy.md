---
name: Cloud GPU strategy
description: Which GPU to rent, what approach to use. B200 Blackwell = same as Kaggle RTX Pro 6000. Don't waste time on vLLM — use proven native approach.
type: feedback
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Rule: Use what's proven. Stop overengineering.**

## Exact B200 recipe (replicate Kaggle v43 GRPO that was working)
1. `uv venv --python 3.12` + `torch==2.10.0+cu128`
2. `kaggle datasets download mayukh18/nemotron-packages` → install wheels (peft old + mamba_ssm + causal_conv1d)
3. Override: `transformers>=5.5.3`, `trl>=1.1.0`, `peft>=0.18.1`, `huggingface_hub>=1.5` (all --no-deps)
4. Strip auto_map from model config, remove modeling_*.py / configuration_*.py
5. Load v38a adapter as PeftModel (is_trainable=True) — NOT merge_and_unload
6. NO vLLM — just native generation (Komil 27s/step speed)
7. gen=4, max_completion=4096 — B200 192GB has massive headroom, no compromise needed
8. **Kaggle wheel dataset**: `bharatmohan/grpo-wheels-tf53-trl11` has transformers+trl+peft+hf_hub wheels

## GPU Architecture Map
- **Kaggle RTX Pro 6000** = Blackwell (sm_120), 94GB — FREE
- **Vast.ai B200** = Blackwell (sm_120), 179-192GB — $3.44/hr
- **Vast.ai H200** = Hopper (sm_90), 143GB — $3.50/hr
- **RunPod RTX Pro 6000 Blackwell** = Blackwell (sm_120), 94GB — $1.69/hr

## B200 = Kaggle Blackwell. Use Kaggle fixes directly.
B200 is Blackwell sm_120 — exact same arch as Kaggle RTX Pro 6000 Blackwell. ALL Kaggle fixes apply:
- `is_fast_path_available = False` patch
- Triton rmsnorm replacement
- ptxas-blackwell copy
- Same mayukh18 wheels

Just run the proven GRPO v2 script (no vLLM). Expected ~2 min/step with 192GB (zero OOM risk).

## vLLM is NOT viable for NemotronH + LoRA (as of Apr 2026)
- vLLM colocate + PeftModel = KeyError on weight sync (parameter name mismatch)
- merge_and_unload doesn't help because GRPOTrainer re-wraps with peft_config
- B200 flashinfer JIT fails on sm_100a (needs enforce_eager=True patch)
- H200 doesn't have enough VRAM for both model copies without sleep mode
- **Don't attempt vLLM again unless TRL explicitly adds NemotronH+PEFT support**

## Cost comparison for GRPO (200 steps, ~2 min/step = ~7 hrs)
- Kaggle RTX Pro 6000: FREE (30 hrs/week quota)
- Vast.ai B200: $3.44/hr × 7 = ~$24
- Vast.ai H200: $3.50/hr × 7 = ~$24.50
- RunPod RTX Pro 6000: $1.69/hr × 7 = ~$12

**Why:** Wasted ~$10+ and 3 hours of H200+B200 time trying vLLM. Should have just run native GRPO on the first H200 attempt. The H200 was already working at 104s/step when killed.
**How to apply:** Next cloud GPU run: rent B200 or H200, immediately run proven v2 script (no vLLM), monitor steps. Don't experiment.
