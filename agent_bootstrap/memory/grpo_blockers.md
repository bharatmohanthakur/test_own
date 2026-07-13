---
name: GRPO blockers on Nemotron-3-Nano
description: After 6 attempts on RunPod Apr 12, GRPO with Komil's transformers 5.x fix is BLOCKED by Mamba 4-bit incompatibility. Only proven path is Unsloth + TRL 0.24 (slow, 12 min/step).
type: feedback
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
## GRPO on RunPod — blocked (Apr 12, 2026)

### What was tried
1. transformers 5.5.3 + TRL 1.1.0 + native NemotronH (Komil's speed fix)
2. Clean venv (no mayukh18 wheel conflicts)
3. bf16 full model → OOM at 94.05GB/94.98GB during generation
4. 4-bit quantization (bitsandbytes nf4) → `RuntimeError: mat1 and mat2 shapes cannot be multiplied` in Mamba SSM ops — bitsandbytes can't handle Mamba's custom CUDA kernels
5. Multiple TRL API fixes (max_new_tokens→max_completion_length, tokenizer→processing_class, config→args)

### Root cause
Mamba's `mamba_split_conv1d_scan_combined` kernel calls `F.linear(out, outproj_weight)` where `outproj_weight` is quantized to (1, 5505024) instead of normal shape. bitsandbytes intercepts F.linear but can't handle the Mamba output projection.

### What DOES work (but slow)
Unsloth + TRL 0.24 + mayukh18 wheels + v28 monkey-patch + is_fast_path=False:
- 12 min/step (2 tok/s generation)
- v31 ran 10 GRPO steps in 121 min
- Stable VRAM at ~94.5GB (no OOM)
- Proven for v31/v32 adapters

### What would fix it
- Unsloth adds native transformers 5.3+ support (fixes the speed AND compatibility)
- OR: mamba_ssm releases a version compatible with bitsandbytes 4-bit
- OR: use vLLM for generation (separate process) — but TRL's use_vllm=False is required for Nemotron

### RunPod spending
- Started: $11.51
- After pod (30 min): $10.61
- Remaining: enough for ~6 hrs training if needed later

**Why:** Komil's 20x speedup is real but creates Mamba/quantization incompatibility. Until Unsloth supports transformers 5.3+, GRPO must use the slow path.
**How to apply:** Don't attempt Komil's fix for GRPO. Use Unsloth + TRL 0.24 pipeline if GRPO is needed. Budget 12 min/step for planning.
