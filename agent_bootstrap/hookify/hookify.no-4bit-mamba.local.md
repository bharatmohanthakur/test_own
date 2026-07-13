---
name: no-4bit-mamba
enabled: true
event: bash
conditions:
  - field: command
    operator: regex_match
    pattern: load_in_4bit\s*=\s*True|--load-in-4bit(?!\s+False)|bnb_4bit|BitsAndBytesConfig|--4bit
  - field: command
    operator: not_contains
    pattern: nf4-acknowledged
action: block
---

🛑 **4-bit quantization with Mamba SSM = guaranteed failure.**

Per `memory/grpo_blockers.md`:
> "4-bit quantization (bitsandbytes nf4) → RuntimeError: mat1 and mat2 shapes cannot be multiplied — bitsandbytes can't handle Mamba's custom CUDA kernels."

This was burned **6 times** on RunPod Apr 12. Nemotron-3-Nano is Mamba-Transformer hybrid; 4-bit will not work.

**Use instead:**
- `load_in_4bit=False, dtype=torch.bfloat16`
- Unsloth FastLanguageModel works in **bf16** mode (no 4-bit) — still gets xformers/triton speedup
- For training script flag: `--no-4bit`

If you have a NEW reason to believe 4-bit works (mamba_ssm patched, etc.), append `# nf4-acknowledged` to bypass.
