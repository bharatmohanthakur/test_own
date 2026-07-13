---
name: THK adapter analysis — exact config + weaknesses
description: THK's v26 adapter exact training config, submission transformation, and per-type weaknesses — the blueprint for refinement.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Analyzed:** 2026-04-16
**Source:** `samvalladares/huikang-nemotron-artifacts` + `huikang/tinker-submission-notebook`

## Public artifacts
- **Weights:** `samvalladares/huikang-nemotron-artifacts/adapter_v26_model.safetensors` (1.54 GB)
- **Config:** `adapter_v26_config.json` in same dataset
- **Model hub:** `huikang/nemotron-adapter/transformers/default/26` (inference-ready format)
- **Submission NB:** `huikang/tinker-submission-notebook` (shows transformation logic)
- **Ref adapter NB:** `huikang/nvidia-nemotron-all-linear` (ref submission config)
- **Corpus:** `corpus.jsonl` (42.8 MB, full training data with traces)
- **Bit-manip extras:** `bit_manipulation_3input_traces.jsonl` (203 KB, new 3-input traces)

## THK's ACTUAL training config (adapter_v26_config.json)
```json
{
  "r": 32,
  "lora_alpha": 32,
  "lora_dropout": 0,
  "target_modules": "all-linear",    // every nn.Linear in the model
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

## THK's submission transformation (from tinker-submission-notebook)
The raw Tinker adapter has `target_modules: "all-linear"` but the SUBMITTED version:

1. **Rewrites target_modules to exactly 9 names:**
   ```python
   ["k_proj", "o_proj", "in_proj", "q_proj", "up_proj", "v_proj",
    "down_proj", "out_proj", "lm_head"]
   ```
2. **Renames tensor keys:** `base_model.model.model.*` → `base_model.model.backbone.*`
3. **Merges Mamba adapters:** `gate_proj + x_proj` LoRA → `in_proj` LoRA (Nemotron-H specific)
4. **Drops empty w3 expert LoRAs** (MoE experts that never fire)

## WHY our previous refinement scored 0.66 (THK adapter + Unsloth)
- We assumed Kh0a config (alpha=16) — actual is alpha=32
- We assumed 9 modules without lm_head — actual includes lm_head
- We assumed dropout=0.1 — actual is 0
- Unsloth's `train_on_responses_only` + packing masked tokens differently than THK's training

## WHY our B200 SFT scored 0.26
- Kh0a config on THK's data is a total mismatch (see above)
- max_seq=4096 truncated THK's 6000-8000 token traces
- LR=1e-4 vs THK's 2e-4
- No wandb → ran blind through divergence

## THK's self-reported per-type success (problems.jsonl)

| Category | Total | Solved | % | Unsolved lever |
|---|---|---|---|---|
| cipher | 1576 | 1576 | 100% | 0 |
| gravity | 1597 | 1597 | 100% | 0 |
| numeral | 1576 | 1576 | 100% | 0 |
| unit_conversion | 1594 | 1594 | 100% | 0 |
| equation_numeric_deduce | 596 | 540 | 90.6% | 56 |
| bit_manipulation | 1602 | 1364 | 85.1% | **238** |
| equation_numeric_guess | 136 | 21 | 15.4% | **115** |
| cryptarithm_deduce | 659 | 54 | 8.2% | **605** |
| cryptarithm_guess | 164 | 11 | 6.7% | **153** |

**Total unsolved by THK: 1,167 problems = 12.3%** — this is the ceiling gap to 1.0.

## Refinement strategy (path 0.85 → 0.90+)

### Tier 1 opportunity: cryptarithm (758 problems, +8%)
- THK solved only 8% because cryptarithm is CSP-hard with 12+ unique symbols
- **Attack:** proper CSP solver (constraint propagation, not brute force)
- Build new CoT traces for these 758 then fine-tune

### Tier 2 opportunity: bit_manipulation (238 problems, +2.5%)
- THK has `bit_manipulation_3input_traces.jsonl` in his artifacts — new 3-input traces he produced AFTER v26
- These are the exact problems v26 missed
- **Attack:** quick SFT top-up on just these 238 examples with EXACT THK config

### Tier 3 opportunity: equation_numeric_guess (115 problems, +1.2%)
- Weird non-standard operations (rev, multi-op)
- **Attack:** more search combos in guess-type solver

## THK's OWN fix plan (visible from his artifacts)
He already uploaded `bit_manipulation_3input_traces.jsonl` → means he plans another SFT round on these. We should do the same FIRST since data is already public.

## Mandatory refinement config (DO NOT DEVIATE)
```python
LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    target_modules="all-linear",   # critical: NOT a named list
    bias="none",
    task_type="CAUSAL_LM",
)
# Training args: LR=2e-5 (10x lower for refinement, not 2e-4),
# batch_size match THK's effective batch (64/16=4 micro × 16 grad_accum),
# max_length=8192 (THK's actual, not 4096),
# 1 epoch over only the failure-type data
# report_to="wandb" (MANDATORY, see logging_policy.md)
# local eval every 10 steps via tracking/
```

## Submission pipeline (from THK's notebook)
After training:
1. Rename keys: `base_model.model.model.*` → `base_model.model.backbone.*`
2. Merge Mamba LoRA: `gate_proj + x_proj + out_proj` → `in_proj`
3. Drop empty `.experts.w3` LoRAs
4. Rewrite adapter_config.json target_modules to the 9-name list
5. Zip adapter_config.json + adapter_model.safetensors

Skip ANY of these and the adapter won't load in vLLM correctly.
