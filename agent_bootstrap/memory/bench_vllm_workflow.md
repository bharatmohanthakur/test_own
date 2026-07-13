---
name: Kaggle bench-vllm workflow (60-prompt per-type eval)
description: Validated offline vLLM benchmark on Kaggle RTX 6000 — compares adapters per-type. 20 min / run. Inline SVD transform handles THK-style raw adapters.
type: reference
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Location:** `notebooks/bench_v28_v26_vllm/train.py`
**Kernel slug:** `bharatmohan/nemotron-bench-vllm`
**Validated:** Apr 18, 2026 (v26=44/60, v28=33/60 — within ±0.01 of Kaggle scores after +0.12 offset)

## What it does
- Loads Nemotron-3-Nano-30B base + two LoRA adapters via vLLM (LoRARequest)
- Samples 60 prompts from train.csv (10 per type × 6 types, seed=42)
- Generates with Kaggle eval params: temp=0, max_tokens=7680, max_model_len=8192
- Outputs `/kaggle/working/bench_out.json` with per-type correct/total

## Config that matters
```python
# MUST set BEFORE any import for spawn issue
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
```

`VLLM_USE_V1=0` is **obsolete** in modern vLLM (ignored with "Unknown vLLM environment variable" warning).

## kernel-metadata.json
```json
{
  "id": "bharatmohan/nemotron-bench-vllm",
  "enable_gpu": "true",
  "enable_internet": "false",
  "dataset_sources": [
    "bharatmohan/thk-nemotron-v26-raw",
    "bharatmohan/nemotron-v28-crypt-adapter",
    "mayukh18/nemotron-packages"
  ],
  "competition_sources": ["nvidia-nemotron-model-reasoning-challenge"],
  "model_sources": ["metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"]
}
```

**No internet** — all wheels come from `mayukh18/nemotron-packages` (vllm, peft, causal_conv1d, mamba_ssm).

## Push command
```
kaggle kernels push -p notebooks/bench_v28_v26_vllm --accelerator NvidiaRtxPro6000
```
CLI `--accelerator` flag locks GPU — no UI workflow needed.

## Inline v26-raw transform (THK format → HF format)
If the adapter under test was saved in THK's legacy naming (`base_model.model.model.*`, `.experts.w1/w2/w3`, `gate_proj+x_proj`), the bench script applies the same transform as `submit_thk_v26/train.py`:

1. Rename `base_model.model.model.*` → `base_model.model.backbone.*`
2. Unfuse MoE: `.experts.w1` → per-expert `.experts.{i}.up_proj`, `.w2` → `.down_proj`, drop empty `.w3`
3. Mamba merge: `gate_proj + x_proj` → `in_proj` via SVD (rank=32)
4. Rewrite `adapter_config.json` target_modules to 9-name list incl lm_head

Result lives in `/tmp/v26_transformed/` then passed to `LoRARequest(path=/tmp/v26_transformed)`.

Adapters trained with PEFT directly (like v28) skip this step — they're already HF-compatible.

## Timing (RTX 6000 Blackwell 102GB)
- Wheel install: ~45s
- vLLM load: ~10min (model weights + lora init)
- SVD transform for v26-raw: ~30s
- 60 prompts × 7680 max_tokens × 2 adapters: ~7 min
- **Total wall time: ~20 min**

## Calibration
Local 60-prompt bench vs Kaggle public:
- v26: 0.733 local ↔ 0.85 Kaggle (+0.12)
- v28: 0.550 local ↔ 0.67 Kaggle (+0.12)

Offset is consistent (±0.01). Use the bench to **compare adapters**, not predict absolute Kaggle score.

## Pitfalls (all validated and fixed)
1. **vLLM v1 engine spawn crash** on Kaggle script runner → set `VLLM_WORKER_MULTIPROC_METHOD=fork`
2. **protobuf 6.33.6** incompatibility → use mayukh18 wheels (bundles compatible version); don't try to force `protobuf<5` (not in wheels, 404s offline)
3. **v26-raw adapter target modules mismatch** → inline SVD/rename transform (copy-paste from submit_thk_v26/train.py)
4. **CLI push always starts P100** unless `--accelerator NvidiaRtxPro6000` is passed
5. **Kaggle `kernels output` CLI** returns logs + JSON reliably once kernel COMPLETE. During RUNNING, log may not be exposed yet.
