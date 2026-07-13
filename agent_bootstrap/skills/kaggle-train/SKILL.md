---
name: kaggle-train
description: Train a Nemotron LoRA adapter for the Kaggle reasoning competition. Use when launching any SFT/GRPO/GSPO/RAFT run. Encodes the v26 baseline config, Blackwell fixes, forbidden moves (packing w/o flash_attn_2, lm_head rename, boxed_weight), mandatory wandb, and the "refine from v26 at LR ≤ 5e-5" rule.
---

# kaggle-train — Nemotron training playbook

## Gold baseline (never break)
- `thk_v26` = **0.85 Kaggle** (our floor). Adapter at `adapters/thk_v26/`.
- Any new run must beat v26's local bench (44/60) before submit.

## Proven config (use as default)
```python
base_model = "nvidia/Nemotron-3-Nano-30B"
LoraConfig(
    r=32, lora_alpha=32, lora_dropout=0.0,
    target_modules="all-linear",   # THK's choice; DO NOT add QKV-only or router
    modules_to_save=["lm_head"],   # THK includes it — keep
    bias="none", task_type="CAUSAL_LM",
)
# training
max_seq_length=8192
packing=True                        # ONLY if flash_attention_2 loads — else False
flash_attn_2=True                   # MANDATORY with packing
per_device_train_batch_size=2
gradient_accumulation_steps=16      # eff batch 32
learning_rate=2e-4                  # FROM-BASE only
num_train_epochs=1
optim="adamw_torch_fused"
lr_scheduler_type="cosine"
warmup_ratio=0.05
report_to="wandb"                   # MANDATORY — never "none" + nohup
```

## LR rules (violation history ⇒ regressions)
| Scenario | LR | Why |
|---|---|---|
| From base (Nemotron-3-Nano-30B raw) | 2e-4 | THK's original |
| **Refine from v26** | **≤ 5e-5** | v28 used 1e-4 → 0.67 (−0.18 regression) |
| DPO / RAFT on refined | 1e-5 to 5e-5 | preserve format |

## Forbidden moves (known regressions)
- ❌ `target_modules=["q_proj","k_proj","v_proj"]` — v8 scored 0.60 vs 0.69 baseline
- ❌ `packing=True` without flash_attn_2 loaded — v28 cross-contaminated binary 9/10 → 0/10
- ❌ Renaming `backbone.lm_head.*` → `lm_head.*` then retraining at LR=1e-4 — v28 disaster
- ❌ `boxed_weight` loss term — v? hurt, removed from recipe
- ❌ Training FROM BASE on CoT-only data (no `\boxed{}` after `</think>`) — v29 = 0.22
- ❌ `target_modules` including `router|gate|expert` regex — Mamba router layers break
- ❌ Adding `embed_tokens` to LoRA targets for 30B — OOM risk, rarely helps
- ❌ `report_to="none"` + `nohup` — running blind, no recovery from crashes

## Pre-flight checklist (MANDATORY before launching)
1. **Data format check**: ≥ 95% of examples have `\boxed{ANSWER}` AFTER `</think>`. Verify with:
   ```python
   import json
   lines = [json.loads(l) for l in open("data/your_training.jsonl")]
   good = sum(1 for x in lines if "\\boxed{" in x["text"].split("</think>")[-1])
   print(f"{good}/{len(lines)} = {good/len(lines):.1%}")  # want ≥ 0.95
   ```
2. **Ground-truth match**: every `\boxed{}` answer must match `train.csv` label. See `kaggle-data-quality` skill.
3. **Smoke test**: run with 20–50 examples first. If loss doesn't drop from ~3 to ~1.5 in 10 steps, STOP and diagnose.
4. **wandb online**: `wandb login --verify` returns success. Fallback to `WANDB_MODE=offline` only if network blocked.
5. **Blackwell fixes loaded** (if RTX Pro 6000 / B200 / H200): see `blackwell_training_fix.md` memory — 5-step patch.

## GPU selection
| Workload | GPU | $/hr |
|---|---|---|
| Training 30B + LoRA + grad ckpt | B200 / RTX Pro 6000 Blackwell / H200 | $0.92–3.44 |
| Eval (load + vLLM batch) | H100 SXM (CUDA 12.6 works) | $1.50 |
| NEVER: use B200 for eval | — | wastes premium + flashinfer broken on sm_100a |

## What doesn't work for this comp (don't retry)
- **GRPO with vLLM colocate** — INCOMPATIBLE with PeftModel + NemotronH. Use native generation (~100s/step).
- **GRPO with beta=0.0 or reward<50%** — v38a(0.72) + GRPO → 0.47 disaster. Need beta>0 + Goldilocks data.
- **NeMo-RL "single GPU" recipe** — it's actually 32×8 multi-node. Skip.
- **Puzzle-KD 50K blind mix** — v36 cipher collapsed.

## Workflow
1. Start pod (see `vastai-pod` skill). Use `stop` never `destroy`.
2. Upload training data + script via `vastai copy` or scp.
3. Launch with `nohup python train.py > train.log 2>&1 &` AND `wandb` online.
4. Monitor wandb loss curve. Kill if diverging (loss > 5 after step 50).
5. On completion: save adapter → Kaggle dataset OR scp locally to `adapters/v<N>/`.
6. **STOP (don't destroy) the pod**.
7. See `kaggle-submit` skill for bench-before-submit gate.

## Recent memory pointers
- `thk_v26_baseline.md`, `thk_adapter_analysis.md` — canonical config
- `v27_regression.md`, `v28_regression.md`, `v29_disaster.md` — what NOT to do
- `blackwell_training_fix.md` — 5-step Blackwell patch
- `training_methods_sota.md` — SOTA ordering: RAFT first, then token-priority SFT, GSPO last
