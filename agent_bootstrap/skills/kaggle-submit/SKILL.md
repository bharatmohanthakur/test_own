---
name: kaggle-submit
description: Gate-guarded Kaggle submission for the Nemotron competition. Use BEFORE `kaggle competitions submit` or Playwright submit click. Runs the local bench on the trained adapter, compares to v26 baseline, blocks submit if regressing, then uses the CPU-only kernel pattern so zero GPU quota is spent.
---

# kaggle-submit — Pre-submit gate

## RULE: NEVER submit without a local bench result newer than the adapter.

Every disaster (v27=0.82, v28=0.67, v29=0.22, GRPO_B200=0.47) would have been caught by a 20-min bench. Submission slots are scarce — 5/day.

## The gate (in order)

### 1. Local bench (required)
```bash
# Pick the cheapest eval GPU (H100 $1.50/hr, CUDA 12.6 works for vLLM)
cd tracking/
python run_bench_fast.py --adapter adapters/v<N> --max-tokens 2048
# or on Kaggle RTX 6000: see notebooks/bench_v28_v26_vllm/
```
- Runs 60 prompts (10 per puzzle type), vLLM batched, temp=0
- ~20 min on RTX 6000 Blackwell, ~15 min on H100
- **Local ↔ Kaggle offset = +0.12 ±0.01**. So local 0.73 ⇒ est Kaggle 0.85.

### 2. Compare to v26 baseline
```bash
python tracking/compare.py --baseline adapters/thk_v26 --candidate adapters/v<N>
```
- v26 local: 44/60 = 0.733 (Kaggle 0.85)
- **Block submit if candidate < baseline − 0.02** (noise floor at n=60 is ±0.03).

### 3. Per-type regression check
```
Binary 9/10 → 0/10 happened on v28 WITHOUT touching binary data.
Always compare per-type. If any type drops >3/10 → DON'T submit, diagnose first.
```

### 4. Format sanity check on the adapter
```bash
python tracking/submit_gate.py --adapter adapters/v<N>
# verifies: \boxed{} emits AFTER </think>, adapter_config.json has all-linear targets,
# lm_head present if refined from v26, SVD transform applied if submitting THK-raw
```

### 5. Submit via CPU-only Kaggle kernel (zero GPU quota)
Pattern (see `kaggle_submit_workflow.md` memory):
- Push adapter as Kaggle dataset: `kaggle datasets version -p adapters/v<N>/ -m "v<N>"`
- CPU kernel loads base model + LoRA, runs inference on hidden test, writes `submission.csv`
- Click Submit via Playwright OR `kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge -f submission.csv -m "v<N>"`
- **Do NOT use a GPU kernel** for submit — burns quota you need for training

## What to record after each submit
Append to `notes/submissions.md`:
```
v<N> | YYYY-MM-DD | base=thk_v26|base | LR=5e-5 | data=<file> n=<count>
local bench: <n>/60 = <score> (est Kaggle <score+0.12>)
per-type: binary N/10, cipher N/10, crypt N/10, gravity N/10, roman N/10, unit N/10
Kaggle score: <actual> | rank: <n>
notes: <what changed, what surprised>
```

## Red flags that mean STOP, don't submit
- Local bench < v26 − 0.02
- Any type dropped >3/10 vs v26
- Training loss never went below 2.0 (for from-base) or 1.0 (for refine)
- ≥ 5% of adapter outputs hit max_tokens with no `\boxed{}`
- Wandb run was offline / logs missing (can't audit)
- Adapter was saved via `trainer.save_model()` and is 0 bytes (known Kaggle silent-fail)

## After submit
- Check score in 15–30 min via Playwright leaderboard OR `kaggle competitions submissions -c nvidia-nemotron-model-reasoning-challenge`
- If local predicted vs actual > 0.03 off → investigate (hidden test drift, format issue, submit kernel bug)
- Update `MEMORY.md` index if this run established a new baseline

## Memory pointers
- `tracking_system.md` — local bench protocol
- `bench_vllm_workflow.md` — Kaggle RTX 6000 vLLM bench
- `kaggle_submit_workflow.md` — CPU submit kernel pattern
- `v27_regression.md`, `v28_regression.md`, `v29_disaster.md`, `grpo_b200_result.md` — what the gate prevents
