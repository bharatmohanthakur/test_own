---
name: v28 training workflow
description: v28 = THK v26 + 538 corrected crypt traces. Gated by validation before submit.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
## v28 Status (Apr 17, 2026)

**Goal:** push from 0.85 → 0.87-0.89 by fixing 538 crypt training examples.

**Ingredients ready:**
- `data/thk_training_v3.jsonl` (16720 examples, 538 crypt corrections) — local, not uploaded
- `adapters/thk_v26/submitted/submission.zip` (3.3GB, THK v26 transformed) — local
- `runpod_v40/train_refine_thk_v28.py` — B200 training script
- `tracking/run_bench_fast.py` — fast eval (H100, 15 min)
- `tracking/compare.py` — local diff of two adapter evals
- `tracking/submit_gate.py` — refuses submit if regression
- **Vast.ai balance: $24.08** (topped up Apr 17)

## Execution order

1. **Upload v3 training data to Kaggle** (3 min, free)
   `kaggle datasets create -p datasets/thk_training_v3/`

2. **Rent B200 Vast.ai** ($3.44/hr, est 3 hrs = $10)
   - Image: pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
   - Disk: 120 GB

3. **Upload + train** (~3 hrs)
   ```
   scp train_refine_thk_v28.py submission.zip thk_training_v3.jsonl root@<pod>:/workspace/data/
   ssh root@<pod> 'python3 /workspace/data/train_refine_thk_v28.py'
   ```

4. **Save refined adapter to Kaggle dataset** (~5 min, free)
   Upload /workspace/output_v28/refined/ as `bharatmohan/nemotron-v28-adapter`

5. **Rent H100 SXM Vast.ai** for eval ($1.50/hr, 30 min = $1)
   NOT Blackwell — flashinfer needs CUDA 12.8 on Blackwell

6. **Run bench on H100** (~15 min, $0.50)
   ```
   python3 run_bench_fast.py \
     --model /workspace/model --train /workspace/data/train.csv \
     --adapters v26:/workspace/v26 v28:/workspace/v28 \
     --n-per-type 10 --out /workspace/bench_v28.json
   ```

7. **Local compare**
   ```
   python3 tracking/compare.py bench_v28.json
   # ✅ PASS | ⚠️ MARGINAL | ❌ REGRESSION
   ```

8. **Gated submit**
   ```
   python3 tracking/submit_gate.py \
     --bench bench_v28.json --candidate v28 \
     --submission-zip submission_v28.zip \
     --competition nvidia-nemotron-model-reasoning-challenge \
     --message "v28: v26 + 538 crypt corrections"
   ```

## Expected numbers
- v26 overall: 0.85
- v28 target: 0.87-0.89
- Crypt-specific: 0.25 → 0.55 (biggest lift)
- Budget: ~$15 total

## Guardrails
- If v28 local < v26 local → submit_gate blocks, investigate
- If v28 local > v26 but < +0.01 → marginal warning, don't burn slot
- If training loss diverges or eval regresses during training → early stop
- Keep pods alive until user says destroy (per `feedback_pod_destroy.md`)

## Keys/refs
- Vast.ai key: `memory/vastai_key.md`
- Wandb key: `memory/wandb_key.md`
- THK v26 adapter URL: `kaggle models download huikang/nemotron-adapter/transformers/default/26`
