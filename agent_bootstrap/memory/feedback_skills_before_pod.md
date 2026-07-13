---
name: feedback_skills_before_pod
description: ALWAYS load Skill(vastai-pod) BEFORE renting any GPU pod. Last session burned $7 by skipping the validated workflow.
type: feedback
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Rule**: Before any `vastai create instance`, `vastai start instance`, OR `ssh root@<pod>` Bash call — invoke `Skill(vastai-pod)` first. Then read `vastai_h200_workflow.md` and `runpod_workflow.md` from memory.

**Why**: 2026-05-02 session burned **$7+ over 4 hours** because:
- Rented B200 with broken CUDA driver ($3.30) — no skill check
- Built mamba_ssm from source twice ($0.50+) — memory said use mayukh18 wheels
- `pip install transformers/vllm` pulled torch+cu13, broke torchvision ($1+) — workflow says use `--no-deps` after mayukh18 wheels
- Multiple "loading" pods with stuck images ($1+) — skill says specific image only

**How to apply**:
1. **First action of any pod-touching turn**: `Skill(vastai-pod)`
2. **Before pip install on remote**: re-read `vastai_h200_workflow.md` Step 3+4 install order
3. **Before mamba_ssm/causal_conv1d install**: check if mayukh18 wheels work for current torch version FIRST

**Hookify rules added** (`.claude/hookify.*.local.md`):
- `vastai-pod-skill-required` — blocks `vastai create instance` (bypass: `# skill-loaded`)
- `remote-pip-torch-drift` — blocks remote pip install of transformers/vllm/torch without --no-deps/--find-links
- `no-mamba-source-build` — blocks source build, points to mayukh18 wheels
- `no-4bit-mamba` — blocks `load_in_4bit=True`/bnb_4bit/BitsAndBytesConfig (Mamba SSM incompatible) — bypass: `# nf4-acknowledged`
- `read-memory-before-train` — blocks `ssh ... python train_*` until 5-item checklist confirmed (bypass: `# memory-checked`)

**Pre-flight hook script** (`.claude/hooks/preflight_pod.sh`):
- Validates balance >$5
- Validates training data file exists
- Validates training script exists
- Validates correct image flag
- Validates --disk ≥150
- **Validates driver ≥580** (vLLM 0.19+ PTX needs newer toolchain — older driver = `cudaErrorUnsupportedPtxVersion` mid-bench)
- Bypass: `# preflight-acknowledged`

**Driver lesson** (2026-05-03): Tennessee pod driver 570 trained fine but vLLM bench failed with PTX error. Cost $0.40 wasted. Fix: always filter offers by `driver_version>=580` before renting if vLLM bench is needed.

**Bypass**: only if explicitly verified (e.g. wheels incompatible with torch version).
