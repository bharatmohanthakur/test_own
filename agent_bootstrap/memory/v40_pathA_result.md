---
name: v40-patha-result
description: "v40 Path A SFT FAILED — local bench 0.43 vs v26's 0.85 baseline; embed_tokens-LoRA + fresh-from-base ≠ improvement"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---

**v40 = 0.433 on 60-prompt local bench** (n=10/type). Trained 2026-06-01 on pod 38935791 (RTX Pro 6000 Blackwell).

Per-type vs v26 (THK 0.85 corpus baseline):
| Type        | v40    | v26 (Kaggle 0.85 implies) |
|-------------|--------|----------------------------|
| roman       | 10/10  | ~10/10                     |
| unit        | 6/10   | ~9/10 ↓                    |
| gravity     | 5/10   | ~9/10 ↓                    |
| cipher      | 3/10   | ~9/10 ↓↓                   |
| binary      | 1/10   | ~9/10 ↓↓↓                  |
| cryptarithm | 1/10   | ~2/10 (unchanged ceiling)  |

**Why v40 failed:**
1. Most binary/cipher rollouts came back with `predicted: ""` — model overthinks past max_tokens=2048 without producing `\boxed{}`. Symptom of CoT length explosion from from-base + diverse training mix.
2. embed_tokens LoRA did NOT lift cryptarithm — still 1/10. Falsifies the "rare-symbol embedding bottleneck" hypothesis from Diag 2.
3. 2,677 rows is too few to teach binary/cipher from base — v26 implicitly leveraged THK's 10K-row corpus.
4. Fresh-from-base lost the v26 baseline competence on every type except roman.

**Why:** The Path A bet was "structural change beats data tweak". The structural change broke the baseline harder than it added anything. v33's same-data approach already lost (0.85 = no gain) — we should have realized v26's gains come from corpus scale + Tinker-style training, not from any architectural lever LoRA could target.

**How to apply:** Do NOT retry from-base SFT on small (<5k) corpora. embed_tokens LoRA is not the lever for symbol crypt. Returning to v26 + targeted refinement (LR ≤ 5e-5, post-v26 init) is the only path that has ever moved the score. See [[thk_v26_baseline]], [[path_to_090]] for the corpus-scale + tool-use direction.

**Status:** NOT SUBMITTED to Kaggle (would have wasted a sub on 0.43). Pod 38935791 exited mid second-bench due to GPU preemption; v40_full adapter still exists on pod, but training/bench investment yielded only this falsification.

Cost summary: ~$3-4 Vast.ai spend across model download + training (81 min) + bench (~5 min).
