---
name: track_a1_bo8_diagnostic
description: 2026-05-09 best-of-N+vote bench on v26 alone proved sampling diversity adds only +0.017 (marginal). Critical: crypt and unit have 0/8 correct paths even at temp=0.7, so the bottleneck is training, not sampling. Submission is temp=0 single-shot so bo8 can't ship anyway.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
**Result (60-prompt local bench, 2026-05-09):**
- Greedy (n=1, temp=0):  44/60 = 0.733
- BoN-vote (n=8, temp=0.7): 45/60 = 0.750
- Oracle (any-of-8 correct): 46/60 = 0.767

**Per-type:**
| Type | Greedy | BoN-vote | Oracle |
|---|---|---|---|
| cipher | 1.00 | 1.00 | 1.00 |
| roman | 1.00 | 1.00 | 1.00 |
| binary | 0.90 | 0.90 | 1.00 |
| gravity | 0.70 | 0.80 | 0.80 |
| unit | 0.60 | 0.60 | 0.60 |
| cryptarithm | 0.20 | 0.20 | 0.20 |

**Why:** Inference-time scaling could have been the cheap path to 0.90, but two findings rule it out:
1. **Submission rule = temp=0 single-shot** (per `notes/competition-summary.md`). bo8 can't ship.
2. **Crypt and unit are 0/8 in oracle** = the model has NO correct paths even with diverse sampling. Sampling diversity cannot create paths that don't exist.

**How to apply:** When considering inference-time tricks (best-of-N, prompt-perturb, ensembling), check the per-type oracle ceiling first. If oracle == greedy on weak categories, training-side intervention is the only lever. Don't waste GPU on sampling experiments for categories where oracle ceiling proves the model lacks the capability entirely.

**Implication for next steps:**
- Track F (on-policy distillation) and Track G (joint CSP curriculum) are the only paths forward for crypt 2/10.
- Gravity has +0.10 oracle headroom = some sampling-fixable cases exist; targeted gravity training data could capture this without changing inference.
