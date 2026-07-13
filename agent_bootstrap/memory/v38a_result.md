---
name: v38a result Apr 12 2026
description: v38a scored 0.72 — NEW BEST. THK sequence-matching binary solver (1131 traces at 70.6%) + Kh0a non-binary (1550) + Kh0a 11-target config.
type: project
originSessionId: 9eb28593-1e7c-47c0-8e2b-71a04fc43abb
---
# v38a Result (Apr 12, 2026)

**Kaggle public score: 0.72** — NEW BEST (+0.03 over v37's 0.69, +0.05 over v34c's 0.67)

## What worked
- 1131 binary examples from THK-style sequence matching solver (70.6% coverage, expanded gates: AND-NOT, OR-NOT, XOR-NOT)
- 1550 non-binary from Kh0a's formatted_train_dataset.csv
- Kh0a's exact config: alpha=16, dropout=0.1, LR=1e-4, 11 targets incl embed_tokens+lm_head, 2 epochs, batch=2
- Total: 2681 examples

## Score progression
| Version | Score | Delta | Key change |
|---------|-------|-------|------------|
| v34c | 0.67 | — | v22 base + 50 binary verifier |
| v37 | 0.69 | +0.02 | Kh0a data 2450 + Kh0a config |
| **v38a** | **0.72** | **+0.03** | THK binary solver + Kh0a |
| v39 | pending | — | Full 3910 Kh0a + boxed weight |

## What the +0.03 likely came from
- Binary: THK's expanded gate search → model sees AND-NOT, OR-NOT, XOR-NOT rules → better binary accuracy (probably 30-40% vs v37's ~22%)
- Kh0a config maintained: cipher ~92%, equation ~70%, gravity/roman/unit 100%

## Next steps
- v39 (training on Kaggle) should be ~0.73 (matching Kh0a with full data + boxed weight)
- GRPO on RunPod running in parallel
- To push past 0.73: need 3-input gates (MAJ/CHO/PAR3) for binary + more equation coverage
