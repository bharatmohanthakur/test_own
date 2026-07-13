---
name: crypt-v6-v7-tabulate-final
description: "Tabulate-intersect arc FINAL: v6 (verbose tables, 1ep) = 6/24 holdout with 6/6 correct-among-finished; v7 (terse tables, 4ep) = 0/24, 0/5 correct-among-finished, length barely moved. Generation length does NOT anchor to exemplar length; terse pruning destroyed robustness. Crypt closed after 7 experiments — v26 0.85 stands."
metadata:
  type: project
---

**2026-06-10, experiments 6-7 (the last).** ENUMERATE→TABULATE→INTERSECT traces — the
branch-free algorithm a greedy decoder can execute.

- **v6**: 694 verbose table traces (avg 4,964 chars), 1 epoch (budget salvage) → holdout
  **6/24 (25%, best-ever)**, and ALL 6 finished generations were CORRECT; the 18 fails all
  hit the 7,680-token ceiling (avg 6,163 gen tokens). Looked like a length-only problem.
- **v7**: traces trimmed to 0.45x (avg 2,214 chars, prune-at-first-✗, 12 candidates), 4
  epochs → holdout **0/24**. Gen length BARELY dropped (5,934 avg; 18/24 still ceiling) and
  the 5 finishers were all WRONG.

**Two hard lessons:**
1. **Generation length does not anchor to exemplar length** for this model — verbosity at
   inference is self-amplifying drift, not style imitation. Halving traces moved gen length
   ~4%.
2. **Pruned tables removed the redundancy that made v6's completions correct** — single-check
   dismissals mean one arithmetic slip silently flips the survivor; v6's full rows
   cross-checked. And v6's 6/6-correct was n=6 shine, not a robust property.

**SEASON VERDICT on crypt (7 experiments: terse/verbose/decisive/decomposed/SimPO-killed/
table-verbose/table-terse):** best = 25% on the easier DD half ⇒ crypt overall ≲0.17,
BELOW v26's implied ~0.3. Nothing beats v26 on crypt. CLOSED. Remaining value: the audited
binary refine-from-v26 (running as r1) gated by Kaggle bench; else hold 0.85.
