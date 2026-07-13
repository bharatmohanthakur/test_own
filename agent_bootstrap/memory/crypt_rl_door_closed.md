---
name: crypt-rl-door-closed
description: "GRPO/RL on crypt foreclosed by pass@8 probe: v6 @ temp0.7 x8 = pass@8 25%, mean win 5.7%, 18/24 prompts 0/8 (zero GRPO gradient). Sampling unlocks NO new puzzles vs temp=0. RL would starve. Crypt fully closed; v26 0.85 stands."
metadata:
  type: project
---

**2026-06-10 final probe.** Hypothesis (user's, theoretically sound): overflow/verbosity is a
BEHAVIORAL failure → RL (reward = boxed==truth) optimizes the policy's own rollouts and could
fix what SFT exemplars can't. Goldilocks precondition: pass@8 ≥ ~40% for dense reward.

**Result: pass@8 = 25%, mean win-rate 5.7%, only 59/192 rollouts finished </think>.** Wins
confined to the SAME ~6 easy puzzles that pass at temp=0; 18/24 prompts = 0/8 → all-fail GRPO
groups → zero advantage signal. RL trains on what it already solves, starves elsewhere.

Combined with [[crypt-v6-v7-tabulate-final]] (7 SFT styles capped ≤25% on the easy half),
EVERY rules-legal lever is now exhausted with evidence: SFT (5 styles), decomposition,
tabulation, RL (foreclosed by sparsity), tool-use (banned), sampling (eval is temp=0).
**Crypt ends at v26's level. Season result rides on v26 0.85 + the r1 binary refine candidate
(local: adapters/v26_refine_binary.tgz, train_loss 1.34) gated via Kaggle bench.**
