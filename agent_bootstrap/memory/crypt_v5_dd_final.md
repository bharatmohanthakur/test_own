---
name: crypt-v5-dd-final
description: "crypt_v5 (DIGITS->DIGITS half, 612 perfect decomposed traces) = 4/24 (16.7%) held-out. SIXTH and FINAL crypt experiment. Comprehension+termination fixed; failure is now pure RULE-SELECTION — model can't deduce which rule at temp=0. ~16.7% = first-guess luck rate. Crypt is closed for good."
metadata:
  type: project
---

**2026-06-10.** Discovered crypt splits 823 SYMBOLS->SYMBOLS / 732 DIGITS->DIGITS (47%).
Built a 100%-coverage DD solver (gen_crypt_dd.py, all 732 solved, 8354 verified arithmetic
claims, new negabsdiff family). Trained crypt_v5 fresh-base on 612, benched 24 HELD-OUT:
**4/24 = 16.7%.** 23/24 terminated decisively (~600 tok) — comprehension (digit-native) and
termination both FIXED — but answers wrong via RULE-SELECTION errors (wrong family/reversal/
sign convention; e.g. pred '31' vs exp '/13', pred '-91' vs exp '04').

**The complete 6-experiment picture:** terse 16.7% / verbose 3.3% / decisive 10% / decomposed
0% / DD-decomposed 16.7% holdout. Every fixable sub-skill got fixed (format, termination,
comprehension, arithmetic execution) — what remains, candidate-elimination deduction at greedy
temp=0, never moved. 16.7% = the rate the first-guess rule is luckily right.

**Closed forever:** crypt is not trainable into this model under comp rules. Best use of
remaining time/budget: protect v26 (0.85); optional careful non-crypt refine (v45_binary_
solver_topup ready, LR<=2e-5, gate on local bench vs v26). HF bench on Blackwell pods needs
transformers-package cache_position patch (site-packages/transformers/models/nemotron_h/) —
the /workspace/model glob patches nothing on transformers 5.5.
