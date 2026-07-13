---
name: crypt-final-closure
description: "FINAL crypt closure 2026-06-11: 11 experiments all ≤25%. v9 leaders-technique pattern-recall 0/24+0/24 (4x verbosity drift, atom-soup decode); v10 SimPO-on-v9 with gold-fallback pairs hit pref_acc 0.94 in training yet 0/24+0/24 greedy — preference learning provably does not move greedy decode. Only unfalsified lever: 10x synthetic corpus scale. v26 0.85 stands."
metadata:
  type: project
---

**2026-06-11, the last crypt experiments.**

- **v8** (example-major intersection tables: full 22-candidate first-example filter, survivor-only
  verification, X-rows 4.6%): **0/24 holdout.** 16/24 overflow (drift 2.4x — example-major didn't
  tame verbosity either); 8 finished all wrong, with near-miss convention errors (pred '3372' vs
  exp '2733' = exact reversal slip).
- **Shaped-reward pipeline** (user's hypothesis: proper reward fixes what binary reward can't):
  best-of-8 sampling found correct answers on **28/300 prompts (9.3%)** and built 104 contrast
  pairs (reward = answer 1.0 + rule 0.4 + claims-true 0.2 + finished 0.1). The signal EXISTS in
  sampling. **v8+DPO on those pairs: 1/24** — and generation got MORE verbose (avg 6518 tok,
  19/24 ceiling). DPO (ref-free via PEFT adapter-toggle, beta 0.1, precompute_ref_log_probs,
  max_len 2816 after OOM) could not transfer sampled contrast into greedy decoding at 104 pairs.
  (SimPO untestable: TRL 1.4 removed CPOTrainer; user rightly noted a separate venv with old TRL
  was the cleaner fix — lesson: version conflicts get a venv, not an algorithm swap.)

**COMPLETE LEDGER (all on same/comparable holdouts):** terse 16.7% | verbose-verify 3.3% |
decisive 10% | decomposed-glyph 0% | DD-decomposed 16.7% | v6 verbose-tables 25% | v7 terse-tables
0% | v8 example-major 0% | v8+DPO 4.2% | RL-Goldilocks probe: 18/24 prompts 0/8 (GRPO starves).
In-dist ≈0 whenever measured. v6's 25% = noise high-water at n=24.

**Closed conclusions:** (1) greedy temp-0 cannot execute the candidate-elimination + multi-step
execution chain regardless of trace algorithm; (2) generation verbosity does NOT anchor to
exemplar length (drift 1.7-3.6x, self-amplifying); (3) preference optimization at feasible pair
counts cannot bridge sampled-success → greedy-success. Crypt ends at v26's level. PERIOD.

**Remaining season assets:** v26 0.85 submitted; r2 = first-ever COMPLETE v26 refine (binary
topup, transform verified bitwise, train_loss 1.03) parked locally at adapters/r2_kaggle/
(3.3GB tgz — contains PEFT embed/lm_head full weights, audit before push), gate = Kaggle bench
vs v26. Deadline June 15.

---
**2026-06-11 ADDENDUM — the properly-done version, final.** k=16 sampling over 600 prompts:
**119/600 (19.8%) prompts had a fully-correct rollout** (2x the k=8 rate — sampling finds wins).
RAFT filter (chosen = correct-only) + manual length-normalized SimPO (no TRL, 15 updates,
beta 2.0 gamma 0.5 lr 8e-6): **holdout 1/24 — identical to DPO.**
KEY MEASUREMENT: pref_acc started at 0.19-0.25 — the greedy policy assigns HIGHER likelihood
to its wrong generations than its own correct ones on ~80% of pairs. The sampled→greedy gap is
deep, structural, and not bridgeable by feasible preference optimization on this model.
SFT (8 variants), DPO, SimPO-with-RAFT, RL-probe: ALL measured. Crypt = v26's level. THE END.

---
**2026-06-11 ADDENDUM 2 — v9 pattern-recall + v10 SimPO-on-v9. Experiments #10-11, both 0/24.**

- **v9** (the LEADERS' technique from the discussion writeup: 51 single-BPE Cyrillic atoms,
  432 canonicalized mul-patterns → 4,692 memorized candidates, first-rejection filtering,
  lm_head LoRA, loss 0.136 = deepest convergence ever): **SS 0/24, DD 0/24.** 19/24 + 17/24
  ceiling at avg ~6,300 tok despite 1,576-tok traces (4x drift). Finished preds = atom-soup
  ('(+×?') — atom↔symbol decode never anchored. Likely cause of leaders' success vs ours:
  corpus scale ("crypt dominated token share" = thousands of synthetic puzzles, ours 515) —
  and writeup author himself plateaued 0.86.
- **v10** (SimPO on v9, the length-normalized objective aimed EXACTLY at the overflow failure;
  300 pairs with GOLD-FALLBACK — chosen = verified 1,576-tok ref trace when no rollout correct,
  which was 298/300 since v9 pass@6@temp0.8 = 2/300 = 0.7%): trained pref_acc 0.19→**0.94**,
  loss 0.83→0.73 — the preference FULLY flipped in training. **Transfer to greedy: SS 0/24
  (finished 7/24 vs 5, avg 5,556 vs 6,304 — marginal shortening), DD 0/24 (worse: 2/24
  finished, avg 7,110).** Near-miss: pred ':|' vs exp ':!:|'.

**Strongest evidence yet for the core law: preference-space learning (even at pref_acc 0.94
on an objective that directly encodes the failure mode) does NOT move the greedy decode path
of this 3B-active MoE.** That's now shown for correctness-contrast (DPO), shaped-reward
contrast (RAFT-SimPO), and length-contrast (gold-fallback SimPO).

The ONLY unfalsified lever left: corpus scale — synthesize thousands of puzzles from our
generators (we own the rule families), crypt-dominant mix, more epochs. Needs ~$15-18 +
~1 day; balance after v10 = $6.48. v10 artifacts: data/bench_v10_{ss,dd}.json,
data/v10_pairs.jsonl, data/rollouts_v9_k6.jsonl (1,800 gens), adapters/crypt_v9/ local.
