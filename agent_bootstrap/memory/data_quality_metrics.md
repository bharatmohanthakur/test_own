---
name: Data quality metrics for training iteration
description: Concrete measurable signals to track per training run so we know whether a data change is helping or hurting — grounded in the v30 vs v31 failure analysis.
type: project
---

# Data Quality Metrics (for each adapter / iteration)

Run `/tmp/infer_compare.py` (or extension) against 60 train prompts with LoRARequest swap. Track these metrics in a CSV row per adapter version.

## Tier 1 — CAPABILITY (accuracy) — the only metric that matters for the leaderboard
Per-type and total accuracy. Track delta vs previous adapter AND vs v22 base (0.65 baseline) AND vs v31 best (0.66).

```
per_type_acc: {binary, cipher, roman, unit, gravity, equation} → fraction
total_acc:    fraction correct (compare to Kaggle public score)
```

**Target**: total_acc should be within 0.02 of public Kaggle score (60-sample noise floor). If local score and Kaggle score diverge by >0.05, the sample is too small or biased.

## Tier 2 — FORMAT HYGIENE (does the answer even come out?)
Precondition for any accuracy. Kaggle eval only reads `\boxed{…}`.

```
pct_has_boxed:     % of outputs with valid \boxed{...} (per type)
pct_truncated:     % of outputs with no boxed AND char_len > 15000 (hit token limit)
n_boxed_per_trace: avg number of \boxed{...} in trace (>1 means model emitted multiple, risks wrong one being chosen)
```

**Target**: pct_has_boxed = 100% per type. pct_truncated < 5%. If v30 dropped to 90% on cipher, that's a data problem (training examples with missing/late boxed).

## Tier 3 — STEP DERIVATION QUALITY (the v30 failure signal)
This is where v30 broke. Metrics that would have CAUGHT v30 before submission.

### 3a. Table accuracy (cipher-specific but generalizable)
For cipher, extract the TABLE section and count how many letter mappings are correct against the ground-truth substitution (can be derived from the examples in the prompt).

```
cipher_table_accuracy: avg fraction of correct mappings in v30's TABLE
cipher_table_coverage: avg # of mappings emitted
```

**v30 actual**: ~40-60% table accuracy → broke. **Target**: >95% per trace when model emits a table.

### 3b. VER honesty (the biggest tell)
When the model says `VER: PASS`, is the pass justified? Cross-check: apply the derived table to the example input, compare to expected example output.

```
ver_pass_rate:         % of traces emitting VER PASS
ver_pass_when_correct: % of VER PASS that are actually justified (table correctly decodes example)
ver_pass_when_wrong:   % of VER PASS that are lies (table fails to decode example)
```

**v30 actual**: ver_pass_rate=100%, ver_pass_when_wrong≈100% on cipher — catastrophic lying. **Target**: ver_pass_when_wrong < 5%. Track honest NO (VER: FAIL) as a POSITIVE signal.

### 3c. Vocab-fill abuse (cipher)
Count traces where VOCAB fill snaps a partial decryption to a 77-vocab word AND the snap disagrees with the ground truth.

```
vocab_fill_usage_rate:      % of traces using VOCAB fill
vocab_fill_correctness:     when fill is used, does it land on the right word?
vocab_fill_wrong_and_used:  # traces where fill changed a correct partial to a wrong word
```

**v30 actual**: fill_usage=100%, fill_correctness≈0%. **Target**: vocab_fill_usage_rate < 30% (used only as rescue), fill_correctness > 90%.

## Tier 4 — BEHAVIORAL MARKERS (diagnostic, not causal)
These are symptoms, not root causes, but they tell you which way the model is drifting.

```
donald_label_density:  avg # Donald labels (LEN/TABLE/VER/SCAN/ANS/etc) per trace per type
hallucination_markers: avg # "I identify" / "after analyzing" / "obviously" per trace
execution_markers:     avg # "bit N", "step N", "position N" per trace (shows work)
ver_attempts:          avg # ver/check/verify mentions per trace
char_length:           avg char length per trace per type
```

**v31 binary**: halluc=2.0 → a specific failure ("I identify the pattern" hallucination). **Target**: halluc<0.5 per trace. **v30 cipher**: donald_labels=3.8, vocab_fill=100%, ver_pass=100% → pattern of "stamp template and go". **Target**: v30-style label density < 2.5 AND ver_honesty > 0.9 simultaneously.

## Tier 5 — REGRESSION TRACKING
Compare current adapter to every previous adapter on the same 60 prompts.

```
both_right:    # prompts where BOTH current and prev are correct
both_wrong:    # prompts where BOTH are wrong (need new data)
regressions:   # prompts where PREV was right, CURRENT is wrong
improvements:  # prompts where PREV was wrong, CURRENT is right
```

**v30 vs v31**: both_right=31, both_wrong=18, regressions=9 (all cipher), improvements=2 (binary). **Target**: regressions < 2 per iteration. If regressions > improvements, the training data hurt.

## Tier 6 — DATA SOURCE ATTRIBUTION (for mixed datasets)
For each training example in the dataset, track:
```
source_tag:      which source generated this example (v22/v29/donald/grok/etc)
puzzle_type:     which of the 6 types
ver_events:      # of examples where VER said NO and was correct vs lied PASS
derive_events:   # of examples where steps actually derived from prompt vs pre-computed
```

**v29/v30 flaw**: all cipher examples had VER PASS, all had pre-computed tables. The model couldn't learn VER honesty because the training distribution never showed a VER failure.

## Concrete checks for NEXT iteration
Before deploying v34+ SFT/GRPO:
- [ ] Generate 60-prompt validation run against v22 (0.65) and v31 (0.66) AND new adapter
- [ ] Print per-type delta table
- [ ] Print VER honesty table — if ver_pass_when_wrong > 10%, STOP, retrain data
- [ ] Print regression count — if regressions > 2, investigate before submitting
- [ ] Check both_wrong set — target is to shrink this (currently 18, mostly binary+equation)

## What NOT to track
- wandb training loss curves — we have them, they're not predictive of score
- Number of training examples — we've seen 500 and 1000 both underperform
- Adapter file size — always same
- Training wall-clock time — noise

## Stored locations
- 60-sample results: `/workspace/compare_v30_v31.json` (on pod, ephemeral)
- Metrics summary: `/workspace/compare_metrics.json`
- Script: `/tmp/infer_compare.py` (local) and `/workspace/infer_compare.py` (pod)
- Usage: `python3 infer_compare.py` after setting ADAPTER_V30/ADAPTER_V31 paths — swap in any two adapters
