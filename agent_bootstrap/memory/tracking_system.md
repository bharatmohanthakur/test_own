---
name: Tracking system for adapter evaluation
description: Location, usage, and capabilities of the ground-truth-aware evaluation framework at tracking/ — used after every training run to understand WHY an adapter succeeds or fails before deciding to submit.
type: reference
---

# Tracking System (tracking/)

Location: `/Users/bharat/Downloads/kaggle/tracking/`

Purpose: after training a new adapter, run the 60-prompt benchmark locally, verify the model's **internal reasoning** against ground truth derived from the prompt, produce a per-type failure report, and decide whether to submit.

## Validated calibration
60-sample local score matches Kaggle public score within ±0.01:
- v30: local 0.550 / Kaggle 0.540
- v31: local 0.667 / Kaggle 0.660

**Rule**: if the local score isn't better than the prior adapter's local score, do NOT submit. Save Kaggle submissions for real improvements.

## Key tools

| file | purpose |
|---|---|
| `run_eval.py` | batched vLLM inference on pod, saves full prompts + traces |
| `analyze.py` | raw traces → per-type metrics + markdown report |
| `diff.py` | compare two adapters (regressions/improvements/both-wrong) |
| `history.py` | append run summary, show trajectory table |
| `verifiers/cipher.py` | derives true substitution, grades model's table + VER honesty + vocab-fill abuse |
| `verifiers/binary.py` | classifies true per-bit gate, detects lazy patterns |
| `verifiers/equation.py` | detects cipher-template-misfire on arithmetic |
| `verifiers/{gravity,unit,roman}.py` | ground truth + drift detection |

## Standard workflow after training
```bash
# 1. Upload run_eval.py + adapter, run on pod
scp -P $P tracking/run_eval.py root@$H:/workspace/
ssh -p $P root@$H "python3 /workspace/run_eval.py \
    --model /workspace/model \
    --train /workspace/data/train.csv \
    --adapters NEW:/workspace/adapter_new PREV:/workspace/adapter_prev \
    --n-per-type 10 --out /workspace/eval.json"

# 2. Download raw JSON
scp -P $P root@$H:/workspace/eval.json tracking/history/runs/$(date +%F)_NAME.json

# 3. Analyze + diff + history
cd tracking/
python3 analyze.py history/runs/$(date +%F)_NAME.json
python3 diff.py history/runs/$(date +%F)_NAME.json PREV NEW
python3 history.py history/runs/$(date +%F)_NAME.json --notes "what changed"
python3 history.py --show
```

## What the verifiers catch (that string-match misses)

- **Cipher `ver_honesty`**: is the model's `VER: PASS` claim justified by its own table? Catches the v30 "template teaches lying" failure.
- **Cipher `table_accuracy`**: fraction of model's letter mappings that match ground truth. v30 was 0.79 vs v31 0.98 — explains why v30 broke on cipher.
- **Cipher `vocab_fill_used`**: detects the v30 VOCAB-fill rescue that snaps wrong decryptions to wrong words (the actual root cause of cipher regression, not VER lying as first assumed).
- **Binary `matches_simple_truth`**: computes true answer for solvable-with-simple-gates cases, grades the model against it. Shows model's reach beyond string match.
- **Equation `ran_cipher_template`**: detects the v30 cross-type contamination where cipher template was applied to arithmetic puzzles (9/10 v30 equation traces misfired).
- **Drift flags per type**: non-cipher types that start running cipher template → cross-type contamination.

## Bad signals to watch for (would block submit)
- `ver_honesty_dist` heavy on `PASS_LIE` — model rubber-stamping VER
- `vocab_fill_used` == 100% on cipher — will produce hallucinated words
- `binary.matches_simple_truth << solvable_simple` — model can't reach what's reachable
- `equation.ran_cipher_template` > 50% — template poisoning
- `drift_flags` non-empty — cipher template leaking to other types
- Regressions count in diff > improvements count

## Good signals (safe to submit)
- `avg_table_accuracy >= 0.95` on cipher
- `VER honesty` shows `PASS_TRUE` + some `FAIL_TRUE` (honest NO)
- `both_wrong` count shrinking vs prior adapter
- Total local acc > prior local acc by at least 0.02 (within noise floor)

## Expanded validator (built 2026-04-09 via parallel agents)

5-layer pre-submission validator at `tracking/validator.py`. Run:
```
python3 tracking/validator.py history/runs/X.json
```

Combines:
- L1: per-type accuracy (existing)
- L2: 8-step reasoning chain grading per type (`chain_grader/{type}.py`)
- L3: behavioral fingerprint (`fingerprint.py` — template_label_count, numeric_density, distinct_rule_attempts)
- L4: confidence calibration (`calibration.py` — ECE, overclaim_rate, conf_when_wrong)
- L5: source attribution (`attribution.py` — credits training data sources for changes)

**Validated on v30/v31 — caught v30 on every relevant gate (verdict: ABORT):**
- cipher_accuracy 0% < 80% (hard gate fail)
- regressions > improvements (hard gate fail)
- overclaim_rate 41.7% > 30% (lying-VER signature, calibration ECE 0.20 vs v31's 0.03)

v31 → MARGINAL (passes hard gates, fails soft gates because total 0.667 < 0.70 strong-submit threshold)

**Decision rules:**
- HARD gates (any fail → ABORT): cipher >= 80%, no template drift, regressions < improvements
- SOFT gates: total >= 0.70, binary >= 50%, overclaim <= 30%, ECE <= 0.10
- Verdicts: STRONG SUBMIT / MARGINAL / WEAK / ABORT

**Training data source tagging**: `data/generate_v34.py` adds `_meta.source` (v22_base or v34_binary_solver) and `_meta.rule_kinds` per binary example. Attribution cross-references these to credit sources for score changes.

**Mandatory pre-submit workflow:**
```
1. python3 tracking/validator.py history/runs/<new>.json
2. Read verdict
3. ABORT → don't submit, iterate on training data
4. WEAK → don't submit, investigate failed gates
5. MARGINAL → submit only if total > prior best
6. STRONG SUBMIT → submit
```
