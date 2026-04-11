# Nemotron Tracking System

Ground-truth-aware evaluation framework for understanding WHY an adapter succeeds or fails.

**Why this exists**: after training a new adapter, `kaggle competitions submit` tells you the
public score (e.g., 0.54) but gives ZERO insight into what the model is actually doing wrong.
String-match accuracy hides failure modes like "model's substitution table is correct but vocab-fill
snaps the answer to a wrong word". This system runs the 60-prompt benchmark locally, verifies the
model's INTERNAL reasoning against ground truth derived from the prompt, and produces a detailed
report of failure modes per type.

## Quick start

```bash
# 1. After training an adapter, upload adapters to the pod and run inference
scp -P <port> run_eval.py root@<host>:/workspace/
ssh -p <port> root@<host> "python3 /workspace/run_eval.py \
    --model /workspace/model \
    --train /workspace/data/train.csv \
    --adapters v33:/workspace/adapter_v33 v31:/workspace/adapter_v31 \
    --n-per-type 10 \
    --out /workspace/eval_v33.json"

# 2. Download the raw results
scp -P <port> root@<host>:/workspace/eval_v33.json history/runs/2026-04-XX_v33.json

# 3. Analyze
python3 analyze.py history/runs/2026-04-XX_v33.json
# → writes history/runs/2026-04-XX_v33_report.md

# 4. Diff against prior best
python3 diff.py history/runs/2026-04-XX_v33.json v31 v33
# → writes reports/diff_v31_vs_v33.md

# 5. Append to trajectory history
python3 history.py history/runs/2026-04-XX_v33.json --notes "v33 GRPO on v22 base" --kaggle-score 0.67
python3 history.py --show
```

## File layout

```
tracking/
├── README.md               # this file
├── train_loader.py         # load train.csv, classify, lookup full prompt by (type, answer)
├── run_eval.py             # batched vLLM inference on pod (saves full prompts)
├── analyze.py              # raw traces → per-type metrics + markdown report
├── diff.py                 # compare two adapters in the same dump
├── history.py              # append run summary to history.jsonl, show trajectory
├── verifiers/              # per-type ground-truth verification
│   ├── __init__.py         # dispatch table
│   ├── cipher.py           # derive true substitution table, grade model's table + VER + vocab
│   ├── binary.py           # classify true per-bit gate (constant/identity/NOT/2gate/majority)
│   ├── equation.py         # detect cipher-template-misfire on arithmetic
│   ├── gravity.py          # compute true g, detect drift
│   ├── unit.py             # compute true slope, detect drift
│   └── roman.py            # compute true roman numeral, detect drift
├── history/
│   ├── history.jsonl       # append-only run summaries
│   └── runs/               # full raw JSONs archived per run
└── reports/                # generated markdown reports
```

## What the verifiers check (beyond string match)

### Cipher
- Parses prompt example pairs, builds the TRUE substitution table char-by-char
- Extracts the model's emitted TABLE: section
- Grades: `table_accuracy` (fraction of mappings correct), `ver_honesty` (did model's VER: PASS
  claim decode example correctly?), `vocab_fill_used`, per-word answer accuracy
- Failure modes: `template_with_lying_ver`, `template_with_wrong_table`, `hallucinated_vocab`,
  `free_form_wrong_mapping`, `no_boxed`

### Binary
- Parses input→output pairs, tries to classify each output bit as CONSTANT/IDENTITY/NOT/2-input
  gate (AND/OR/XOR/NAND/NOR/XNOR)/3-input MAJORITY
- Distinguishes `SIMPLE` (solvable with these) from `HARD` (3+ input general)
- Detects lazy patterns: `lazy_constant_0/1`, `lazy_copy` (input verbatim), `lazy_not`
- Computes the true answer for SIMPLE cases so we know if the model's answer matches derived truth

### Equation
- Detects SYMBOL-DIGIT vs CIPHER-DIGIT sub-type per Donald playbook
- Counts cipher-template markers (LEN:/TABLE:/DECOMPOSE/CAT/VOCAB fill) and arithmetic markers
- Flags `cipher_template_misfire` — the v30 failure of running cipher pipeline on arithmetic
- Flags `copied_input` and `no_arithmetic_attempted`

### Gravity / Unit / Roman
- Computes the true answer from the prompt
- Detects drift: if the model is running cipher template on these types, flag it
- All three are 10/10 saturated on v30/v31 — verifiers exist for drift detection on future adapters

## Interpreting the report

Key metrics in priority order:

1. **Tier 1 — Per-type accuracy** — ground truth, compare against prior run
2. **Tier 3 — Cipher deep-dive** — `avg_table_accuracy`, `ver_honesty_dist`, `vocab_fill_used`
3. **Binary difficulty** — how many samples are SIMPLE? Of those, how many did the adapter get right?
4. **Equation subtype + misfire** — is the adapter running cipher template on arithmetic?
5. **Failure modes per type** — categorized root causes
6. **Template drift** — non-cipher types running cipher template (should be 0)

Good signals:
- `avg_table_accuracy >= 0.95` on cipher
- `ver_honesty_dist` has `PASS_TRUE` dominating, with some `FAIL_TRUE` (honest no) for partial data
- `vocab_fill_used` low (<30%) — only used as rescue
- `binary.matches_simple_truth == binary.solvable_simple` (gets all simple cases right)
- `equation.ran_cipher_template == 0` (no template misfire)
- `drift_flags == {}` for non-cipher types

Bad signals:
- `ver_honesty_dist` has lots of `PASS_LIE` — model learned to rubber-stamp VER
- `vocab_fill_used` == 100% — model rescues wrong decryptions with wrong words (v30 pattern)
- `binary.matches_simple_truth` much lower than `solvable_simple` — model can't handle cases
  the per-bit classifier can solve
- `equation.ran_cipher_template` > 50% — template contamination (v30 pattern)
- `drift_flags` non-empty — training data poisoned cross-type

## Ground-truth prompt matching

The `run_eval.py` runner saves full prompts in the JSON. For legacy dumps (like
`/tmp/compare_v30_v31.json` which only has `prompt_preview`), the analyzer falls back to
looking up the full prompt by `(type, expected_answer)` in `data/train.csv`. Collisions are rare
in the 60-sample benchmark.

## Local vs Kaggle score calibration

From the first run: v30 local 0.550 vs Kaggle 0.540; v31 local 0.667 vs Kaggle 0.660.
**60-sample benchmark tracks public score within ±0.01.** This means you can test a new adapter
locally and skip a Kaggle submission if the local delta is negative. Save the submissions for
genuine improvements.

## Extending

- **New puzzle type**: add `verifiers/<type>.py` with `verify(prompt, expected, trace, boxed)`,
  register in `verifiers/__init__.py`
- **New quality metric**: add to `analyze.py`'s `analyze_adapter` aggregation step
- **Failure mode change**: add to the type-specific `verify()` function's failure_mode classification
