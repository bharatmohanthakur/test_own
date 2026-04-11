# Validator Report — adapter `v30`
Samples: 60

## Layer 1 — Accuracy

| type     | acc | n | chain_score |
|----------|-----|---|-------------|
| binary   | 3/10 (30%) | 10 | 0.74 |
| cipher   | 0/10 (0%) | 10 | 0.86 |
| equation | 0/10 (0%) | 10 | 0.72 |
| gravity  | 10/10 (100%) | 10 | 0.97 |
| roman    | 10/10 (100%) | 10 | 1.00 |
| unit     | 10/10 (100%) | 10 | 0.96 |

**Total accuracy: 0.550**

## Layer 2 — Reasoning chain (per-step pass rates)

### binary
  ✓ identify_type          10/10 (100%)
  ✗ parse_examples         3/10 (30%)
  ✓ per_bit_tried          10/10 (100%)
  ✓ rule_derived           10/10 (100%)
  ✗ consistency_ok         3/10 (30%)
  ✗ rule_applied           3/10 (30%)
  ✓ valid_binary           10/10 (100%)
  ✓ boxed_format           10/10 (100%)
  first_failure breakdown: {'parse_examples': 7, 'consistency_ok': 2}

### cipher
  ✓ identify_type          10/10 (100%)
  ✓ parse_examples         10/10 (100%)
  ✓ char_align             10/10 (100%)
  ✓ derive_table           10/10 (100%)
  ✗ table_accuracy         3/10 (30%)
  ✓ apply_table            9/10 (90%)
  ~ valid_english          8/10 (80%)
  ✓ boxed_format           9/10 (90%)
  first_failure breakdown: {'table_accuracy': 7, 'valid_english': 1}

### equation
  ✓ identify_type          10/10 (100%)
  ~ parse_operands         8/10 (80%)
  ✓ detect_subtype         10/10 (100%)
  ✓ arithmetic_attempted   10/10 (100%)
  ✗ NOT_cipher_template    1/10 (10%)
  ✗ rule_committed         1/10 (10%)
  ~ target_processed       8/10 (80%)
  ✓ boxed_format           10/10 (100%)
  first_failure breakdown: {'NOT_cipher_template': 7, 'parse_operands': 2}

### gravity
  ✓ identify_type          10/10 (100%)
  ✓ parse_observations     9/10 (90%)
  ✓ g_computed             10/10 (100%)
  ✓ g_consistency_check    10/10 (100%)
  ✓ applied_to_target      9/10 (90%)
  ✓ boxed_format_2dec      10/10 (100%)
  first_failure breakdown: {'parse_observations': 1}

### roman
  ✓ identify_type          10/10 (100%)
  ✓ direction_detected     10/10 (100%)
  ✓ conversion_logic       10/10 (100%)
  ✓ applied_to_target      10/10 (100%)
  ✓ boxed_format           10/10 (100%)

### unit
  ✓ identify_type          10/10 (100%)
  ✓ parse_examples         9/10 (90%)
  ✓ slope_computed         10/10 (100%)
  ✓ applied_to_target      9/10 (90%)
  ✓ boxed_format_2dec      10/10 (100%)
  first_failure breakdown: {'parse_examples': 1}

## Layer 3 — Behavioral fingerprint (avg per type)

### binary
  template_label_count             4
  reasoning_word_count             153.7
  numeric_density                  5.97
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### cipher
  template_label_count             3.8
  reasoning_word_count             115
  numeric_density                  0.83
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### equation
  template_label_count             6.9
  reasoning_word_count             132.9
  numeric_density                  5.45
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### gravity
  template_label_count             6
  reasoning_word_count             88
  numeric_density                  5.5
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### roman
  template_label_count             6
  reasoning_word_count             97.5
  numeric_density                  2.76
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### unit
  template_label_count             6
  reasoning_word_count             69
  numeric_density                  5.36
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

## Layer 4 — Confidence calibration

  accuracy:                  0.550
  mean_stated_confidence:    0.752
  expected_calibration_error: 0.202
  overclaim_rate:            0.417  (wrong + claims confident)
  underclaim_rate:           0.000  (correct + sounds uncertain)
  conf_when_correct:         0.755
  conf_when_wrong:           0.750

## Layer 5 — Attribution (vs `v31`)

  improvements: 2
  regressions:  9
  both_right:   31
  both_wrong:   18

  improvements_by_type: {'binary': 2}
  regressions_by_type:  {'cipher': 9}
  improvements_by_source: {'v34_binary_solver': 2}
  regressions_by_source:  {'v22_base': 9}

  **v31->v30: net -7 (2 up, 9 down) | +2 binary | -9 cipher | improvements attributed: 2 via v34_binary_solver | regressions attributed: 9 via v22_base**

## Decision Gate

### Hard gates (any failure → ABORT)
  ✗ cipher_accuracy >= 0.80
  ✓ no_template_drift
  ✓ format_pct == 100%
  ✗ regressions < improvements

### Soft gates
  ✗ total_acc >= 0.7
  ✗ binary_acc >= 0.5
  ✗ overclaim_rate <= 0.3
  ✗ ECE <= 0.1

### Verdict: **ABORT — hard gate failed**
