# Validator Report — adapter `v34`
Samples: 60

## Layer 1 — Accuracy

| type     | acc | n | chain_score |
|----------|-----|---|-------------|
| binary   | 3/10 (30%) | 10 | 1.00 |
| cipher   | 7/10 (70%) | 10 | 0.95 |
| equation | 0/10 (0%) | 10 | 0.99 |
| gravity  | 10/10 (100%) | 10 | 1.00 |
| roman    | 10/10 (100%) | 10 | 1.00 |
| unit     | 10/10 (100%) | 10 | 1.00 |

**Total accuracy: 0.667**

## Layer 2 — Reasoning chain (per-step pass rates)

### binary
  ✓ identify_type          10/10 (100%)
  ✓ parse_examples         10/10 (100%)
  ✓ per_bit_tried          10/10 (100%)
  ✓ rule_derived           10/10 (100%)
  ✓ consistency_ok         10/10 (100%)
  ✓ rule_applied           10/10 (100%)
  ✓ valid_binary           10/10 (100%)
  ✓ boxed_format           10/10 (100%)

### cipher
  ✓ identify_type          10/10 (100%)
  ✓ parse_examples         10/10 (100%)
  ✓ char_align             10/10 (100%)
  ✓ derive_table           10/10 (100%)
  ~ table_accuracy         8/10 (80%)
  ✓ apply_table            10/10 (100%)
  ~ valid_english          8/10 (80%)
  ✓ boxed_format           10/10 (100%)
  first_failure breakdown: {'table_accuracy': 2, 'valid_english': 1}

### equation
  ✓ identify_type          10/10 (100%)
  ✓ parse_operands         10/10 (100%)
  ✓ detect_subtype         10/10 (100%)
  ✓ arithmetic_attempted   10/10 (100%)
  ✓ NOT_cipher_template    10/10 (100%)
  ✓ rule_committed         10/10 (100%)
  ✓ target_processed       9/10 (90%)
  ✓ boxed_format           10/10 (100%)
  first_failure breakdown: {'target_processed': 1}

### gravity
  ✓ identify_type          10/10 (100%)
  ✓ parse_observations     10/10 (100%)
  ✓ g_computed             10/10 (100%)
  ✓ g_consistency_check    10/10 (100%)
  ✓ applied_to_target      10/10 (100%)
  ✓ boxed_format_2dec      10/10 (100%)

### roman
  ✓ identify_type          10/10 (100%)
  ✓ direction_detected     10/10 (100%)
  ✓ conversion_logic       10/10 (100%)
  ✓ applied_to_target      10/10 (100%)
  ✓ boxed_format           10/10 (100%)

### unit
  ✓ identify_type          10/10 (100%)
  ✓ parse_examples         10/10 (100%)
  ✓ slope_computed         10/10 (100%)
  ✓ applied_to_target      10/10 (100%)
  ✓ boxed_format_2dec      10/10 (100%)

## Layer 3 — Behavioral fingerprint (avg per type)

### binary
  template_label_count             0
  reasoning_word_count             193.4
  numeric_density                  5.29
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      1
  confidence_ratio                 0.0

### cipher
  template_label_count             0
  reasoning_word_count             252.6
  numeric_density                  0.26
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### equation
  template_label_count             0
  reasoning_word_count             117.4
  numeric_density                  1.84
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### gravity
  template_label_count             0
  reasoning_word_count             173
  numeric_density                  8.12
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### roman
  template_label_count             0
  reasoning_word_count             94
  numeric_density                  2.9
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      1
  confidence_ratio                 0.0

### unit
  template_label_count             0
  reasoning_word_count             87.6
  numeric_density                  5.59
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

## Layer 4 — Confidence calibration

  accuracy:                  0.667
  mean_stated_confidence:    0.717
  expected_calibration_error: 0.050
  overclaim_rate:            0.117  (wrong + claims confident)
  underclaim_rate:           0.000  (correct + sounds uncertain)
  conf_when_correct:         0.707
  conf_when_wrong:           0.735

## Layer 5 — Attribution (vs `v31`)

  improvements: 2
  regressions:  1
  both_right:   38
  both_wrong:   19

  improvements_by_type: {'binary': 2}
  regressions_by_type:  {'cipher': 1}
  improvements_by_source: {'v34_binary_solver': 2}
  regressions_by_source:  {'v22_base': 1}

  **v31->v34: net +1 (2 up, 1 down) | +2 binary | -1 cipher | improvements attributed: 2 via v34_binary_solver | regressions attributed: 1 via v22_base**

## Decision Gate

### Hard gates (any failure → ABORT)
  ✗ cipher_accuracy >= 0.80
  ✓ no_template_drift
  ✓ format_pct == 100%
  ✓ regressions < improvements

### Soft gates
  ✗ total_acc >= 0.7
  ✗ binary_acc >= 0.5
  ✓ overclaim_rate <= 0.3
  ✓ ECE <= 0.1

### Verdict: **ABORT — hard gate failed**
