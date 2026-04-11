# Validator Report — adapter `v34`
Samples: 120

## Layer 1 — Accuracy

| type     | acc | n | chain_score |
|----------|-----|---|-------------|
| binary   | 5/20 (25%) | 20 | 1.00 |
| cipher   | 16/20 (80%) | 20 | 0.97 |
| equation | 0/20 (0%) | 20 | 0.99 |
| gravity  | 20/20 (100%) | 20 | 1.00 |
| roman    | 20/20 (100%) | 20 | 1.00 |
| unit     | 20/20 (100%) | 20 | 1.00 |

**Total accuracy: 0.675**

## Layer 2 — Reasoning chain (per-step pass rates)

### binary
  ✓ identify_type          20/20 (100%)
  ✓ parse_examples         20/20 (100%)
  ✓ per_bit_tried          20/20 (100%)
  ✓ rule_derived           20/20 (100%)
  ✓ consistency_ok         20/20 (100%)
  ✓ rule_applied           20/20 (100%)
  ✓ valid_binary           20/20 (100%)
  ✓ boxed_format           20/20 (100%)

### cipher
  ✓ identify_type          20/20 (100%)
  ✓ parse_examples         20/20 (100%)
  ✓ char_align             20/20 (100%)
  ✓ derive_table           20/20 (100%)
  ✓ table_accuracy         18/20 (90%)
  ✓ apply_table            20/20 (100%)
  ✓ valid_english          18/20 (90%)
  ✓ boxed_format           20/20 (100%)
  first_failure breakdown: {'valid_english': 1, 'table_accuracy': 2}

### equation
  ✓ identify_type          20/20 (100%)
  ✓ parse_operands         20/20 (100%)
  ✓ detect_subtype         20/20 (100%)
  ✓ arithmetic_attempted   20/20 (100%)
  ✓ NOT_cipher_template    20/20 (100%)
  ✓ rule_committed         20/20 (100%)
  ✓ target_processed       19/20 (95%)
  ✓ boxed_format           20/20 (100%)
  first_failure breakdown: {'target_processed': 1}

### gravity
  ✓ identify_type          20/20 (100%)
  ✓ parse_observations     20/20 (100%)
  ✓ g_computed             20/20 (100%)
  ✓ g_consistency_check    20/20 (100%)
  ✓ applied_to_target      20/20 (100%)
  ✓ boxed_format_2dec      20/20 (100%)

### roman
  ✓ identify_type          20/20 (100%)
  ✓ direction_detected     20/20 (100%)
  ✓ conversion_logic       20/20 (100%)
  ✓ applied_to_target      20/20 (100%)
  ✓ boxed_format           20/20 (100%)

### unit
  ✓ identify_type          20/20 (100%)
  ✓ parse_examples         20/20 (100%)
  ✓ slope_computed         20/20 (100%)
  ✓ applied_to_target      20/20 (100%)
  ✓ boxed_format_2dec      20/20 (100%)

## Layer 3 — Behavioral fingerprint (avg per type)

### binary
  template_label_count             0
  reasoning_word_count             190.85
  numeric_density                  5.23
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      1
  confidence_ratio                 0.0

### cipher
  template_label_count             0
  reasoning_word_count             249.55
  numeric_density                  0.26
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### equation
  template_label_count             0
  reasoning_word_count             115.95
  numeric_density                  1.62
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### gravity
  template_label_count             0
  reasoning_word_count             171.8
  numeric_density                  8.09
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### roman
  template_label_count             0
  reasoning_word_count             92.5
  numeric_density                  2.78
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      1
  confidence_ratio                 0.0

### unit
  template_label_count             0
  reasoning_word_count             88.8
  numeric_density                  5.64
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

## Layer 4 — Confidence calibration

  accuracy:                  0.675
  mean_stated_confidence:    0.717
  expected_calibration_error: 0.042
  overclaim_rate:            0.125  (wrong + claims confident)
  underclaim_rate:           0.000  (correct + sounds uncertain)
  conf_when_correct:         0.706
  conf_when_wrong:           0.738

## Layer 5 — Attribution (vs `v31`)

  improvements: 6
  regressions:  3
  both_right:   75
  both_wrong:   36

  improvements_by_type: {'binary': 4, 'cipher': 2}
  regressions_by_type:  {'cipher': 3}
  improvements_by_source: {'v34_binary_solver': 4, 'v22_base': 2}
  regressions_by_source:  {'v22_base': 3}

  **v31->v34: net +3 (6 up, 3 down) | +4 binary, +2 cipher | -3 cipher | improvements attributed: 4 via v34_binary_solver, 2 via v22_base | regressions attributed: 3 via v22_base**

## Decision Gate

### Hard gates (any failure → ABORT)
  ✓ cipher_accuracy >= 0.80
  ✓ no_template_drift
  ✓ format_pct == 100%
  ✓ regressions < improvements

### Soft gates
  ✗ total_acc >= 0.7
  ✗ binary_acc >= 0.5
  ✓ overclaim_rate <= 0.3
  ✓ ECE <= 0.1

### Verdict: **MARGINAL — review carefully**
