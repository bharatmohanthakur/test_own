# Validator Report — adapter `v31`
Samples: 60

## Layer 1 — Accuracy

| type     | acc | n | chain_score |
|----------|-----|---|-------------|
| binary   | 1/10 (10%) | 10 | 0.57 |
| cipher   | 9/10 (90%) | 10 | 0.99 |
| equation | 0/10 (0%) | 10 | 0.95 |
| gravity  | 10/10 (100%) | 10 | 0.97 |
| roman    | 10/10 (100%) | 10 | 1.00 |
| unit     | 10/10 (100%) | 10 | 0.96 |

**Total accuracy: 0.667**

## Layer 2 — Reasoning chain (per-step pass rates)

### binary
  ✓ identify_type          10/10 (100%)
  ✗ parse_examples         3/10 (30%)
  ✗ per_bit_tried          0/10 (0%)
  ✓ rule_derived           10/10 (100%)
  ✗ consistency_ok         0/10 (0%)
  ✗ rule_applied           3/10 (30%)
  ✓ valid_binary           10/10 (100%)
  ✓ boxed_format           10/10 (100%)
  first_failure breakdown: {'parse_examples': 7, 'per_bit_tried': 3}

### cipher
  ✓ identify_type          10/10 (100%)
  ✓ parse_examples         10/10 (100%)
  ✓ char_align             10/10 (100%)
  ✓ derive_table           10/10 (100%)
  ✓ table_accuracy         10/10 (100%)
  ✓ apply_table            10/10 (100%)
  ✓ valid_english          9/10 (90%)
  ✓ boxed_format           10/10 (100%)
  first_failure breakdown: {'valid_english': 1}

### equation
  ✓ identify_type          10/10 (100%)
  ~ parse_operands         8/10 (80%)
  ✓ detect_subtype         10/10 (100%)
  ✓ arithmetic_attempted   10/10 (100%)
  ✓ NOT_cipher_template    10/10 (100%)
  ✓ rule_committed         10/10 (100%)
  ~ target_processed       8/10 (80%)
  ✓ boxed_format           10/10 (100%)
  first_failure breakdown: {'parse_operands': 2}

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
  template_label_count             0
  reasoning_word_count             117
  numeric_density                  4.34
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### cipher
  template_label_count             0
  reasoning_word_count             252.4
  numeric_density                  0.26
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### equation
  template_label_count             0
  reasoning_word_count             116.1
  numeric_density                  3.06
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### gravity
  template_label_count             0
  reasoning_word_count             165.8
  numeric_density                  8.05
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### roman
  template_label_count             0
  reasoning_word_count             89.8
  numeric_density                  2.59
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      1
  confidence_ratio                 0.0

### unit
  template_label_count             0
  reasoning_word_count             92.4
  numeric_density                  5.81
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

## Layer 4 — Confidence calibration

  accuracy:                  0.667
  mean_stated_confidence:    0.700
  expected_calibration_error: 0.033
  overclaim_rate:            0.000  (wrong + claims confident)
  underclaim_rate:           0.000  (correct + sounds uncertain)
  conf_when_correct:         0.700
  conf_when_wrong:           0.700

## Layer 5 — Attribution (vs `v30`)

  improvements: 9
  regressions:  2
  both_right:   31
  both_wrong:   18

  improvements_by_type: {'cipher': 9}
  regressions_by_type:  {'binary': 2}
  improvements_by_source: {'v22_base': 9}
  regressions_by_source:  {'v34_binary_solver': 2}

  **v30->v31: net +7 (9 up, 2 down) | +9 cipher | -2 binary | improvements attributed: 9 via v22_base | regressions attributed: 2 via v34_binary_solver**

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
