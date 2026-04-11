# Validator Report — adapter `v34c`
Samples: 120

## Layer 1 — Accuracy

| type     | acc | n | chain_score |
|----------|-----|---|-------------|
| binary   | 0/20 (0%) | 20 | 0.00 |
| cipher   | 0/20 (0%) | 20 | 0.00 |
| equation | 0/20 (0%) | 20 | 0.12 |
| gravity  | 0/20 (0%) | 20 | 0.00 |
| roman    | 0/20 (0%) | 20 | 0.00 |
| unit     | 0/20 (0%) | 20 | 0.00 |

**Total accuracy: 0.000**

## Layer 2 — Reasoning chain (per-step pass rates)

### binary
  ✗ identify_type          0/20 (0%)
  ✗ parse_examples         0/20 (0%)
  ✗ per_bit_tried          0/20 (0%)
  ✗ rule_derived           0/20 (0%)
  ✗ consistency_ok         0/20 (0%)
  ✗ rule_applied           0/20 (0%)
  ✗ valid_binary           0/20 (0%)
  ✗ boxed_format           0/20 (0%)
  first_failure breakdown: {'identify_type': 20}

### cipher
  ✗ identify_type          0/20 (0%)
  ✗ parse_examples         0/20 (0%)
  ✗ char_align             0/20 (0%)
  ✗ derive_table           0/20 (0%)
  ✗ table_accuracy         0/20 (0%)
  ✗ apply_table            0/20 (0%)
  ✗ valid_english          0/20 (0%)
  ✗ boxed_format           0/20 (0%)
  first_failure breakdown: {'identify_type': 20}

### equation
  ✗ identify_type          0/20 (0%)
  ✗ parse_operands         0/20 (0%)
  ✗ detect_subtype         0/20 (0%)
  ✗ arithmetic_attempted   0/20 (0%)
  ✓ NOT_cipher_template    20/20 (100%)
  ✗ rule_committed         0/20 (0%)
  ✗ target_processed       0/20 (0%)
  ✗ boxed_format           0/20 (0%)
  first_failure breakdown: {'identify_type': 20}

### gravity
  ✗ identify_type          0/20 (0%)
  ✗ parse_observations     0/20 (0%)
  ✗ g_computed             0/20 (0%)
  ✗ g_consistency_check    0/20 (0%)
  ✗ applied_to_target      0/20 (0%)
  ✗ boxed_format_2dec      0/20 (0%)
  first_failure breakdown: {'identify_type': 20}

### roman
  ✗ identify_type          0/20 (0%)
  ✗ direction_detected     0/20 (0%)
  ✗ conversion_logic       0/20 (0%)
  ✗ applied_to_target      0/20 (0%)
  ✗ boxed_format           0/20 (0%)
  first_failure breakdown: {'identify_type': 20}

### unit
  ✗ identify_type          0/20 (0%)
  ✗ parse_examples         0/20 (0%)
  ✗ slope_computed         0/20 (0%)
  ✗ applied_to_target      0/20 (0%)
  ✗ boxed_format_2dec      0/20 (0%)
  first_failure breakdown: {'identify_type': 20}

## Layer 3 — Behavioral fingerprint (avg per type)

### binary
  template_label_count             0
  reasoning_word_count             0
  numeric_density                  0.0
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### cipher
  template_label_count             0
  reasoning_word_count             0
  numeric_density                  0.0
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### equation
  template_label_count             0
  reasoning_word_count             0
  numeric_density                  0.0
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### gravity
  template_label_count             0
  reasoning_word_count             0
  numeric_density                  0.0
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### roman
  template_label_count             0
  reasoning_word_count             0
  numeric_density                  0.0
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

### unit
  template_label_count             0
  reasoning_word_count             0
  numeric_density                  0.0
  distinct_rule_attempts           0
  self_correction_count            0
  explicit_verification_count      0
  confidence_ratio                 0.0

## Layer 4 — Confidence calibration

  accuracy:                  0.000
  mean_stated_confidence:    0.500
  expected_calibration_error: 0.500
  overclaim_rate:            0.000  (wrong + claims confident)
  underclaim_rate:           0.000  (correct + sounds uncertain)
  conf_when_correct:         0.000
  conf_when_wrong:           0.500

## Layer 5 — Attribution (vs `v31`)

  improvements: 0
  regressions:  78
  both_right:   0
  both_wrong:   42

  improvements_by_type: {}
  regressions_by_type:  {'binary': 1, 'cipher': 17, 'gravity': 20, 'roman': 20, 'unit': 20}
  regressions_by_source:  {'v34_binary_solver': 1, 'v22_base': 77}

  **v31->v34c: net -78 (0 up, 78 down) | -20 gravity, -20 roman, -20 unit, -17 cipher, -1 binary | regressions attributed: 77 via v22_base, 1 via v34_binary_solver**

## Decision Gate

### Hard gates (any failure → ABORT)
  ✗ cipher_accuracy >= 0.80
  ✓ no_template_drift
  ✓ format_pct == 100%
  ✗ regressions < improvements

### Soft gates
  ✗ total_acc >= 0.7
  ✗ binary_acc >= 0.5
  ✓ overclaim_rate <= 0.3
  ✗ ECE <= 0.1

### Verdict: **ABORT — hard gate failed**
