# Tracking Report — 60 samples

## Tier 1 — Accuracy per type

| type     | v30 | v31 |
|----------|--------|--------|
| binary   | 3/10 (30%) | 1/10 (10%) |
| cipher   | 0/10 (0%) | 9/10 (90%) |
| equation | 0/10 (0%) | 0/10 (0%) |
| gravity  | 10/10 (100%) | 10/10 (100%) |
| roman    | 10/10 (100%) | 10/10 (100%) |
| unit     | 10/10 (100%) | 10/10 (100%) |

**Totals:**
- v30: 33/60 = 0.550
- v31: 40/60 = 0.667

## Tier 3 — Cipher step-derivation quality

### v30
- cipher samples: 10
- table emitted: 10/10
- avg table accuracy (when emitted): 0.79
- vocab fill used: 10/10
- VER honesty distribution: {'PASS_LIE': 2, 'PASS_TRUE': 8}

### v31
- cipher samples: 10
- table emitted: 10/10
- avg table accuracy (when emitted): 0.98
- vocab fill used: 0/10
- VER honesty distribution: {'NONE': 10}

## Binary difficulty breakdown

### v30
- difficulty mix: {'SIMPLE': 6, 'HARD': 4}
- solvable with simple gates (constant/identity/NOT/2-input/majority): 6/10
- answer matches derived truth: 2/10

### v31
- difficulty mix: {'SIMPLE': 6, 'HARD': 4}
- solvable with simple gates (constant/identity/NOT/2-input/majority): 6/10
- answer matches derived truth: 0/10

## Equation: subtype + cipher-template misfire

### v30
- subtype mix: {'SYMBOL-DIGIT': 7, 'CIPHER-DIGIT': 3}
- ran cipher template on equation: 9/10
- attempted arithmetic: 5/10

### v31
- subtype mix: {'SYMBOL-DIGIT': 7, 'CIPHER-DIGIT': 3}
- ran cipher template on equation: 0/10
- attempted arithmetic: 4/10

## Failure modes (per type, per adapter)

### binary
- v30: {'wrong_2gate': 4, 'complex_gates': 3, 'correct': 3}
- v31: {'wrong_2gate': 5, 'complex_gates': 3, 'correct': 1, 'lazy_constant_0': 1}

### cipher
- v30: {'template_with_lying_ver': 2, 'hallucinated_vocab': 6, 'template_with_wrong_table': 1, 'no_boxed': 1}
- v31: {'correct': 9, 'other_wrong': 1}

### equation
- v30: {'copied_input': 3, 'cipher_template_misfire': 6, 'arithmetic_wrong': 1}
- v31: {'no_arithmetic_attempted': 6, 'arithmetic_wrong': 4}

### gravity
- v30: {'correct': 10}
- v31: {'correct': 10}

### roman
- v30: {'correct': 10}
- v31: {'correct': 10}

### unit
- v30: {'correct': 10}
- v31: {'correct': 10}

## Template drift (non-cipher types running cipher template)

- v30: none
- v31: none
