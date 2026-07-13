---
name: v30 vs v31 inference comparison findings
description: Concrete per-type findings from batched vLLM comparison of v30 (0.54) and v31 (0.66) on 60 train.csv prompts. Cipher is the whole regression.
type: project
---

# v30 vs v31 comparative inference (RunPod, 2026-04-09)

60 train prompts (10/type × 6 types). vLLM FLASHINFER backend, temp=0.0, top_p=1.0, max_tokens=7680, max_model_len=8192, LoRARequest for each adapter. Matches Kaggle eval exactly.

## Headline numbers
| type     | v31 (0.66) | v30 (0.54) | delta |
|----------|------------|------------|-------|
| gravity  | 10/10      | 10/10      |  0.00 |
| unit     | 10/10      | 10/10      |  0.00 |
| roman    | 10/10      | 10/10      |  0.00 |
| cipher   | 9/10       | **0/10**   | −0.90 |
| binary   | 1/10       | 3/10       | +0.20 |
| equation | 0/10       | 0/10       |  0.00 |
| TOTAL    | 40/60      | 33/60      | −0.117 |

**The entire v30 regression is cipher.** Binary actually improved (+2). Gravity/unit/roman unchanged (both saturated). Equation_transform is 0/10 on both adapters.

## Cipher pathology (v30)
Every v30 cipher trace follows the Donald labeled template:
```
LEN → TABLE → VER → DECRYPT → VOCAB fill → CHECK → ANS
```

What actually happens:
1. **TABLE is wrong.** v30 emits a substitution table but many entries are hallucinated (e.g., `b→p` when correct is `b→c`). It learned to emit the label, not derive the mapping from examples.
2. **VER PASS is a lie.** 10/10 v30 cipher traces say `VER PASS` even though the table is clearly wrong. The template's training examples all had VER PASS (because they were correct), so the model learned VER is ceremonial.
3. **VOCAB fill creates hallucinations.** Wrong decryptions get snapped to valid 77-word vocab entries, producing answers like "teacher potion the potions for" instead of "teacher chases the colorful key".
4. First word correct 7/10, last word correct 5/10 — partial competence that degrades across the sentence because vocab fill compounds errors.

Compare v31: free-form CoT listing example-by-example letter mapping, builds complete table, decrypts word-by-word, solves 9/10. v31 has **0 Donald labels** on cipher; v30 has **3.8 per trace**.

## Binary (both terrible)
Neither adapter can solve most binary problems. Donald's template only covers CONSTANT/IDENTITY/NOT/2-input cases. v30 gets:
- Row 2 (all-1s constant): both OK
- Row 3 (identity bit pattern): v30 OK, v31 NO
- Row 6 (constant 00000001): v30 OK, v31 NO

All 10 rows fall into 3+ input gate space where neither model has machinery. Binary is 10-15% of the total score gap — biggest growth headroom besides equation.

## Equation_transform (both 0/10)
v30 mis-categorizes as a cipher problem — runs LEN/TABLE/VER/CAT and just copies the input verbatim. v31 free-forms, claims to identify the rule, but just guesses. Neither model does the 47-combo arithmetic scan Donald described. Both produce wrong answers with zero useful reasoning. This is the **single biggest unlocked type** (~17% of rows).

## Donald label density (avg per trace, by type)
| type     | v31 | v30 |
|----------|-----|-----|
| binary   | 0.0 | 3.0 |
| cipher   | 1.0 | 3.8 |
| equation | 0.0 | 5.1 |
| gravity  | 0.0 | 6.0 |
| roman    | 0.0 | 1.0 |
| unit     | 0.0 | 6.0 |

v31 uses labels only sparingly on cipher (incidental). v30 heavily uses labels on equation and gravity/unit but labels don't help those types — they already saturate with free-form (v31 gets 10/10 with zero labels on gravity/unit).

## Hallucination markers
Only v31 binary has them (2.0/trace: "I identify", "after analyzing"). v30 eliminated these via the labeled template — **but replaced them with a new failure mode: confident lying (VER PASS)**.

## Key lesson for data quality
The v29 templated training data had three structural flaws:
1. **Every example had VER PASS** → model learned VER is a rubber stamp
2. **TABLE was pre-computed in example traces** → model learned to emit, not derive
3. **VOCAB fill was a "repair" step on already-correct decryptions** → model learned to repair wrong decryptions with vocab substitutions

Fix direction: training data must include **negative examples** where early steps are wrong and VER catches it (honest NO, then retry). Without failure cases, the model never learns that VER is a real check.
