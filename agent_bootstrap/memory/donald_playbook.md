---
name: donald_playbook
description: Donald Galliano III reverse-engineered 100% of the dataset on Apr 2026 — 7 puzzle types with structured pipelines (TABLE/VER/CHECK/ANS), per-step verification, and template-specific reward shaping. Top of board claims 0.84 → 100% achievable.
type: reference
---

**Discussion**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688461
**Local copy**: notes/donald-galliano-playbook.md

Key findings (use this when designing training data, GRPO rewards, or analyzing failures):

1. **Sudoku-style verify-then-commit** is the universal philosophy. Every step has a VER/CHK that proves the constraint before moving on. Backtrack on failure, never guess forward.

2. **7 puzzle types**, not 6. equation_transform is actually 2 sub-types: SYMBOL-DIGIT (raw operator) and CIPHER-DIGIT (cipher-overlaid).

3. **Bit-serial / char-by-char execution** is the canonical fix for hallucination. Never let the model do parallel/whole-word/whole-blob ops — language prior hijacks it. Force one-at-a-time.

4. **VER must be independent of the answer path**. Don't recompute the answer two ways (same arithmetic weakness → confident wrong agreement). For gravity/unit_con: derive RATE from EX1 and EX2 separately, check |RATE-RATE2| < tolerance. For roman: round-trip re-parse the output string.

5. **VOCAB fill** for cipher: there's a fixed ~90-word vocabulary (character-verb-object patterns). With 6-8 example phrases you never cover all 26 letters → use vocab to fill gaps. **Concrete TODO**: mine train.csv to extract the vocab set.

6. **47-combo scan** for equation_transform: dominant combo is `BA_DC | add | rev` (13%), second is `BA_DC | mul | rev` (12%). 4 pairings × 14 ops × 14 formats = 784 possible, top 47 cover 99%. **Concrete TODO**: enumerate combos from train.csv to build the scan order.

7. **Format strictness**: gravity and unit_con require exactly `X.XX` format → wrong decimal places = 0. Missing `\boxed{}` = -12 (instant death).

8. **Reward shape for GRPO**:
   - Per-step partial credit on each labeled stage
   - Tiered VER honesty: lying YES (worst) > confused NO > honest NO > honest YES (best)
   - Champagne bonus on fully-correct (+5 superlinear)
   - Penalize contamination markers (other templates' language)
   - Penalize thrash markers ("hmm", "let me try", "actually")
   - Cipher-digit perfect trace = +33 (most layers of any factory)

**How to use**: When you generate training data, generate templated traces with hard-labeled steps (LEN/TABLE/VER/DECRYPT/CHECK/ANS/`\boxed{}`). When you design GRPO rewards, score each labeled step independently with tiered VER honesty.

**Why this matters**: Donald put 200+ hours into this and bowed out due to Kaggle GPU access. He says these templates are 100% solvable. Top of board is 0.84, our best is 0.69 — gap of 0.15 closeable by using these structured pipelines instead of generic CoT.
