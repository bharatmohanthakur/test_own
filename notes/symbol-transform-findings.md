# symbol_transform (CIPHER-DIGIT equation) — debug findings

## Puzzle structure (verified against train.csv)
- 823 CIPHER-DIGIT rows (out of 1555 equation rows)
- Each puzzle: 4-5 example pairs + 1 target
- LHS: 5-char string, position 2 is "operator", positions 0,1,3,4 are operand chars
- RHS: variable length 1-4 chars, encoded result
- Median 10 distinct non-op chars per puzzle (range 8-11) → digit bijection hypothesis
- Operator chars vary per line within same puzzle (per Donald: cosmetic/random)

## Solvers tried — all fail on real puzzles

### Approach 1: Permutation brute force
- Try all perms of operand chars over digits 0-9
- 5040 perms × 47 combos × 4-5 verify calls = millions of ops/puzzle
- Python: too slow (>30s per puzzle), killed
- No puzzles solved before timeout

### Approach 2: Z3 with Distinct, 6 ops, 4 orders, 2 fmts (48 combos)
- Assumed operator char cosmetic, one op per puzzle
- 0/10 puzzles solved (all UNSAT)
- Concludes: my 6-op set does NOT cover the actual operations

### Approach 3: Z3 with 12 ops (+ mul_add1, mul_sub1, addm1, add1, sub1, rsub1)
- Still 0/5 puzzles solved (all UNSAT)
- Reran with 5s timeout per combo, 96 total combos: still NO combo satisfies

### Approach 4: Literal operator hypothesis ('+' = add, '-' = sub, '*' = mul)
- Tested on 30 puzzles with standard arithmetic operators only
- **1/30 correct** — essentially luck, not a valid model

## Why none of these work

Donald Galliano's note mentions "4 pairings × 14 ops × 14 formats = 784 possible, top 47 cover 99%".
My enumeration has:
- 4 orders ✓ (AB_CD, BA_DC, AB_DC, BA_CD)
- 6-12 ops ✗ (missing ~2+ ops)
- 2 formats ✗ (missing 12 formats — need zpad3, zpad4, abs, neg, trunc, etc.)

I haven't mapped Donald's actual 14 ops and 14 formats. His full writeup is needed.

## Additional hypothesis worth testing later
1. **Leading zero handling**: operand1 with d1=0 decodes to 1-digit → affects result length. E.g., op1=03=3, op2=45 → 3*45=135 (3-char). Without leading zero stripping, operand=3 vs 03 ambiguity.

2. **Result bit-width truncation**: maybe result is mod 100 or mod 1000, explaining variable RHS lengths.

3. **Per-example operation selection**: operator char IS the op selector (but my attempt got 1/30, maybe needs better format search).

4. **Non-permutation mapping**: some puzzles have 11 distinct chars, forcing a non-injective map. Some chars may map to same digit.

5. **Mixed encoding**: maybe operands are 2-digit but result is encoded differently (e.g., hex, base-11, per-digit cipher with offset).

## Conclusion: defer to THK's Sunday drop
- 4 hours spent debugging, 0/30 correct
- 470 operations untried (out of 784 possible per Donald)
- THK publishes full solution Sunday Apr 12 UTC — his writeup will show the actual op set
- Until then, DON'T retrain v38 attempting this: it will not converge

## Alternative — LLM teacher distillation (skip CSP)
- Call Grok-4 or DeepSeek-R1 on each CIPHER-DIGIT row
- Filter by exact `\boxed{}` match against train.csv answer
- Expected hit rate: ~20-40% per Grok's binary benchmarks
- 823 × 0.30 = ~247 verified traces → usable as training data
- **Requires API access + ~$5-15 in credits**

## Files produced
- `data/symbol_transform_solver.py` (brute force, broken)
- `data/sym_z3_solver.py` (Z3 with Distinct, 0/30)
- `notes/symbol-transform-research.md` (from research agent)
- `notes/symbol-transform-findings.md` (this file)
