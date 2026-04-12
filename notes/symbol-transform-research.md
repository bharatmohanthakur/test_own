# Symbol-Transform Research (2026-04-11)

## TL;DR

**Symbol_transform (cipher-digit equation_transform) is largely unsolved and likely unsolvable at scale via any pipeline approach I could engineer in a few hours.** My exhaustive Z3 CSP solver, my 4×14×14 = 784-combo brute-force enumerator (Donald-playbook op set), and Z3 with bijective `Distinct` constraint ALL solve essentially 0/30 real cipher-digit puzzles and <3% of pure symbol-digit puzzles. The dominant public belief (m4nocha 0/82, Kh0a 0/82, THK is the only known solver) is consistent with what I found: the rule space is larger, more exotic, or structurally different from what the playbook documents.

**Assessment: this is a partial dead end.** Getting to 0.90 via symbol_transform as a standalone route is NOT realistic with current public knowledge. Wait for THK's writeup Sunday April 12. In the meantime, hybrid LLM distillation (approach #3) is the only tractable path.

---

## Puzzle structure

Analyzed all 1555 `equation_transform` rows in `train.csv`:

| Sub-type | Count | Description |
|----------|-------|-------------|
| "pure symbol-digit" (all LHS chars are digits) | 553 | `34/44 = 1`, `76-68 = -91` |
| Mixed/cipher-digit (LHS contains non-digit symbols) | 1002 | `` `!*[{ = '"[` `` |

- LHS is always 5 chars; operator at position 2 in **99% of rows**.
- The operator char VARIES within a single puzzle's 3-5 examples in 717/732 cases (confirmed — so it is NOT semantic; it is cosmetic).
- Cipher-digit puzzles have 10-12 distinct non-op symbols → consistent with a **bijective 10-symbol → 10-digit map**.
- RHS length varies 1-4 chars. Query result is always encoded in the same cipher.
- **Unknowns per puzzle**: up to 10 symbol→digit assignments + 1 of ~784 (order × op × format) combos = roughly `10! × 784 ≈ 2.8B` raw search space, but modern SMT prunes this in ms per feasible combo.

## What I built and measured

All code in `/tmp/test_solver_full.py`, `/tmp/test_z3_v2.py`, `/tmp/synth_test.py`.

### Approach 1: Pure brute-force enumeration (symbol-digit)

- 4 pair orders (AB_CD, BA_DC, AB_DC, BA_CD) × 14 ops (add/sub/rsub/mul/cat/rcat + ±1 variants) × ~20 output formats (raw/rev/abs/zpad_k/rev_zpad_k/last_k/first_k/dsum/half/double/sq)
- Complexity: ~2000 combos/puzzle, <1 ms per puzzle
- **Result on pure symbol-digit: 15/553 = 2.7%** (no, not 99% as the playbook claims)
- Even ignoring the cipher overlay (which simplifies to bare symbol-digit), my enumerator found solutions for only 2-4% of real puzzles.

### Approach 2: Z3 CSP with `Distinct` bijective constraint (cipher-digit)

```python
d = {c: Int(...) for c in content_chars}   # 10 content chars
s.add(Distinct(list(d.values())))           # bijective to 0-9
# Enumerate (order, op, fmt) externally; add arithmetic constraints per example
for (lhs, rhs) in examples:
    a = d[lhs[ai]]*10 + d[lhs[aj]]
    b = d[lhs[bi]]*10 + d[lhs[bj]]
    res = opf(a, b)  # by enumerated op
    rhs_val = Sum(d[rhs[i]] * 10**(len(rhs)-1-i) for i in ...)
    s.add(res == rhs_val)
```

- **Validated on synthetic puzzle (BA_DC | add | rev): solved in 4.0s. Z3 machinery works.**
- **Result on first 30 real cipher-digit puzzles: 0/30 correct, ~10s/puzzle with 4×12×2 = 96 combos.**
- Z3 cannot even find a satisfying assignment for the 4 hand-inspected cipher-digit examples, meaning the true rule is NOT in {add, sub, rsub, mul, cat, rcat, ±1 variants} × {raw, rev}.

### Approach 3: Hand-decoding puzzle 1 (`` `!*[{ = '"[` ``)

- 10 non-op distinct symbols — perfect bijection with 10 digits.
- Brute-forced through 4 orders × 13 ops × 2 formats. Z3 reports `unsat` for every combo.
- **Implication: the rule structure is NOT (extract 2 two-digit operands, apply single op, format output).** It must involve something like per-digit transformation, bit operations, Caesar-shift on digits, asymmetric operand lengths, or completely different decoding.

## Why Donald's playbook claim is probably wrong

Donald's playbook claims:
- `BA_DC | add | rev` at 13%
- `BA_DC | mul | rev` at 12%
- 47 combos cover 99% of equation_transform distribution
- Cipher layer is "cosmetic" — crack cipher → bare symbol-digit

**My empirical data contradicts all 4 claims.** Running the full 4×14×14 = 784-combo scanner on 553 pure symbol-digit puzzles (where there is NO cipher overlay to crack):
- `('BA_DC', 'sub', 'rev')` solves 2 puzzles (not 13%)
- `('BA_DC', 'cat', 'rev_zpad1')` solves 2 puzzles
- Top frequency is 2-count combos, not 50-60 count as the 13%×553 = 72 figure would imply.
- Total coverage across all 784 combos + custom op-char-semantic rules: **15/553 = 2.7%, not 99%**.

Either Donald was hallucinating, the playbook summarizes a SYNTHETIC dataset not the actual competition data, or the real rule family is **fundamentally different** from concat/add/sub/mul of 2-digit operands.

## Top 3 implementation approaches

### Approach A: Z3 CSP + massive op library (best if tractable)

- **Sketch**: enumerate (input order, op, output format) externally; Z3 solves digit assignment per combo with `Distinct`.
- **Complexity**: `O(combos × Z3_check_time)`. At 500ms timeout × 200 combos = 100s/puzzle. Too slow for training data gen unless combo count is ~10.
- **Expected solve rate**: **0-5%** with current op library. Would need reverse-engineering 30-50+ additional exotic ops.
- **Effort**: 2-3 days to reverse-engineer from training CSV with pattern mining; could be blocked if ops are inherently per-symbol/non-arithmetic.
- **Dead-end-probability**: HIGH unless we discover the true rule family.

### Approach B: Brute-force with learned op prior (cheap)

- **Sketch**: Mine training data for ALL (input_xform, op, output_xform) combos that solve SOMETHING. Rank by frequency. Use top-K at inference.
- **Complexity**: preprocessing once, inference is 1-10ms/puzzle.
- **Expected solve rate**: bounded by how many puzzles our enumerator can solve ANY WAY (currently ~3%). Ceiling low.
- **Effort**: 4 hours.
- **Verdict**: run this to confirm the 3% ceiling and verify playbook claim is wrong BEFORE any distillation effort.

### Approach C: LLM distillation + rejection sampling (only approach with >10% ceiling)

- **Sketch**: Send each training row to DeepSeek-R1 or Grok-4 (or nemotron-latest) with prompt engineering. Parse their `\boxed{}` answer. Keep only rows where the LLM gets the training answer correct. Use THOSE as SFT data with the LLM's reasoning trace as the CoT.
- **Complexity**: API cost ~$50-200 for 1555 rows. Rate-limited to hours, not seconds.
- **Expected solve rate**:
  - Upper bound: Grok-4 / R1 ceiling on these puzzles. Unknown, but probably 20-40% given THK solves ~82% with a trained model.
  - Even 20% success on 1100 cipher-digit rows = 220 high-quality SFT examples per run.
- **Effort**: 1 day to script + sample + verify.
- **Killer benefit**: we don't need to understand the rule — we just need the LLM to produce a verifiable trace, and self-filter via exact-match on the training answer.

## Code skeleton for Approach C (recommended)

```python
import anthropic, csv, json
from pathlib import Path

# Use R1 or Grok-4 via OpenRouter; shown here with DeepSeek API
client = ... # DeepSeek client, key in memory/deepseek_key.md

ROWS = [r for r in csv.DictReader(open('data/train.csv'))
        if 'transformation rules is applied to equations' in r['prompt']]

SYSTEM = """You are solving a cryptarithmetic / cipher-digit puzzle. Each symbol
maps to a digit 0-9 (bijective within a puzzle). The middle character of each
LHS is the operator, which may or may not be semantically meaningful.

Your task: figure out the rule from the 3-5 examples, verify it, then apply to
the query. Show your work step-by-step. End with \\boxed{answer}."""

def distill_row(row):
    resp = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": row['prompt']}],
        max_tokens=8000,
        temperature=0.0,
    )
    trace = resp.choices[0].message.content
    # Extract \boxed{...}
    import re
    m = re.search(r'\\boxed\{([^}]+)\}', trace)
    if not m: return None
    predicted = m.group(1).strip()
    # Case-insensitive exact match (competition metric)
    if predicted.lower() == row['answer'].strip().lower():
        return {
            "messages": [
                {"role": "user", "content": row['prompt'] +
                 '\nPlease put your final answer inside `\\boxed{}`.'},
                {"role": "assistant", "content": f"<think>\n{trace}\n</think>\n\n\\boxed{{{predicted}}}"}
            ]
        }
    return None  # reject

out_path = Path('data/symbol_transform_distilled.jsonl')
with out_path.open('w') as f:
    for r in ROWS:
        result = distill_row(r)
        if result:
            f.write(json.dumps(result) + '\n')
            f.flush()
```

Budget: at ~$0.08 per 8K-token reasoner call × 1555 rows ≈ $124. Can do parallel batches. Filter yield expected 15-30%.

## Is 0.90 realistic via symbol_transform?

**No, not as a standalone route.**

Evidence:
- m4nocha 0/82, Kh0a 0/82 in public. Only THK (1st, 0.85) claims to solve it — writeup Sunday.
- Standard cryptarithmetic (Z3 + standard ops + bijective mapping) FAILS on real puzzles. The rule family is exotic.
- My brute-force + Z3 approaches cap at 2-3% on pure symbol-digit, which is the "easy half."
- 7680-token generation budget means even if we figured out the rule, the model's trace would have to crank through symbol-decode → op-search → rehydrate in ~1500 tokens per puzzle. Expensive but not impossible.
- Symbol_transform is ~10% of the test set. A perfect solver gains +0.10. A 30%-yield distillation buys +0.03. A failed approach buys +0.00.

**The only tractable ~0.90 path right now**:
1. Wait 24-48 hours for THK's Sunday writeup. If he reveals the rule family, we're gold.
2. In parallel, run Approach C distillation with DeepSeek-R1 + Grok-4 on all 1555 equation_transform rows. Best case: we get 300-500 verified SFT examples that teach the model to reason about these puzzles, even if we don't understand them ourselves.
3. Combine distilled equation_transform data with v22 SFT base + the puzzle-KD strategy in `notes/research-apr9.md`.

## Files

- `/tmp/test_solver_full.py` — Z3 cipher-digit solver (0/30 on real data)
- `/tmp/test_z3_v2.py` — earlier Z3 variant without Distinct
- `/tmp/synth_test.py` — synthetic puzzle validator (Z3 solves in 4s, proves the machinery is correct)
- `/Users/bharat/Downloads/kaggle/notes/donald-galliano-playbook.md` — the playbook (likely wrong about op frequencies)
- `/Users/bharat/Downloads/kaggle/notes/research-2026-04-11-to-090.md` — general 0.90 plan
