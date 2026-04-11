# Donald Galliano III: 100% Solve Rate Reverse-Engineering Playbook

**Source**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688461
**Posted**: ~April 2026, 48 upvotes, 23 comments
**Author**: Donald Galliano III (bowed out due to Kaggle GPU access issues, opened up his playbook)
**Why huge**: He says all 6 (actually 7) puzzle types are 100% solvable with structured pipelines. He put 200+ hours into reverse engineering the dataset generators.
**Status**: Apr 7, 2026 — top of leaderboard 0.84, he claims 100% is achievable with this playbook.

---

## CORE PHILOSOPHY (applies to ALL types)

> **Sudoku-style reasoning**: verify the math at each step, lock what you can prove, and if you hit a contradiction later, backtrack to the last verified step and try the next branch. **Never guess forward hoping it works.** Every VER step, every CHK, every false-positive rejection in the scan is the same principle: prove each constraint before committing.

Common reward shape across every category (per-step GRPO):
- Partial credit at every layer (no single pass/fail gate until the very end)
- Tiered VER honesty: lying YES (claimed match but wrong) is the worst, honest NO (caught own error) scores partial, confused NO scores worse than honest NO
- "Champagne bonus" for fully correct outputs (superlinear, +5 flat)
- Heavy negative weights for **contamination markers** (language from other templates leaking in: B0:, MAP:, LOCK:, RATE:, DECOMPOSE, brackets/semicolons in pure-prose templates, etc.)
- Heavy negatives for **thrash markers**: "let me try", "hmm", "actually" — model must commit and execute, not spiral
- Missing `\boxed{}` is instant death (-12 in unit_con)

---

## 1. BINARY (bit_manipulation)

8-bit string in, 8-bit string out. Each output bit is an **independent boolean function** of the input bits. **Solve 8 one-bit problems and concatenate.**

### Pipeline
For each bit, scan in fixed order (first match wins):
1. **CONSTANTS**: 0 (all bits 0), 1 (all bits 1)
2. **1-INPUT**: IDENTITY (OUT[i] = IN[j]), NOT (OUT[i] = NOT IN[j])
3. **2-INPUT GATES**: AND, OR, XOR, NAND, NOR, XNOR + 4 negation variants:
   - (NOT IN[j]) AND IN[k]
   - IN[j] AND (NOT IN[k])
   - (NOT IN[j]) OR IN[k]
   - IN[j] OR (NOT IN[k])
4. **3-INPUT**: MAJ, CHO (IF j==1 THEN k ELSE l), PAR3 (j XOR k XOR l), AO/OA/AX/OX/XA/XO
5. **4-INPUT**: AOA, OAO, PAR4, XX, AXA

### Two things that matter
1. **Bit-serial gate computation** (NOT parallel). Model can't do multi-bit AND/OR/XOR in parallel — accuracy craters to 9.3%. Force it to spell out every op one bit at a time: `0&1=0  1&1=1  0&0=0`.
2. **Target verification (VER)**. Multiple ops can match all 8 example columns by coincidence. Every candidate must be checked against the actual test input, not just visible examples.

### Reward structure
Per-bit GRPO at every step. Partial credit on:
- Laying out OUT columns correctly
- Picking the right op
- Bit-serial gate computation
- VER target check
- Final answer bit (each bit position scored independently)

8/8 correct triggers superlinear "champagne" bonus (+5 flat).

### Brute force baseline (m4nocha comment)
Applying these gate definitions with brute force only: **53.87% accuracy (863/1602)**. Need bit-serial reasoning + VER on top.

---

## 2. ENGLISH CIPHERS (cipher)

Encrypted phrase in, plaintext phrase out. The cipher is a **bijective derangement** on a-z (every letter maps to exactly one other letter, no letter maps to itself).

### Pipeline
1. **LEN** — extract word lengths from the target
2. **TABLE** — extract mappings by walking example pairs letter by letter
3. **VER** — cross-validate the table against all examples + a held-out example
4. **DECRYPT** — char-by-char on the target, back-referencing confirmed mappings
5. **CHECK** — length match, alpha-only, vocabulary membership, no gaps
6. **ANS**

### Words come from a fixed ~90-word vocabulary
Predictable patterns: character-verb-object, character-verb-adjective-object, etc. The phrase structure is tight.

### Two things that matter
1. **Char-by-char decryption** (cipher version of bit-serial fix). If you let the model decrypt whole words, the language prior hijacks it and it hallucinates plausible English instead of applying the table. Force one cipher letter at a time, look up each mapping.
2. **VOCAB fill** for incomplete tables. With 6-8 example phrases you almost never cover all 26 letters → target has gaps marked `?`. When a decoded word has gaps, match against the fixed vocabulary by length, find the word that fits the confirmed letters, and fill the remaining mappings back into the table for downstream words.

### Reward
Per-letter GRPO: per-letter table accuracy, table coverage (N/26), VER cross-check, per-letter decrypt correctness, VOCAB fill validity, per-word answer, final CHECK pass. All-words-correct champagne bonus.

---

## 3. GRAVITY

Given a few `(t, d)` pairs that follow `d = 0.5 * g * t^2`, find distance for a new time. **g is randomized per problem** — must derive from examples, can't memorize.

### Pipeline
1. **SOLVE** — use EX1 to get the rate constant, apply to target
2. **VER** — rate consistency check against EX2
3. **ANS** — exactly `X.XX` format (2 decimal places, **wrong format = 0 points**)

### Two things that matter
1. **Rate-first decomposition**. Don't extract g explicitly, then compute `0.5*g*t_target^2` in 5 ops. Instead: `RATE = d/t^2` directly from EX1 (which IS 0.5g already), square the target time, multiply. **2 ops instead of 5** — smaller intermediates, fewer arithmetic mistakes.
2. **Rate consistency for VER**, NOT full recomputation. Recomputing two ways hits the same arithmetic weaknesses → confident wrong agreement. Instead: derive RATE from EX1, derive RATE2 from EX2 independently, check `|RATE - RATE2| < 0.05`. Catches "both computations wrong in the same direction".

### Reward
Per-step GRPO: term, preamble structure, math accuracy at each intermediate (t^2, rate, target squared, result, rounded), tiered VER honesty, format compliance on X.XX, final answer.

---

## 4. ROMAN (numeral_system)

**Bidirectional**, trained 50/50 (Int↔Roman OR Roman↔Int) so the model is bulletproof both directions in case the hidden test set flips on you.

### Forward pipeline (Int → Roman)
1. **DECOMPOSE** — split target into all 4 place slots TH/HU/TE/ON, with zeros shown explicitly as SKIP
2. **CAT** — incremental concatenation, one segment at a time
3. **VER** — round-trip re-parse the assembled string
4. **ANS**

### Reverse pipeline (Roman → Int)
1. **PARSE** — walk symbol groups with running total, subtractive pairs (CM/CD/XC/XL/IX/IV) treated as **atomic units**
2. **VER** — rebuild Roman string from integer answer, string-compare to input
3. **ANS**

### Two things that matter
1. **Incremental CAT kills transposition errors**. If the model emits `MMDCLX` as one token blob, it WILL eat XL as LX or swap CM for MC (token attractors). Force `MM + DC = MMDC`, then `MMDC + LX = MMDCLX` — every concat step auditable.
2. **Round-trip VER**. Don't sum the original decomposed values (agrees with wrong answers because you're checking the shelf, not the pill bottle). **CataFix re-parses the assembled output string back to integers and sums.** If the model transposed during CAT, the reparse total won't match the target → CHK fires NO.

### Reward
Tiered VER honesty most pronounced here. Honest YES > honest NO > confused NO > lying YES (worst). Punishes "VER rubber-stamps wrong answer" failure mode harder than just getting VER wrong.

### Preamble anchor
Hard anchor on "Roman numeral" identity because this template cross-contaminates easily with binary/symbol/digit if model drifts.

---

## 5. UNIT CONVERSION (unit_conversion)

Structurally same as gravity but linear. `output = input * factor`, factor randomized per problem.

### Pipeline
1. **SOLVE** — RATE = out1/in1 from EX1, RESULT = target * RATE, round to 2dp
2. **VER** — rate consistency `|RATE - RATE2| < 0.01` (tighter than gravity's 0.05 because no squaring)
3. **ANS** — exactly X.XX format, **enforced in preamble**

### Format rules
- Intermediates: fmt4 (4 decimal places, trailing zeros stripped)
- Final answer: fmt2 (exactly 2 decimal places, **1.50 stays 1.50**)
- Wrong decimal places = 0
- Missing `\boxed{}` = -12 (instant death)

### Reward
Per-step GRPO: term, preamble identity, RATE/RESULT/RND accuracy, tiered VER honesty, format compliance, final answer. Perfect trace = +18. Contamination on brackets, semicolons, xor, AND, anything that smells like other templates leaking in.

---

## 6. SYMBOL-DIGIT (this is what we called equation_transform!)

Input is `AB⊕CD` (4 digits split by a random operator character), output is a number. Three things to figure out at once:

1. **How are the 4 digits paired into 2 two-digit operands?** (BA,DC? AB,CD? reversed?)
2. **What operation to apply?** 14 ops total: add, mul, sub, cat, mulsub1, muladd1, etc.
3. **What output format?** 14 formats: rev, raw, abs, dsum (digit sum), zpad2 (zero-padded), operator-prefixed variants, etc.

That's **4 × 14 × 14 = 784 possible combos**.

### Pipeline
1. **PARSE** — identify operator char, extract EX1 digits
2. **SCAN** — frequency-ordered brute force through **47 combos** that cover 99% of competition distribution
3. **LOCK** — commit to winning combo
4. **APPLY** — run on target
5. **ANS**

### Example
`03}43 = 47` resolves to `AB_CD | add1 | raw` because `L=3, R=43, 3+43+1=47`, raw format, match.

### Frequency hints
- Most common: `BA_DC | add | rev` at 13%
- Second: `BA_DC | mul | rev` at 12%

### Two things that matter
1. **Verify against EX2 immediately when a combo matches EX1**. Catches coincidental matches (especially add vs addm1 vs add1 differing by ±1).
2. **HARDSTOP teaching**. 10% of training traces use combos NOT in the 47-entry scan, so model learns to emit `#STOP:SCAN_LIMIT` and lock answer. Teaches "I scanned everything and nothing matched" instead of hallucinating a fake match.

### Reward
Per-step GRPO: scan quality (every scan line's arithmetic is verifiable), LOCK accuracy, VER correctness, final answer, **HARDSTOP bonus** for correctly emitting #STOP on unsolvable scans.

### Contamination markers
Template-specific (RATE:, DECOMPOSE, Roman numeral, gravity/unitconv language) — NOT character bans, because operator characters legitimately include brackets/braces/symbols.

---

## 7. CIPHER-DIGIT (NEW — also part of equation_transform?)

Symbol-digit + bijective cipher overlay. Every char (including digits!) replaced by random symbol via fresh bijective cipher per problem.

`*#\< = ##:#` — need to figure out BOTH cipher AND operation.

### Pipeline
1. **DETECT** — identify operator symbol by position (always **index 2** in input)
2. **CRACK** — build symbol→digit mapping from examples (e.g. `*=9, #=1, \=6, :=3`)
3. **SCAN** — same 47-entry frequency-ordered brute force, on decoded digits
4. **LOCK**
5. **APPLY** — decode target, run operation
6. **ENCODE** — re-encrypt numeric answer back to cipher symbols, **one digit at a time**
7. **ANS** — boxed in cipher symbols, NOT digits

Format set reduced to just `rev/raw/abs` (operator-prefixed formats make no sense when operator is encrypted).

### Two things that matter
1. **Answer must come back in cipher**. Crack cipher → solve symbol-digit on decoded → reverse cipher to encode output. Any mistake propagates.
2. **Factory enforces** every digit appearing in target's output is visible somewhere in examples (otherwise model has no way to learn that digit's encoding).
3. **Full-pipeline VER**: decode EX2's input, form operands, compute, format, re-encrypt, compare to EX2's actual cipher output. Catches errors at every stage.

### Reward (most layers of any factory)
- `r_cipher` — per-symbol cipher accuracy (each mapping pair scored independently)
- decode CHK
- scan quality
- LOCK
- tiered VER honesty
- `r_encode` — per-digit encoding accuracy (each output digit scored)
- final answer
- HARDSTOP bonus

**Perfect trace = +33** (highest of any factory).

A wrong cipher mapping that cascades into wrong answer still scores partial credit on scan quality, LOCK, and any correct encode digits.

### Convergence with symbol-digit
Donald confirmed via k-means on operation frequencies: centroids land within 0.0004. **The cipher layer is cosmetic.** Once you crack the mapping, it IS bare symbol-digit.

---

## What This Means For Our Approach

### Immediate consequences
1. **Our SFT data is wrong-shaped.** We've been generating generic CoT. Donald's playbook says we need TEMPLATE-SPECIFIC pipelines with hard step structure (TABLE/VER/CHECK/ANS labels) and per-step verification.
2. **Cipher and equation_transform are actually 7 sub-types**, not 6 — symbol-digit and cipher-digit are both in equation_transform.
3. **VER/CHECK steps are mandatory** — not optional reasoning. Every intermediate must be checked.
4. **Bit-serial / char-by-char execution is the canonical fix** for hallucination — never let the model do parallel/whole-word/whole-blob ops.
5. **Format strictness** is severe: gravity/unit_con need exactly X.XX or score 0.

### Actionable training-data changes
- Generate per-category pipeline templates with hard step labels (`LEN:`, `TABLE:`, `VER:`, `DECRYPT:`, `CHECK:`, `ANS:`, `\boxed{}`)
- For binary: bit-serial computation, scan order CONSTANTS→IDENTITY→NOT→2-input→3-input→4-input, VER on actual test input
- For cipher: char-by-char decrypt, VOCAB fill with the ~90-word vocabulary
- For gravity/unit_con: rate-first, two-rate VER instead of recomputation
- For roman: incremental CAT, round-trip VER (re-parse assembled string)
- For equation_transform: 47-combo scan, EX2 verification, HARDSTOP for unsolvable
- For cipher-digit: index-2 operator detection, full-pipeline VER

### Reward function for GRPO
- Per-step partial credit on each labeled stage
- Tiered VER honesty (honest YES > honest NO > confused NO > lying YES)
- Champagne bonus on fully-correct
- Penalize contamination markers (other templates' language)
- Penalize thrash markers ("hmm", "let me try", "actually")

### Reverse-engineer the ~90-word cipher vocabulary
This is a concrete TODO. Mine training data for all unique decoded words in cipher type → build the vocabulary set.

### Reverse-engineer the 47-combo scan order
For symbol-digit: enumerate all (pair-order, op, format) combos in training data, sort by frequency, take top 47.

---

## Quotes worth keeping

> "You don't guess forward hoping it works. Every VER step, every CHK, every false-positive rejection in the scan is the same principle: prove each constraint before committing, and if the proof fails downstream, walk it back. **That's why the templates work. That's why they're 100% solvable.**"

> "The Open Contribution Awards for best data method, best RL method, and best fine-tuning method are all locked behind a top 10% final leaderboard placement, which means even if your methodology is the cleanest in the competition, you can't win recognition for it without the hardware to push a submission into the top of the board."

> "BA_DC | add | rev at 13%, BA_DC | mul | rev at 12%" — the dominant equation_transform combo is operand-reversed addition with reversed output.
