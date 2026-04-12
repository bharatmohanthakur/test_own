# Distilled CoT Guide — Nvidia Nemotron Reasoning Challenge

**Dataset:** ~69,030 training examples across 6 puzzle types
**Goal:** For each puzzle type, a distilled reasoning template to solve any instance

---

## Type 1: Bit Manipulation (~1,602 examples)

**Pattern:** An 8-bit binary input is transformed to an 8-bit binary output via a hidden rule. ~8 input→output pairs are given; solve for a new input.

**Distilled CoT Template:**

1. **Test single operations first:** NOT, reverse bits, rotate left/right by 1–7, shift left/right by 1–7.
2. **Test XOR with a fixed mask:** Compute input XOR output for each pair. If the mask is the same across all pairs → answer = query XOR mask.
3. **Test two-step combos:** rotate-then-XOR, NOT-then-rotate, shift-then-AND, etc.
4. **Verify:** Any candidate rule MUST produce the correct output for ALL given pairs before applying to the query.
5. **Apply:** Once the consistent rule is found, apply it to the query input.

**Key insight:** Each problem has a UNIQUE rule. Common operations: rotate left/right N, XOR constant, NOT, bit reverse, shift + mask. Many are two-step combos.

---

## Type 2: Cipher / Encryption (~1,576 examples)

**Pattern:** Encrypted words → plaintext words. The cipher applies letter-by-letter.

**Distilled CoT Template:**

1. **Align ciphertext and plaintext character by character** across all example pairs.
2. **Build the substitution table:** For each cipher letter, record which plaintext letter it maps to.
3. **Verify consistency:** Each cipher letter should ALWAYS map to the same plaintext letter (monoalphabetic substitution).
4. **Check shift pattern:** Sometimes the shift is constant (Caesar cipher, e.g., every letter shifted by +13). Other times each letter has its own mapping.
5. **Decode the query:** Look up each cipher letter in your substitution table and output the plaintext.

**Key insight:** Always monoalphabetic substitution. Build the mapping table from ALL example pairs (more pairs = more letters discovered). Spaces and word boundaries are preserved.

---

## Type 3: Numeral System (~1,576 examples)

**Pattern:** A decimal number is converted to a different numeral system.

**Distilled CoT Template:**

1. **Identify the system from examples.** Common systems:
   - Roman numerals (XI, XIV, XLII, etc.)
   - Binary, octal, hexadecimal
   - Custom base-N systems
   - Mayan or other historical systems
2. **Verify:** Check that your identified system correctly produces ALL example outputs.
3. **Convert the query number** using the identified system's rules.

**Key insight:** Most commonly it's Roman numerals. For Roman: decompose the number into standard values (1000=M, 900=CM, 500=D, 400=CD, 100=C, 90=XC, 50=L, 40=XL, 10=X, 9=IX, 5=V, 4=IV, 1=I) and concatenate.

---

## Type 4: Unit Conversion (~1,594 examples)

**Pattern:** A measurement (e.g., in meters) is converted to a secret unit via a linear scaling factor.

**Distilled CoT Template:**

1. **Compute the ratio** output ÷ input for each example pair.
2. **Check consistency:** All ratios should be approximately equal (within rounding tolerance).
3. **Determine the conversion factor** by averaging the ratios.
4. **Apply:** Multiply the query value by the factor. Round to 2 decimal places.

**Key insight:** It's always a simple multiplication: output = input × constant. The constant varies per problem but is consistent within a problem. Common real-world conversion factors include meters→feet (3.281), meters→yards (1.094), km→miles (0.6214), etc., but Wonderland may use any factor.

---

## Type 5: Gravity / Custom g (~1,597 examples)

**Pattern:** Free-fall distances are given for various times under a secret gravitational constant. Formula: d = 0.5 × g × t².

**Distilled CoT Template:**

1. **Back-solve g from any example pair:** g = 2d / t²
2. **Verify across all pairs:** Compute g from each pair; they should all agree (within rounding).
3. **Average g** for maximum precision.
4. **Apply to query:** d = 0.5 × g × t_query²
5. **Round to 2 decimal places.**

**Key insight:** Each problem has a different secret g. Earth's g ≈ 9.81, but Wonderland values range widely. The formula is always d = 0.5·g·t². Just find g and plug in.

---

## Type 6: Equation Transform / Symbol Arithmetic (~1,555 examples)

**Pattern:** Arithmetic equations are written using symbol characters instead of digits. ~4 example equations are given; solve a new expression.

**Distilled CoT Template:**

1. **Parse the structure:** The left side is typically 5 characters: two-char operand, one operator (+, -, *), two-char operand. The right side is the result.
2. **Identify the operator:** The middle character of the left side is +, -, or *.
3. **List all unique symbols** and treat each as an unknown digit (0–9).
4. **Set up equations:** E.g., if `AB*CD = EFG`, then (10A+B) × (10C+D) = (100E+10F+G).
5. **Solve by constraint propagation or enumeration:**
   - Leading digits can't be 0
   - Each symbol maps to exactly one digit (injective mapping)
   - Use the ~4 equations to narrow down possibilities
6. **Apply the solved mapping** to the query expression and compute the result.

**Key insight:** Each problem has its own symbol→digit mapping. Operators are real arithmetic. Some use special characters (backtick, backslash, etc.) which can be tricky to parse. Focus on identifying the operator position first.

---

## General Strategy for All Types

1. **Classify the puzzle type** by reading the first line of the prompt.
2. **Apply the type-specific CoT template** above.
3. **Always verify** your discovered rule/mapping/constant against ALL given examples before answering.
4. **Format the answer** to match the expected output format (binary string, text, number with decimals, Roman numeral, or symbol string).
