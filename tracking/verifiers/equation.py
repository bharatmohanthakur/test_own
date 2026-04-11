"""Equation transform verifier.

Ground truth: 5-char inputs with operator at position 2; operands at [0,1] and [3,4].
Two sub-types:
- SYMBOL-DIGIT: operand chars are raw digits 0-9 (operator char is random/cosmetic)
- CIPHER-DIGIT: operand chars are symbols; model must crack a symbol→digit map first

Per Donald Galliano III: the operation is one of 47 (order, op, reverse) combos.
The operator char in the input is NOT the operation — it's cosmetic.

This verifier does NOT attempt to solve equation puzzles. It detects:
- whether input is digit vs symbol
- whether model ran a cipher-template (LEN/TABLE/DECOMPOSE/CAT) — the v30 failure mode
- whether model tried arithmetic at all
- string-match correctness

Failure modes:
- correct
- cipher_template_misfire  — v30-style: ran cipher pipeline on arithmetic puzzle
- copied_input             — answer identical to input target (did nothing)
- no_arithmetic_attempted  — no numbers in trace, no operation keywords
- arithmetic_wrong         — engaged but got wrong answer
- no_boxed
"""
import re
from typing import Optional, List, Tuple


def parse_prompt(prompt: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    pairs = []
    target = None
    for line in prompt.split('\n'):
        line = line.strip()
        # Match "XXXXX = YYY" (input is exactly 5 chars, output variable)
        m = re.match(r'^(.{5})\s*=\s*(.+)$', line)
        if m and 'example' not in line.lower() and 'determine' not in line.lower():
            pairs.append((m.group(1), m.group(2).strip()))
            continue
        m2 = re.search(r'determine the result for[:\s]+(.{1,10})$', line, re.IGNORECASE)
        if m2:
            target = m2.group(1).strip()
    return pairs, target


def detect_subtype(pairs: List[Tuple[str, str]]) -> str:
    """SYMBOL-DIGIT (raw digits) vs CIPHER-DIGIT (symbol substitution)."""
    all_operand_chars = []
    for inp, _ in pairs:
        if len(inp) >= 5:
            all_operand_chars.extend([inp[0], inp[1], inp[3], inp[4]])
    if not all_operand_chars:
        return "UNKNOWN"
    digit_frac = sum(1 for c in all_operand_chars if c.isdigit()) / len(all_operand_chars)
    return "SYMBOL-DIGIT" if digit_frac > 0.7 else "CIPHER-DIGIT"


# ----------------------------------------------------------------------------
# Trace analysis
# ----------------------------------------------------------------------------
CIPHER_TEMPLATE_MARKERS = [
    "len:", "table:", "decompose", "decrypt", "vocab fill", "cat:", "check:",
    "table (extracted", "substitution table",
]

ARITHMETIC_MARKERS = [
    "add", "sum", "+", "multiply", "product", "×", "*",
    "subtract", "difference", "divide", "quotient",
    "reverse", "concat", "swap",
]


def detect_cipher_template(trace: str) -> int:
    """Count cipher-template markers in trace. >3 means model is mis-applying cipher pipeline."""
    tl = trace.lower()
    return sum(1 for m in CIPHER_TEMPLATE_MARKERS if m in tl)


def detect_arithmetic_attempt(trace: str) -> int:
    tl = trace.lower()
    return sum(1 for m in ARITHMETIC_MARKERS if m in tl)


def verify(prompt: str, expected: str, trace: str, boxed: str) -> dict:
    pairs, target = parse_prompt(prompt)
    subtype = detect_subtype(pairs)

    cipher_markers = detect_cipher_template(trace)
    arith_markers = detect_arithmetic_attempt(trace)

    correct = (boxed.strip() == str(expected).strip())

    if correct:
        failure_mode = "correct"
    elif not boxed:
        failure_mode = "no_boxed"
    elif target and boxed.strip() == target:
        failure_mode = "copied_input"
    elif cipher_markers >= 3:
        failure_mode = "cipher_template_misfire"
    elif arith_markers < 2:
        failure_mode = "no_arithmetic_attempted"
    else:
        failure_mode = "arithmetic_wrong"

    return {
        "correct": correct,
        "failure_mode": failure_mode,
        "ground_truth": {
            "n_examples": len(pairs),
            "target": target,
            "subtype": subtype,
        },
        "model_facts": {
            "boxed": boxed,
            "cipher_template_markers": cipher_markers,
            "arithmetic_markers": arith_markers,
        },
        "quality_flags": {
            "ran_cipher_template": cipher_markers >= 3,
            "attempted_arithmetic": arith_markers >= 2,
        },
    }


if __name__ == '__main__':
    test = """In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
34/44 = 1
41/32 = 9
34|25 = 69
87\\64 = 8853
Now, determine the result for: 69/52"""
    r = verify(test, "17", "<no trace>", "17")
    print("digit subtype:", r["ground_truth"]["subtype"], "failure_mode:", r["failure_mode"])
