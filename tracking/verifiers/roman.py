"""Roman numeral verifier. Direction detection + ground truth computation.

Wonderland uses standard Roman numerals (per Donald playbook). Two directions:
- arabic → roman: "write the number N"
- roman → arabic: "what arabic number is R"
"""
import re
from typing import Optional, Tuple


ARABIC_TO_ROMAN = [
    (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
    (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
    (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),
]


def to_roman(n: int) -> str:
    out = []
    for val, sym in ARABIC_TO_ROMAN:
        while n >= val:
            out.append(sym)
            n -= val
    return ''.join(out)


def to_arabic(s: str) -> Optional[int]:
    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    try:
        total = 0
        prev = 0
        for c in reversed(s.upper()):
            v = vals[c]
            if v < prev:
                total -= v
            else:
                total += v
            prev = v
        return total
    except Exception:
        return None


def parse_prompt(prompt: str) -> Tuple[Optional[int], Optional[str], str]:
    """Returns (target_arabic, target_roman, direction)."""
    for line in prompt.split('\n'):
        m = re.search(r'write the number\s+(\d+)', line, re.IGNORECASE)
        if m:
            return (int(m.group(1)), None, 'arabic_to_roman')
        m2 = re.search(r'what.*?(?:arabic\s+)?number\s+is\s+([IVXLCDM]+)', line, re.IGNORECASE)
        if m2:
            return (None, m2.group(1).upper(), 'roman_to_arabic')
        m3 = re.search(r'convert\s+([IVXLCDM]+)\s+to', line, re.IGNORECASE)
        if m3:
            return (None, m3.group(1).upper(), 'roman_to_arabic')
    return (None, None, 'unknown')


def verify(prompt: str, expected: str, trace: str, boxed: str) -> dict:
    target_a, target_r, direction = parse_prompt(prompt)

    true_answer = None
    if direction == 'arabic_to_roman' and target_a is not None:
        true_answer = to_roman(target_a)
    elif direction == 'roman_to_arabic' and target_r is not None:
        true_answer = str(to_arabic(target_r) or '')

    pred = boxed.strip()
    exp = str(expected).strip()
    correct = (pred.upper() == exp.upper())

    tl = trace.lower()
    cipher_markers = sum(1 for m in ["table:", "decrypt", "vocab fill"] if m in tl)

    if correct:
        failure_mode = "correct"
    elif not boxed:
        failure_mode = "no_boxed"
    elif cipher_markers >= 2:
        failure_mode = "cipher_template_drift"
    else:
        failure_mode = "wrong_answer"

    return {
        "correct": correct,
        "failure_mode": failure_mode,
        "ground_truth": {"direction": direction, "true_answer": true_answer},
        "model_facts": {"boxed": boxed},
        "quality_flags": {"cipher_template_markers": cipher_markers},
    }
