"""Unit conversion verifier: linear conversion with secret slope.

Ground truth: slope s = y/x, averaged across examples, applied to target.
"""
import re
from typing import Optional, List, Tuple


def parse_prompt(prompt: str) -> Tuple[List[Tuple[float, float]], Optional[float]]:
    examples = []
    target = None
    for line in prompt.split('\n'):
        # "X.XX m becomes Y.YY" or similar
        m = re.search(r'([\d.]+)\s*m\s*becomes\s*([\d.]+)', line, re.IGNORECASE)
        if m:
            examples.append((float(m.group(1)), float(m.group(2))))
            continue
        m2 = re.search(r'convert the following measurement[:\s]+([\d.]+)', line, re.IGNORECASE)
        if m2:
            target = float(m2.group(1))
    return examples, target


def solve(examples, target):
    if not examples or target is None:
        return None
    slopes = [y / x for x, y in examples if x != 0]
    s_avg = sum(slopes) / len(slopes)
    return s_avg * target


def verify(prompt: str, expected: str, trace: str, boxed: str) -> dict:
    examples, target = parse_prompt(prompt)
    true_y = solve(examples, target)

    try:
        pred = float(boxed)
        exp = float(expected)
        correct = abs(pred - exp) < 0.01 or (abs(pred - exp) / max(0.01, abs(exp)) < 0.01)
    except Exception:
        correct = False

    tl = trace.lower()
    cipher_markers = sum(1 for m in ["table:", "decrypt", "vocab fill"] if m in tl)

    if correct:
        failure_mode = "correct"
    elif not boxed:
        failure_mode = "no_boxed"
    elif cipher_markers >= 2:
        failure_mode = "cipher_template_drift"
    else:
        failure_mode = "wrong_slope"

    return {
        "correct": correct,
        "failure_mode": failure_mode,
        "ground_truth": {"target": target, "true_y": true_y, "n_examples": len(examples)},
        "model_facts": {"boxed": boxed},
        "quality_flags": {"cipher_template_markers": cipher_markers},
    }
