"""Gravity verifier: d = 0.5 * g * t^2 with secretly-changed g.

Ground truth: derive g from each example, average, apply to target t.
Detection: did model use rate-first (d/t^2 = 0.5g) or full g extraction?
Did model engage in cipher-template drift (shouldn't happen but v30 might)?
"""
import re
from typing import Optional, List, Tuple


def parse_prompt(prompt: str) -> Tuple[List[Tuple[float, float]], Optional[float]]:
    obs = []
    target_t = None
    for line in prompt.split('\n'):
        m = re.search(r't\s*=\s*([\d.]+)\s*s?\s*,\s*distance\s*=\s*([\d.]+)', line, re.IGNORECASE)
        if m:
            obs.append((float(m.group(1)), float(m.group(2))))
            continue
        m2 = re.search(r'falling distance for\s+t\s*=\s*([\d.]+)', line, re.IGNORECASE)
        if m2:
            target_t = float(m2.group(1))
    return obs, target_t


def solve(obs, target_t):
    if not obs or target_t is None:
        return None
    gs = [2 * d / (t * t) for t, d in obs]
    g_avg = sum(gs) / len(gs)
    return 0.5 * g_avg * target_t * target_t


def verify(prompt: str, expected: str, trace: str, boxed: str) -> dict:
    obs, target_t = parse_prompt(prompt)
    true_d = solve(obs, target_t)

    try:
        pred = float(boxed)
        exp = float(expected)
        correct = abs(pred - exp) < 0.01 or (abs(pred - exp) / max(0.01, abs(exp)) < 0.01)
    except Exception:
        correct = False

    # Detect template drift
    tl = trace.lower()
    cipher_template_markers = sum(1 for m in ["table:", "decrypt", "vocab fill", "cipher"] if m in tl)
    has_rate_first = "rate" in tl and ("d/t" in tl or "d / t" in tl)
    has_full_g = "g =" in tl or "g=" in tl

    if correct:
        failure_mode = "correct"
    elif not boxed:
        failure_mode = "no_boxed"
    elif cipher_template_markers >= 2:
        failure_mode = "cipher_template_drift"
    else:
        failure_mode = "wrong_answer"

    return {
        "correct": correct,
        "failure_mode": failure_mode,
        "ground_truth": {
            "n_observations": len(obs),
            "target_t": target_t,
            "true_distance": true_d,
        },
        "model_facts": {
            "boxed": boxed,
            "used_rate_first": has_rate_first,
            "used_full_g": has_full_g,
        },
        "quality_flags": {
            "cipher_template_markers": cipher_template_markers,
        },
    }
