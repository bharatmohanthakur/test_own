"""Gravity reasoning-chain grader.

Scores each step of the model's reasoning on gravity puzzles
(d = 0.5 * g * t^2 with a secretly-changed g).

Uses helpers from ``tracking/verifiers/gravity.py`` for ground truth.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_TRACKING_DIR = str(Path(__file__).resolve().parent.parent)
if _TRACKING_DIR not in sys.path:
    sys.path.insert(0, _TRACKING_DIR)

from verifiers.gravity import parse_prompt  # noqa: E402


# ----------------------------------------------------------------------------
# Step 1: identify_type
# ----------------------------------------------------------------------------
_IDENTIFY_TERMS = (
    'gravitational', 'gravity',
    '0.5*g', '0.5 * g', '0.5g',
    'falling distance', 'falling',
    'g =', 'g=', ' g ',
)


def _step_identify_type(trace: str) -> dict:
    tl = trace.lower()
    hits = [t for t in _IDENTIFY_TERMS if t in tl]
    passed = len(hits) > 0
    return {
        'name': 'identify_type',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': f"gravity terms: {hits[:4]}" if hits else "no gravity terms in trace",
    }


# ----------------------------------------------------------------------------
# Step 2: parse_observations
# ----------------------------------------------------------------------------
def _step_parse_observations(trace: str, obs) -> dict:
    if not obs:
        return {
            'name': 'parse_observations',
            'passed': False,
            'score': 0.0,
            'detail': 'no (t,d) pairs parsed from prompt',
        }
    found = 0
    for t, d in obs:
        # Require BOTH t and d to appear near each other. A simple approximate
        # match: both the t-string and d-string appear in the trace.
        t_str = f"{t:g}"
        d_str = f"{d:g}"
        if t_str in trace and d_str in trace:
            found += 1
    passed = found >= 2
    score = min(1.0, found / max(1, len(obs)))
    return {
        'name': 'parse_observations',
        'passed': passed,
        'score': score,
        'detail': f"{found}/{len(obs)} (t,d) pairs referenced in trace",
    }


# ----------------------------------------------------------------------------
# Step 3: g_computed
# ----------------------------------------------------------------------------
_G_FORM_RE = re.compile(r"g\s*=\s*[-+]?\d", re.IGNORECASE)
_RATE_FORM_RE = re.compile(r"rate\s*[0-9]*\s*=\s*[-+]?\d", re.IGNORECASE)
_G_FORMULA_RE = re.compile(r"2\s*\*\s*d\s*/\s*t", re.IGNORECASE)
_RATE_FORMULA_RE = re.compile(r"d\s*/\s*t\s*[\^*²]?\s*2?", re.IGNORECASE)


def _step_g_computed(trace: str) -> dict:
    has_g = bool(_G_FORM_RE.search(trace))
    has_rate = bool(_RATE_FORM_RE.search(trace))
    has_g_formula = bool(_G_FORMULA_RE.search(trace))
    has_rate_formula = bool(_RATE_FORMULA_RE.search(trace))
    passed = has_g or has_rate or has_g_formula or has_rate_formula
    detail = (
        f"g-form={has_g} rate-form={has_rate} "
        f"g-formula={has_g_formula} rate-formula={has_rate_formula}"
    )
    return {
        'name': 'g_computed',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': detail,
    }


# ----------------------------------------------------------------------------
# Step 4: g_consistency_check
# ----------------------------------------------------------------------------
_CONSISTENCY_TERMS = [
    'average', 'avg',
    'matches', 'match',
    'consistent', 'consistency',
    'ver:', 'ver ', 'verify',
    'tolerance', 'pass', 'similar',
    '|rate', 'rate2', 'rate 2', 'g2', 'g 2',
]


def _step_g_consistency_check(trace: str) -> dict:
    tl = trace.lower()
    hits = [t for t in _CONSISTENCY_TERMS if t in tl]
    passed = len(hits) >= 1
    return {
        'name': 'g_consistency_check',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': f"consistency markers: {hits[:4]}" if hits else "no consistency/verify markers",
    }


# ----------------------------------------------------------------------------
# Step 5: applied_to_target
# ----------------------------------------------------------------------------
def _step_applied_to_target(trace: str, target_t) -> dict:
    if target_t is None:
        return {
            'name': 'applied_to_target',
            'passed': False,
            'score': 0.0,
            'detail': 'no target t extracted from prompt',
        }
    t_str = f"{target_t:g}"
    in_trace = t_str in trace
    return {
        'name': 'applied_to_target',
        'passed': in_trace,
        'score': 1.0 if in_trace else 0.0,
        'detail': f"target t={t_str} {'found in' if in_trace else 'missing from'} trace",
    }


# ----------------------------------------------------------------------------
# Step 6: boxed_format_2dec
# ----------------------------------------------------------------------------
_BOXED_RE = re.compile(r'\\boxed\{([^}]*)\}')
_TWO_DEC_RE = re.compile(r'^-?\d+\.\d{2}$')


def _step_boxed_format_2dec(trace: str, boxed: str) -> dict:
    m = _BOXED_RE.search(trace)
    in_trace = bool(m)
    if not boxed:
        return {
            'name': 'boxed_format_2dec',
            'passed': False,
            'score': 0.0,
            'detail': 'empty boxed',
        }
    stripped = boxed.strip()
    is_2dec = bool(_TWO_DEC_RE.match(stripped))
    # Also accept 1 or 3 decimals as a fallback (close match).
    try:
        float(stripped)
        is_numeric = True
    except Exception:
        is_numeric = False
    passed = in_trace and (is_2dec or is_numeric)
    # Full credit only if exact 2-decimal format
    score = 1.0 if (in_trace and is_2dec) else (0.5 if passed else 0.0)
    return {
        'name': 'boxed_format_2dec',
        'passed': passed,
        'score': score,
        'detail': f"boxed='{stripped}' 2dec={is_2dec} numeric={is_numeric}",
    }


# ----------------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------------
def grade_chain(prompt: str, expected: str, trace: str, boxed: str) -> dict:
    trace = trace or ''
    boxed = boxed or ''

    obs, target_t = parse_prompt(prompt or '')

    steps = [
        _step_identify_type(trace),
        _step_parse_observations(trace, obs),
        _step_g_computed(trace),
        _step_g_consistency_check(trace),
        _step_applied_to_target(trace, target_t),
        _step_boxed_format_2dec(trace, boxed),
    ]

    pass_count = sum(1 for s in steps if s['passed'])
    fail_count = len(steps) - pass_count
    first_failure = next((s['name'] for s in steps if not s['passed']), None)
    chain_score = pass_count / len(steps)

    return {
        'type': 'gravity',
        'steps': steps,
        'first_failure': first_failure,
        'pass_count': pass_count,
        'fail_count': fail_count,
        'chain_score': chain_score,
    }


if __name__ == '__main__':
    test_prompt = (
        "In Alice's Wonderland, the gravitational constant has been secretly changed. "
        "Here are some example observations:\n"
        "For t = 2.12s, distance = 27.94 m\n"
        "For t = 4.4s, distance = 120.36 m\n"
        "Now, determine the falling distance for t = 1.2s given d = 0.5*g*t^2."
    )
    trace = (
        "SOLVE (using d = 0.5*g*t^2, rate-first form):\n"
        "EX1: t=2.12, d=27.94\nRATE = d / t² = 27.94 / 4.4944 = 6.2167\n"
        "EX2: t=4.4, d=120.36\nRATE2 = 6.2169\n"
        "VER: |RATE - RATE2| < 0.05 PASS (g is consistent)\n"
        "APPLY: target_t = 1.2, result = 8.9521\n"
        "\\boxed{8.95}"
    )
    r = grade_chain(test_prompt, '8.95', trace, '8.95')
    print(f"chain_score={r['chain_score']:.2f} first_fail={r['first_failure']}")
    for s in r['steps']:
        mk = '[P]' if s['passed'] else '[F]'
        print(f"  {mk} {s['name']:<22} {s['detail']}")
