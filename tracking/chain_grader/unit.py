"""Unit-conversion reasoning-chain grader.

Scores each step of the model's reasoning on unit-conversion puzzles
(linear conversion with a secret slope).

Uses helpers from ``tracking/verifiers/unit.py`` for ground truth.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_TRACKING_DIR = str(Path(__file__).resolve().parent.parent)
if _TRACKING_DIR not in sys.path:
    sys.path.insert(0, _TRACKING_DIR)

from verifiers.unit import parse_prompt  # noqa: E402


# ----------------------------------------------------------------------------
# Step 1: identify_type
# ----------------------------------------------------------------------------
_IDENTIFY_TERMS = (
    'conversion', 'convert',
    'becomes',
    'ratio',
    'factor',
    'slope',
    'rate',
    'linear',
    'measurement',
)


def _step_identify_type(trace: str) -> dict:
    tl = trace.lower()
    hits = [t for t in _IDENTIFY_TERMS if t in tl]
    passed = len(hits) > 0
    return {
        'name': 'identify_type',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': f"conversion terms: {hits[:4]}" if hits else "no conversion/ratio terms in trace",
    }


# ----------------------------------------------------------------------------
# Step 2: parse_examples
# ----------------------------------------------------------------------------
def _step_parse_examples(trace: str, examples) -> dict:
    if not examples:
        return {
            'name': 'parse_examples',
            'passed': False,
            'score': 0.0,
            'detail': 'no example pairs parsed from prompt',
        }
    found = 0
    for x, y in examples:
        x_str = f"{x:g}"
        y_str = f"{y:g}"
        if x_str in trace and y_str in trace:
            found += 1
    passed = found >= 2
    score = min(1.0, found / max(1, len(examples)))
    return {
        'name': 'parse_examples',
        'passed': passed,
        'score': score,
        'detail': f"{found}/{len(examples)} (in, out) pairs referenced in trace",
    }


# ----------------------------------------------------------------------------
# Step 3: slope_computed
# ----------------------------------------------------------------------------
# Look for any division expression or an explicit slope/rate identifier.
_SLOPE_TERMS = [
    'slope',
    'rate',
    'ratio',
    'factor',
    'y/x',
    'y / x',
    'out/in',
    'out / in',
]
_DIVISION_RE = re.compile(r'\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?')


def _step_slope_computed(trace: str) -> dict:
    tl = trace.lower()
    term_hits = [t for t in _SLOPE_TERMS if t in tl]
    division_hits = _DIVISION_RE.findall(trace)
    passed = len(term_hits) >= 1 or len(division_hits) >= 1
    return {
        'name': 'slope_computed',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': (
            f"slope_terms={term_hits[:3]} divisions={division_hits[:3]}"
            if passed
            else "no slope/ratio term and no numeric division in trace"
        ),
    }


# ----------------------------------------------------------------------------
# Step 4: applied_to_target
# ----------------------------------------------------------------------------
def _step_applied_to_target(trace: str, target) -> dict:
    if target is None:
        return {
            'name': 'applied_to_target',
            'passed': False,
            'score': 0.0,
            'detail': 'no target extracted from prompt',
        }
    t_str = f"{target:g}"
    in_trace = t_str in trace
    return {
        'name': 'applied_to_target',
        'passed': in_trace,
        'score': 1.0 if in_trace else 0.0,
        'detail': f"target={t_str} {'found in' if in_trace else 'missing from'} trace",
    }


# ----------------------------------------------------------------------------
# Step 5: boxed_format_2dec
# ----------------------------------------------------------------------------
_BOXED_RE = re.compile(r'\\boxed\{([^}]*)\}')
_TWO_DEC_RE = re.compile(r'^-?\d+\.\d{2}$')


def _step_boxed_format_2dec(trace: str, boxed: str) -> dict:
    in_trace = bool(_BOXED_RE.search(trace))
    if not boxed:
        return {
            'name': 'boxed_format_2dec',
            'passed': False,
            'score': 0.0,
            'detail': 'empty boxed',
        }
    stripped = boxed.strip()
    is_2dec = bool(_TWO_DEC_RE.match(stripped))
    try:
        float(stripped)
        is_numeric = True
    except Exception:
        is_numeric = False
    passed = in_trace and (is_2dec or is_numeric)
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

    examples, target = parse_prompt(prompt or '')

    steps = [
        _step_identify_type(trace),
        _step_parse_examples(trace, examples),
        _step_slope_computed(trace),
        _step_applied_to_target(trace, target),
        _step_boxed_format_2dec(trace, boxed),
    ]

    pass_count = sum(1 for s in steps if s['passed'])
    fail_count = len(steps) - pass_count
    first_failure = next((s['name'] for s in steps if not s['passed']), None)
    chain_score = pass_count / len(steps)

    return {
        'type': 'unit',
        'steps': steps,
        'first_failure': first_failure,
        'pass_count': pass_count,
        'fail_count': fail_count,
        'chain_score': chain_score,
    }


if __name__ == '__main__':
    test_prompt = (
        "In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:\n"
        "29.01 m becomes 38.90\n"
        "26.89 m becomes 36.06\n"
        "Now, convert the following measurement: 33.89 m"
    )
    trace = (
        "SOLVE:\nEX1: in=29.01, out=38.9\nRATE = out / in = 38.9 / 29.01 = 1.340917\n"
        "EX2: in=26.89, out=36.06\nRATE2 = 1.341019\nVER: PASS\n"
        "APPLY: target = 33.89, result = 45.44\n\\boxed{45.44}"
    )
    r = grade_chain(test_prompt, '45.44', trace, '45.44')
    print(f"chain_score={r['chain_score']:.2f} first_fail={r['first_failure']}")
    for s in r['steps']:
        mk = '[P]' if s['passed'] else '[F]'
        print(f"  {mk} {s['name']:<22} {s['detail']}")
