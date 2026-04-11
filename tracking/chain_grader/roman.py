"""Roman-numeral reasoning-chain grader.

Scores each step of the model's reasoning on roman-numeral puzzles
(bidirectional: arabic -> roman OR roman -> arabic).

Uses helpers from ``tracking/verifiers/roman.py`` for ground truth.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_TRACKING_DIR = str(Path(__file__).resolve().parent.parent)
if _TRACKING_DIR not in sys.path:
    sys.path.insert(0, _TRACKING_DIR)

from verifiers.roman import parse_prompt  # noqa: E402


_ROMAN_LETTERS = set('MDCLXVI')
_ROMAN_RE = re.compile(r'\b[MDCLXVI]+\b')


# ----------------------------------------------------------------------------
# Step 1: identify_type
# ----------------------------------------------------------------------------
_IDENTIFY_TERMS = (
    'roman', 'numeral',
)

# Roman letter tokens in assembled context. Count tokens like "X", "XXX",
# "XLVI", etc. — any 1+ character string made entirely of M/D/C/L/X/V/I.
_ROMAN_TOKEN_RE = re.compile(r'\b[MDCLXVI]{1,}\b')


def _step_identify_type(trace: str) -> dict:
    tl = trace.lower()
    hits = [t for t in _IDENTIFY_TERMS if t in tl]
    # Template-style v30 traces don't use the words "roman"/"numeral", so
    # accept the presence of Roman-letter tokens or TH/HU/TE/ON segment
    # labels as strong evidence the model is working on a Roman puzzle.
    roman_tokens = _ROMAN_TOKEN_RE.findall(trace)
    has_segment_labels = any(
        lbl in trace for lbl in ('TH =', 'HU =', 'TE =', 'ON =', 'segment', 'decompose', 'DECOMPOSE')
    )
    passed = len(hits) > 0 or len(roman_tokens) >= 3 or has_segment_labels
    return {
        'name': 'identify_type',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': (
            f"terms={hits} roman_tokens={len(roman_tokens)} segment_labels={has_segment_labels}"
        ),
    }


# ----------------------------------------------------------------------------
# Step 2: direction_detected
# ----------------------------------------------------------------------------
_A2R_TERMS = ['arabic to roman', 'arabic->roman', 'arabic -> roman', 'write the number', 'convert the number', 'arabic_to_roman']
_R2A_TERMS = ['roman to arabic', 'roman->arabic', 'roman -> arabic', 'what arabic', 'roman_to_arabic']


def _step_direction_detected(trace: str, direction: str) -> dict:
    tl = trace.lower()
    a2r = any(t in tl for t in _A2R_TERMS)
    r2a = any(t in tl for t in _R2A_TERMS)
    # Implicit direction: trace has DECOMPOSE / segments / apply the A->R table
    implicit_a2r = ('decompose' in tl and ('tens' in tl or 'segment' in tl or 'ones' in tl))
    implicit_r2a = ('parse' in tl or 're-parse' in tl) and 'total' in tl
    passed = a2r or r2a or implicit_a2r or implicit_r2a
    return {
        'name': 'direction_detected',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': f"direction={direction} a2r={a2r} r2a={r2a} implicit_a2r={implicit_a2r} implicit_r2a={implicit_r2a}",
    }


# ----------------------------------------------------------------------------
# Step 3: conversion_logic
# ----------------------------------------------------------------------------
_CONVERSION_TERMS = [
    'decompose', 'segment',
    'thousands', 'hundreds', 'tens', 'ones',
    'digit', 'place',
    'cat:', 'concat', 'running',
    'remainder',
    'write x', 'write v', 'write i', 'write l', 'write c', 'write d', 'write m',
]
# Also check for explicit value mappings like "100=C" / "10 = X" / "M = 1000"
_VAL_MAP_RE = re.compile(r"\b(?:1000|900|500|400|100|90|50|40|10|9|5|4|1)\b\s*[=:→]\s*[MDCLXVI]")
_LETTER_VAL_RE = re.compile(r"[MDCLXVI]\s*[=:]\s*(?:1000|900|500|400|100|90|50|40|10|9|5|4|1)")
# Inline decomposition pattern such as "32 >= 10" or ">= 1000"
_GE_DECOMPOSE_RE = re.compile(r">=\s*(?:1000|500|100|50|10|5|1)\b")


def _step_conversion_logic(trace: str) -> dict:
    tl = trace.lower()
    term_hits = [t for t in _CONVERSION_TERMS if t in tl]
    val_maps = _VAL_MAP_RE.findall(trace)
    letter_maps = _LETTER_VAL_RE.findall(trace)
    ge_hits = _GE_DECOMPOSE_RE.findall(trace)
    passed = (
        len(term_hits) >= 1
        or len(val_maps) >= 1
        or len(letter_maps) >= 1
        or len(ge_hits) >= 1
    )
    return {
        'name': 'conversion_logic',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': (
            f"terms={term_hits[:3]} val_maps={len(val_maps)} "
            f"letter_maps={len(letter_maps)} ge_hits={len(ge_hits)}"
            if passed
            else "no decomposition, value mapping, or >= pattern in trace"
        ),
    }


# ----------------------------------------------------------------------------
# Step 4: applied_to_target
# ----------------------------------------------------------------------------
def _step_applied_to_target(trace: str, target_a, target_r) -> dict:
    if target_a is not None:
        t_str = str(target_a)
        in_trace = t_str in trace
        return {
            'name': 'applied_to_target',
            'passed': in_trace,
            'score': 1.0 if in_trace else 0.0,
            'detail': f"arabic target={t_str} {'found' if in_trace else 'missing'}",
        }
    if target_r is not None:
        in_trace = target_r in trace
        return {
            'name': 'applied_to_target',
            'passed': in_trace,
            'score': 1.0 if in_trace else 0.0,
            'detail': f"roman target={target_r} {'found' if in_trace else 'missing'}",
        }
    return {
        'name': 'applied_to_target',
        'passed': False,
        'score': 0.0,
        'detail': 'no target extracted from prompt',
    }


# ----------------------------------------------------------------------------
# Step 5: boxed_format
# ----------------------------------------------------------------------------
_BOXED_RE = re.compile(r'\\boxed\{([^}]*)\}')


def _step_boxed_format(trace: str, boxed: str, expected: str) -> dict:
    in_trace = bool(_BOXED_RE.search(trace))
    if not boxed:
        return {
            'name': 'boxed_format',
            'passed': False,
            'score': 0.0,
            'detail': 'empty boxed',
        }
    stripped = boxed.strip()
    exp = (expected or '').strip()
    # Detect whether expected is roman or arabic and require matching format.
    expected_is_roman = bool(exp) and all(c in _ROMAN_LETTERS for c in exp.upper())
    expected_is_arabic = bool(exp) and exp.lstrip('-').isdigit()
    boxed_is_roman = all(c in _ROMAN_LETTERS for c in stripped.upper()) and bool(stripped)
    boxed_is_arabic = stripped.lstrip('-').isdigit()

    if expected_is_roman:
        format_ok = boxed_is_roman
    elif expected_is_arabic:
        format_ok = boxed_is_arabic
    else:
        format_ok = boxed_is_roman or boxed_is_arabic
    passed = in_trace and format_ok
    return {
        'name': 'boxed_format',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': (
            f"boxed='{stripped}' in_trace={in_trace} "
            f"exp_roman={expected_is_roman} exp_arabic={expected_is_arabic} format_ok={format_ok}"
        ),
    }


# ----------------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------------
def grade_chain(prompt: str, expected: str, trace: str, boxed: str) -> dict:
    trace = trace or ''
    boxed = boxed or ''

    target_a, target_r, direction = parse_prompt(prompt or '')

    steps = [
        _step_identify_type(trace),
        _step_direction_detected(trace, direction),
        _step_conversion_logic(trace),
        _step_applied_to_target(trace, target_a, target_r),
        _step_boxed_format(trace, boxed, expected),
    ]

    pass_count = sum(1 for s in steps if s['passed'])
    fail_count = len(steps) - pass_count
    first_failure = next((s['name'] for s in steps if not s['passed']), None)
    chain_score = pass_count / len(steps)

    return {
        'type': 'roman',
        'steps': steps,
        'first_failure': first_failure,
        'pass_count': pass_count,
        'fail_count': fail_count,
        'chain_score': chain_score,
    }


if __name__ == '__main__':
    test_prompt = (
        "In Alice's Wonderland, numbers are secretly converted into a different numeral system. "
        "Some examples are given below:\n"
        "78 -> LXXVIII\n30 -> XXX\n60 -> LX\n86 -> LXXXVI\n92 -> XCII\n"
        "Now, write the number 32 in the Wonderland numeral system."
    )
    trace = (
        "DECOMPOSE:\nn = 32\nTH = 0 segment = SKIP\nTE = 30 segment = XXX\nON = 2 segment = II\n"
        "CAT: XXXII\nVER: re-parse XXXII back to 32 PASS\n\\boxed{XXXII}"
    )
    r = grade_chain(test_prompt, 'XXXII', trace, 'XXXII')
    print(f"chain_score={r['chain_score']:.2f} first_fail={r['first_failure']}")
    for s in r['steps']:
        mk = '[P]' if s['passed'] else '[F]'
        print(f"  {mk} {s['name']:<22} {s['detail']}")
