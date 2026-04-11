"""Equation-transform reasoning-chain grader.

Scores each step of the model's reasoning on equation-transform puzzles so we
can localize where an adapter broke. The KEY signal this grader exists to
detect is ``cipher_template_misfire`` — v30's failure mode where the model
runs a LEN/TABLE/DECRYPT pipeline on an arithmetic puzzle.

Uses helpers from ``tracking/verifiers/equation.py`` for ground truth.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Ensure tracking/ is on sys.path so `verifiers` resolves regardless of caller.
_TRACKING_DIR = str(Path(__file__).resolve().parent.parent)
if _TRACKING_DIR not in sys.path:
    sys.path.insert(0, _TRACKING_DIR)

from verifiers.equation import (  # noqa: E402
    parse_prompt,
    detect_subtype,
)


# ----------------------------------------------------------------------------
# Step 1: identify_type
# ----------------------------------------------------------------------------
_IDENTIFY_TERMS = (
    'transformation rule', 'transformation rules',
    'operator', 'operand', 'arithmetic',
    'transform', 'operation', 'rule',
    # Template-style engagement markers (v30 adapter uses these instead of
    # natural language puzzle identification):
    'example',
)


def _step_identify_type(trace: str) -> dict:
    tl = trace.lower()
    hits = [t for t in _IDENTIFY_TERMS if t in tl]
    passed = len(hits) > 0
    return {
        'name': 'identify_type',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': (
            f"found terms {hits[:4]}"
            if passed
            else "no transformation/operator/operand/arithmetic terms in trace"
        ),
    }


# ----------------------------------------------------------------------------
# Step 2: parse_operands
# ----------------------------------------------------------------------------
def _step_parse_operands(trace: str, pairs) -> dict:
    if not pairs:
        return {
            'name': 'parse_operands',
            'passed': False,
            'score': 0.0,
            'detail': 'no example pairs parsed from prompt',
        }
    found = 0
    for inp, _ in pairs:
        if inp in trace:
            found += 1
    # Loose threshold: model needs to reference at least one example input.
    # Being strict here would hide the stronger step 5 cipher-template signal.
    passed = found >= 1
    score = min(1.0, found / max(1, len(pairs)))
    return {
        'name': 'parse_operands',
        'passed': passed,
        'score': score,
        'detail': f"{found}/{len(pairs)} example inputs referenced in trace",
    }


# ----------------------------------------------------------------------------
# Step 3: detect_subtype
# ----------------------------------------------------------------------------
def _step_detect_subtype(trace: str, pairs) -> dict:
    subtype = detect_subtype(pairs) if pairs else 'UNKNOWN'
    tl = trace.lower()
    # Digit-vs-symbol awareness terms
    digit_terms = ['digit', 'number', 'numeric']
    symbol_terms = ['symbol', 'cipher', 'substitut', 'encoded', 'map']
    has_digit = any(t in tl for t in digit_terms)
    has_symbol = any(t in tl for t in symbol_terms)
    # Pass if model shows awareness of either category (symbol or digit), OR
    # references the operator character explicitly, OR operates char-by-char
    # on the operands (a v30-style engagement signal).
    operator_aware = 'operator' in tl or 'middle character' in tl
    char_aware = 'char-by-char' in tl or 'char by char' in tl or "per char" in tl
    passed = has_digit or has_symbol or operator_aware or char_aware
    return {
        'name': 'detect_subtype',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': (
            f"subtype={subtype} digit_terms={has_digit} symbol_terms={has_symbol} "
            f"operator_aware={operator_aware} char_aware={char_aware}"
        ),
    }


# ----------------------------------------------------------------------------
# Step 4: arithmetic_attempted
# ----------------------------------------------------------------------------
_ARITH_TERMS = [
    'add', 'sum', '+',
    'subtract', 'minus', '-',
    'multiply', 'product', '*', '×',
    'divide', 'quotient', '/',
    'concat', 'concatenate',
    'reverse', 'swap',
]


def _step_arithmetic_attempted(trace: str) -> dict:
    tl = trace.lower()
    hits = [t for t in _ARITH_TERMS if t in tl]
    passed = len(hits) >= 1
    return {
        'name': 'arithmetic_attempted',
        'passed': passed,
        'score': min(1.0, len(hits) / 2.0),
        'detail': f"arithmetic markers: {hits[:6]}" if hits else "no arithmetic words or operators",
    }


# ----------------------------------------------------------------------------
# Step 5: NOT_cipher_template — THE KEY METRIC for detecting v30 failure
# ----------------------------------------------------------------------------
# Check for literal cipher-template markers — these are textual signatures of
# the LEN/TABLE/DECRYPT pipeline the v30 adapter mis-applies.
_CIPHER_MARKERS = [
    'len:',
    'table:',
    'decrypt',
    'vocab fill',
    'decryption table',
    'substitution table',
    'char-by-char',
    'table extracted',
    'table (extracted',
]


def _step_not_cipher_template(trace: str) -> dict:
    tl = trace.lower()
    hits = [m for m in _CIPHER_MARKERS if m in tl]
    count = len(hits)
    # Pass if fewer than 3 markers present.
    passed = count < 3
    return {
        'name': 'NOT_cipher_template',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': (
            f"{count} cipher-template markers (threshold 3): {hits}"
            if hits
            else "no cipher-template markers detected"
        ),
    }


# ----------------------------------------------------------------------------
# Step 6: rule_committed
# ----------------------------------------------------------------------------
_RULE_COMMIT_TERMS = [
    'rule is', 'rule:', 'the rule',
    'transformation is', 'transformation:',
    'pattern is', 'pattern:',
    'operation is', 'operation:',
    'applying the', 'apply the rule',
    'identified rule', 'identifying the',
    'determined', 'lock:', 'locked',
]


def _step_rule_committed(trace: str) -> dict:
    tl = trace.lower()
    hits = [t for t in _RULE_COMMIT_TERMS if t in tl]
    passed = len(hits) >= 1
    return {
        'name': 'rule_committed',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': f"rule-commit phrases: {hits[:4]}" if hits else "no rule commitment phrase",
    }


# ----------------------------------------------------------------------------
# Step 7: target_processed
# ----------------------------------------------------------------------------
def _step_target_processed(trace: str, target) -> dict:
    if not target:
        return {
            'name': 'target_processed',
            'passed': False,
            'score': 0.0,
            'detail': 'no target extracted from prompt',
        }
    in_trace = target in trace
    return {
        'name': 'target_processed',
        'passed': in_trace,
        'score': 1.0 if in_trace else 0.0,
        'detail': f"target '{target}' {'found in' if in_trace else 'missing from'} trace",
    }


# ----------------------------------------------------------------------------
# Step 8: boxed_format
# ----------------------------------------------------------------------------
_BOXED_RE = re.compile(r'\\boxed\{[^}]*\}')


def _step_boxed_format(trace: str, boxed: str) -> dict:
    has_boxed = bool(_BOXED_RE.search(trace))
    has_answer = bool(boxed and boxed.strip())
    passed = has_boxed and has_answer
    return {
        'name': 'boxed_format',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': f"\\boxed{{}} in trace: {has_boxed}, boxed non-empty: {has_answer}",
    }


# ----------------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------------
def grade_chain(prompt: str, expected: str, trace: str, boxed: str) -> dict:
    trace = trace or ''
    boxed = boxed or ''

    pairs, target = parse_prompt(prompt or '')

    steps = [
        _step_identify_type(trace),
        _step_parse_operands(trace, pairs),
        _step_detect_subtype(trace, pairs),
        _step_arithmetic_attempted(trace),
        _step_not_cipher_template(trace),
        _step_rule_committed(trace),
        _step_target_processed(trace, target),
        _step_boxed_format(trace, boxed),
    ]

    pass_count = sum(1 for s in steps if s['passed'])
    fail_count = len(steps) - pass_count
    first_failure = next((s['name'] for s in steps if not s['passed']), None)
    chain_score = pass_count / len(steps)

    return {
        'type': 'equation',
        'steps': steps,
        'first_failure': first_failure,
        'pass_count': pass_count,
        'fail_count': fail_count,
        'chain_score': chain_score,
    }


if __name__ == '__main__':
    test_prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:\n"
        "19%38 = 3557\n"
        "21%22 = 462\n"
        "89+46 = +43\n"
        "12%21 = 252\n"
        "Now, determine the result for: 95%05"
    )
    trace_v30 = (
        "LEN:\ntarget words = ['95%05']\n"
        "TABLE (extracted from examples, char-by-char):\n"
        "  '+' -> '+'\n  '0' -> 0\n"
        "DECRYPT step...\n"
        "EX1: '19%38' -> apply table\nEX2: '21%22' -> apply table\n"
        "DECOMPOSE:\n'95%05'\nCAT: 95%05\nCHECK: length matches\nANS: 95%05\n"
        "\\boxed{95%05}"
    )
    trace_v31 = (
        "I need to find the transformation rules for these equations.\n"
        "Examples: 19%38, 21%22, 89+46, 12%21. "
        "The format is a 5-character input where the middle character acts as an operator. "
        "Applying the identified rule to 95%05: %60\n\\boxed{%60}"
    )
    for name, tr, box in [('v30', trace_v30, '95%05'), ('v31', trace_v31, '%60')]:
        r = grade_chain(test_prompt, '0592', tr, box)
        print(f"{name} chain_score={r['chain_score']:.2f} first_fail={r['first_failure']}")
        for s in r['steps']:
            mk = '[P]' if s['passed'] else '[F]'
            print(f"  {mk} {s['name']:<22} {s['detail']}")
        print()
