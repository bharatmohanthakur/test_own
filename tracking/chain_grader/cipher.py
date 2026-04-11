"""Cipher reasoning-chain grader.

Scores each step of the model's decryption reasoning so we can localize where
an adapter broke (e.g., wrong table vs. hallucinated vocab fill).

Uses helpers from tracking/verifiers/cipher.py — does not duplicate parse logic.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

# Ensure tracking/ is on sys.path so `verifiers` package resolves whether this
# module is imported from within tracking or from elsewhere.
_TRACKING_DIR = str(Path(__file__).resolve().parent.parent)
if _TRACKING_DIR not in sys.path:
    sys.path.insert(0, _TRACKING_DIR)

from verifiers.cipher import (  # noqa: E402
    parse_prompt,
    derive_true_table,
    extract_model_table,
    apply_table,
)


# ----------------------------------------------------------------------------
# Vocabulary loading (module-level, load once)
# ----------------------------------------------------------------------------
_FALLBACK_COMMON = {
    'the', 'a', 'an', 'and', 'of', 'to', 'in', 'on', 'at', 'for',
    'is', 'was', 'it', 'that', 'this', 'with', 'from', 'by', 'as', 'be',
}


def _load_vocab() -> set:
    vocab: set = set()
    path = '/Users/bharat/Downloads/kaggle/data/cipher_vocab.txt'
    try:
        with open(path, 'r') as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    vocab.add(w)
    except Exception:
        pass
    # Always merge fallback so step 7 is robust even if vocab loading failed
    vocab |= _FALLBACK_COMMON
    return vocab


_CIPHER_VOCAB: set = _load_vocab()


# ----------------------------------------------------------------------------
# Step helpers
# ----------------------------------------------------------------------------
_IDENTIFY_TERMS = ('cipher', 'decrypt', 'substitution', 'letter mapping')


def _step_identify_type(trace: str) -> dict:
    head = trace[:500].lower()
    hits = [t for t in _IDENTIFY_TERMS if t in head]
    passed = len(hits) > 0
    return {
        'name': 'identify_type',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': (
            f"found terms {hits} in first 500 chars"
            if passed
            else "none of ['cipher','decrypt','substitution','letter mapping'] in first 500 chars"
        ),
    }


def _step_parse_examples(trace: str, pairs) -> dict:
    if not pairs:
        return {
            'name': 'parse_examples',
            'passed': False,
            'score': 0.0,
            'detail': 'no example pairs parsed from prompt',
        }
    trace_lower = trace.lower()
    encrypted_words: list[str] = []
    for enc, _dec in pairs:
        for w in enc.split():
            if w:
                encrypted_words.append(w.lower())
    # Dedup but preserve order
    seen = set()
    unique_enc = []
    for w in encrypted_words:
        if w not in seen:
            seen.add(w)
            unique_enc.append(w)
    found = [w for w in unique_enc if w in trace_lower]
    passed = len(found) >= 3
    score = min(1.0, len(found) / 3.0)
    return {
        'name': 'parse_examples',
        'passed': passed,
        'score': score,
        'detail': f"{len(found)}/{len(unique_enc)} encrypted words from examples present in trace",
    }


# Single-letter to single-letter alignment patterns: x->t, x → t, x: t, x = t
_CHAR_ALIGN_RE = re.compile(
    r'(?:^|[\s,;()\[\]])([a-zA-Z])\s*(?:->|→|:|=)\s*([a-zA-Z])(?:[\s,;.()\[\]]|$)'
)


def _step_char_align(trace: str) -> dict:
    matches = _CHAR_ALIGN_RE.findall(trace)
    count = len(matches)
    passed = count >= 5
    return {
        'name': 'char_align',
        'passed': passed,
        'score': min(1.0, count / 5.0),
        'detail': f"{count} single-letter alignment patterns (need >=5)",
    }


def _step_derive_table(model_table: dict) -> dict:
    size = len(model_table)
    passed = size >= 10
    return {
        'name': 'derive_table',
        'passed': passed,
        'score': min(1.0, size / 10.0),
        'detail': f"{size} letter->letter mappings emitted (need >=10)",
    }


def _step_table_accuracy(model_table: dict, true_table: dict) -> dict:
    if not model_table:
        return {
            'name': 'table_accuracy',
            'passed': False,
            'score': 0.0,
            'detail': 'no table emitted',
        }
    if not true_table:
        return {
            'name': 'table_accuracy',
            'passed': False,
            'score': 0.0,
            'detail': 'no ground-truth table (bad prompt parse)',
        }
    correct = 0
    total = 0
    wrong_entries = []
    for k, v in model_table.items():
        if k in true_table:
            total += 1
            if true_table[k] == v:
                correct += 1
            else:
                wrong_entries.append(f"{k}->{v}(true:{true_table[k]})")
    if total == 0:
        return {
            'name': 'table_accuracy',
            'passed': False,
            'score': 0.0,
            'detail': 'model table had no overlap with true table',
        }
    score = correct / total
    passed = score >= 0.9
    detail = f"table {correct}/{total} correct"
    if wrong_entries:
        detail += f"; wrong: {wrong_entries[:4]}"
    return {
        'name': 'table_accuracy',
        'passed': passed,
        'score': score,
        'detail': detail,
    }


def _step_apply_table(prompt_target: Optional[str], boxed: str) -> dict:
    if not prompt_target:
        return {
            'name': 'apply_table',
            'passed': False,
            'score': 0.0,
            'detail': 'no target extracted from prompt',
        }
    if not boxed:
        return {
            'name': 'apply_table',
            'passed': False,
            'score': 0.0,
            'detail': 'empty boxed answer',
        }
    target_words = prompt_target.split()
    boxed_words = boxed.strip().split()
    passed = len(boxed_words) == len(target_words)
    return {
        'name': 'apply_table',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': f"boxed has {len(boxed_words)} words, target has {len(target_words)}",
    }


def _step_valid_english(boxed: str) -> dict:
    if not boxed:
        return {
            'name': 'valid_english',
            'passed': False,
            'score': 0.0,
            'detail': 'empty boxed answer',
        }
    # Strip punctuation and lowercase
    words_raw = boxed.strip().split()
    words = [re.sub(r'[^a-z]', '', w.lower()) for w in words_raw]
    words = [w for w in words if w]
    if not words:
        return {
            'name': 'valid_english',
            'passed': False,
            'score': 0.0,
            'detail': 'no alphabetic words in boxed',
        }
    valid = [w for w in words if w in _CIPHER_VOCAB]
    missing = [w for w in words if w not in _CIPHER_VOCAB]
    frac = len(valid) / len(words)
    passed = frac >= 0.8
    detail = f"{len(valid)}/{len(words)} words in vocab: valid={valid[:5]}"
    if missing:
        detail += f" missing={missing[:5]}"
    return {
        'name': 'valid_english',
        'passed': passed,
        'score': frac,
        'detail': detail,
    }


_BOXED_RE = re.compile(r'\\boxed\{[^}]*\}')


def _step_boxed_format(trace: str, boxed: str) -> dict:
    has_boxed = bool(_BOXED_RE.search(trace))
    has_answer = bool(boxed and boxed.strip())
    passed = has_boxed and has_answer
    detail = f"\\boxed{{}} in trace: {has_boxed}, boxed non-empty: {has_answer}"
    return {
        'name': 'boxed_format',
        'passed': passed,
        'score': 1.0 if passed else 0.0,
        'detail': detail,
    }


# ----------------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------------
def grade_chain(prompt: str, expected: str, trace: str, boxed: str) -> dict:
    """Grade the cipher reasoning chain.

    Returns dict matching the chain-grader contract in __init__.py.
    """
    trace = trace or ''
    boxed = boxed or ''

    pairs, target = parse_prompt(prompt or '')
    true_table = derive_true_table(pairs) if pairs else {}
    model_table = extract_model_table(trace)

    steps = [
        _step_identify_type(trace),
        _step_parse_examples(trace, pairs),
        _step_char_align(trace),
        _step_derive_table(model_table),
        _step_table_accuracy(model_table, true_table),
        _step_apply_table(target, boxed),
        _step_valid_english(boxed),
        _step_boxed_format(trace, boxed),
    ]

    pass_count = sum(1 for s in steps if s['passed'])
    fail_count = len(steps) - pass_count
    first_failure = next((s['name'] for s in steps if not s['passed']), None)
    chain_score = pass_count / len(steps)

    return {
        'type': 'cipher',
        'steps': steps,
        'first_failure': first_failure,
        'pass_count': pass_count,
        'fail_count': fail_count,
        'chain_score': chain_score,
    }


if __name__ == '__main__':
    # Quick self-test
    test_prompt = (
        "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:\n"
        "ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley\n"
        "bxo sfjpov pqrsfv dfjjfig -> the golden dragon follows\n"
        "Now, decrypt the following text: trb wzrswvog hffk"
    )
    test_trace = (
        "I'll decrypt using a substitution cipher.\n"
        "Mappings: u->q, c->u, o->e, v->n, p->d, w->i, g->s, q->r, r->a, b->c, x->h, "
        "j->l, l->y, s->o, f->g, d->n, i->w, t->o, y->v, m->m\n"
        "TABLE:\nu -> q\nc -> u\no -> e\n\\boxed{cat imagines book}"
    )
    result = grade_chain(test_prompt, "cat imagines book", test_trace, "cat imagines book")
    print("self-test chain_score:", result['chain_score'])
    for s in result['steps']:
        mk = '[P]' if s['passed'] else '[F]'
        print(f"  {mk} {s['name']:<18} score={s['score']:.2f}  {s['detail']}")
