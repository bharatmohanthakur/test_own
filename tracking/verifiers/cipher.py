"""Cipher verifier: substitution cipher on text in Alice's Wonderland.

Ground truth derivation:
- Parse each example pair "encrypted -> decrypted"
- Align word-by-word (substitution preserves length), char-by-char
- Build a deterministic letter-map from aligned positions
- The "target" line is "Now, decrypt the following text: <encrypted>"

Model-facts extraction:
- Look for a TABLE: section listing letter mappings (e.g., "b → p", "b -> p")
- Look for VER: PASS/FAIL / CHECK: claims
- Look for "VOCAB fill" markers
- Boxed answer is the final decryption

Grading:
- table_accuracy: fraction of model's table entries that match ground truth
- ver_honesty: PASS_TRUE / PASS_LIE / FAIL_TRUE / FAIL_FALSE / NONE
- vocab_fill_used / vocab_fill_abused
- per_word_accuracy on final answer
- failure_mode: one of {correct, template_with_wrong_table, template_with_lying_ver,
                       free_form_wrong_mapping, hallucinated_vocab, truncated, no_boxed}
"""
import re
from typing import Optional


# ----------------------------------------------------------------------------
# Prompt parsing
# ----------------------------------------------------------------------------
def parse_prompt(prompt: str):
    """Extract (example_pairs, target_encrypted) from cipher prompt."""
    lines = [l.strip() for l in prompt.split('\n') if l.strip()]
    pairs = []
    target = None
    for line in lines:
        if '->' in line:
            left, right = line.split('->', 1)
            left = left.strip()
            right = right.strip()
            # Skip the header line "input -> output"
            if 'input' in left.lower() or 'example' in left.lower():
                continue
            pairs.append((left, right))
        elif 'decrypt the following' in line.lower() or 'decrypt:' in line.lower():
            # "Now, decrypt the following text: xyz"
            m = re.search(r'decrypt[^:]*:\s*(.+)', line, re.IGNORECASE)
            if m:
                target = m.group(1).strip()
    return pairs, target


def derive_true_table(pairs):
    """Build letter substitution map from aligned example pairs.

    Returns a dict {encrypted_char: decrypted_char}. If a char has conflicting
    mappings across examples, the most common wins (shouldn't happen in a clean cipher).
    """
    from collections import Counter
    maps = {}  # char -> Counter(dst → count)
    for enc, dec in pairs:
        enc_words = enc.split()
        dec_words = dec.split()
        if len(enc_words) != len(dec_words):
            continue
        for ew, dw in zip(enc_words, dec_words):
            if len(ew) != len(dw):
                continue
            for ec, dc in zip(ew.lower(), dw.lower()):
                if not (ec.isalpha() and dc.isalpha()):
                    continue
                maps.setdefault(ec, Counter())[dc] += 1
    true_table = {k: c.most_common(1)[0][0] for k, c in maps.items()}
    return true_table


def apply_table(encrypted: str, table: dict) -> str:
    """Apply a substitution table; unknown chars become '?'."""
    out = []
    for c in encrypted:
        if c.isalpha():
            out.append(table.get(c.lower(), '?'))
        else:
            out.append(c)
    return ''.join(out)


# ----------------------------------------------------------------------------
# Model-facts extraction
# ----------------------------------------------------------------------------
def extract_model_table(trace: str) -> dict:
    """Parse model's emitted substitution table.

    Matches patterns like:
      b → p
      b -> p
      b: p
    Also handles block starting with 'TABLE' or 'SUBSTITUTION'.
    """
    table = {}
    # Look for all "letter -> letter" or "letter → letter" style mappings
    # Restrict to single-letter pairs
    for m in re.finditer(r'(?:^|\n|\s)([a-zA-Z])\s*(?:->|→|:|=)\s*([a-zA-Z])\b', trace):
        src, dst = m.group(1).lower(), m.group(2).lower()
        if src not in table:  # first mapping wins
            table[src] = dst
    return table


def extract_ver_claim(trace: str) -> Optional[str]:
    """Return 'PASS' / 'FAIL' / None based on model's VER/CHECK statements."""
    # Look near VER: or CHECK: blocks
    for m in re.finditer(r'(VER|CHECK|VERIFY)[^:]{0,40}:\s*([^\n]{0,200})', trace, re.IGNORECASE):
        text = m.group(2).lower()
        if 'pass' in text or '✓' in text or 'correct' in text or 'matches' in text:
            return 'PASS'
        if 'fail' in text or 'wrong' in text or 'no match' in text or 'mismatch' in text:
            return 'FAIL'
    return None


def extract_vocab_fill(trace: str) -> bool:
    """Did the model use 'VOCAB fill' style rescue?"""
    return bool(re.search(r'vocab\s*fill|vocabulary\s*(fill|match|member)', trace, re.IGNORECASE))


# ----------------------------------------------------------------------------
# Grading
# ----------------------------------------------------------------------------
def grade_table(model_table: dict, true_table: dict) -> dict:
    """Compare model's table to ground truth."""
    if not model_table:
        return {"table_size": 0, "table_correct": 0, "table_wrong": 0,
                "table_accuracy": None, "table_emitted": False}
    correct = sum(1 for k, v in model_table.items() if true_table.get(k) == v)
    wrong = sum(1 for k, v in model_table.items() if k in true_table and true_table[k] != v)
    return {
        "table_size": len(model_table),
        "table_correct": correct,
        "table_wrong": wrong,
        "table_accuracy": correct / max(1, len(model_table)),
        "table_emitted": True,
    }


def check_ver_honesty(ver_claim: Optional[str], model_table: dict, pairs) -> str:
    """Is the model's VER claim justified by its own table?

    Returns one of: NONE, PASS_TRUE, PASS_LIE, FAIL_TRUE, FAIL_FALSE.
    Applies the model's table to the first example; compares to expected.
    """
    if ver_claim is None or not pairs or not model_table:
        return "NONE"
    enc, dec = pairs[0]
    model_decoded = apply_table(enc.lower(), model_table)
    # Count how many chars match
    target = dec.lower()
    if len(model_decoded) != len(target):
        actual_pass = False
    else:
        matches = sum(1 for a, b in zip(model_decoded, target) if a == b or not a.isalpha())
        actual_pass = matches >= 0.9 * len(target)

    if ver_claim == 'PASS':
        return "PASS_TRUE" if actual_pass else "PASS_LIE"
    else:  # FAIL
        return "FAIL_FALSE" if actual_pass else "FAIL_TRUE"


def per_word_accuracy(predicted: str, expected: str) -> float:
    pw = predicted.split()
    ew = expected.split()
    if not ew:
        return 0.0
    if len(pw) != len(ew):
        return 0.0
    return sum(1 for a, b in zip(pw, ew) if a.lower() == b.lower()) / len(ew)


# ----------------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------------
def verify(prompt: str, expected: str, trace: str, boxed: str) -> dict:
    pairs, target = parse_prompt(prompt)
    true_table = derive_true_table(pairs)

    model_table = extract_model_table(trace)
    ver_claim = extract_ver_claim(trace)
    vocab_fill = extract_vocab_fill(trace)

    table_grade = grade_table(model_table, true_table)
    honesty = check_ver_honesty(ver_claim, model_table, pairs)
    word_acc = per_word_accuracy(boxed, expected)
    correct = (boxed.strip().lower() == expected.strip().lower())

    # Derive failure mode
    if correct:
        failure_mode = "correct"
    elif not boxed:
        failure_mode = "no_boxed"
    elif table_grade["table_emitted"] and honesty == "PASS_LIE":
        failure_mode = "template_with_lying_ver"
    elif table_grade["table_emitted"] and table_grade["table_accuracy"] is not None and table_grade["table_accuracy"] < 0.8:
        failure_mode = "template_with_wrong_table"
    elif vocab_fill and word_acc < 1.0:
        failure_mode = "hallucinated_vocab"
    elif not table_grade["table_emitted"]:
        failure_mode = "free_form_wrong_mapping"
    else:
        failure_mode = "other_wrong"

    return {
        "correct": correct,
        "failure_mode": failure_mode,
        "ground_truth": {
            "n_examples": len(pairs),
            "true_table_size": len(true_table),
            "target_encrypted": target,
            "true_decryption": expected,
        },
        "model_facts": {
            "table_emitted": table_grade["table_emitted"],
            "ver_claim": ver_claim,
            "vocab_fill_used": vocab_fill,
            "boxed": boxed,
        },
        "quality_flags": {
            **table_grade,
            "ver_honesty": honesty,
            "per_word_accuracy": word_acc,
            "vocab_fill_used": vocab_fill,
        },
    }


if __name__ == '__main__':
    # Quick self-test
    test_prompt = (
        "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:\n"
        "ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley\n"
        "bxo sfjpov pqrsfv dfjjfig -> the golden dragon follows\n"
        "Now, decrypt the following text: trb wzrswvog hffk"
    )
    pairs, target = parse_prompt(test_prompt)
    print(f"pairs: {len(pairs)}, target: {target!r}")
    tt = derive_true_table(pairs)
    print(f"true table entries: {len(tt)}")
    print(f"first 10: {dict(list(tt.items())[:10])}")
    # Apply to target
    print(f"decoded target: {apply_table(target, tt)!r}")