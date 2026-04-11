"""Binary (8-bit manipulation) reasoning chain grader.

Scores each STEP of the model's reasoning for binary puzzles:

    1. identify_type   - does the trace even know this is bit manipulation?
    2. parse_examples  - did it acknowledge the example pairs?
    3. per_bit_tried   - did it do actual per-bit analysis (KEY v22-hallucinator detector)
    4. rule_derived    - did it commit to a rule per bit (IDENTITY/NOT/AND/OR/XOR/...)?
    5. consistency_ok  - did it claim the rule "matches all examples"?
    6. rule_applied    - did it apply the rule to the target input?
    7. valid_binary    - is the boxed answer 8 chars of 0/1?
    8. boxed_format    - is `\\boxed{...}` actually present and non-empty?

Cross-checks the ground-truth answer via the existing verifier solver:
 - If we can solve the puzzle with simple gates and boxed matches the true answer,
   consistency_ok is treated as a definitive pass.
 - valid_binary reports Hamming distance to the true answer when known.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

# Ensure tracking/ is on sys.path so `verifiers` resolves whether this module
# is imported from within tracking or from elsewhere.
_TRACKING_DIR = str(Path(__file__).resolve().parent.parent)
if _TRACKING_DIR not in sys.path:
    sys.path.insert(0, _TRACKING_DIR)

from verifiers.binary import parse_prompt, solve_ground_truth, classify_bit  # noqa: E402,F401


# ----------------------------------------------------------------------------
# Keyword heuristics
# ----------------------------------------------------------------------------
TYPE_MARKERS = (
    "bit manipulation",
    "bit-manipulation",
    "binary",
    "8-bit",
    "8 bit",
    "per bit",
    "per-bit",
    "bit-by-bit",
    "bit by bit",
    "bit-serial",
    "bit serial",
)

GATE_MARKERS = (
    "IDENTITY",
    "identity",
    "CONSTANT",
    "constant",
    "AND",
    "NAND",
    "OR",
    "NOR",
    "XOR",
    "XNOR",
    "NOT",
    "MAJORITY",
    "majority",
    "input bit",
    "input[",
    "input_bit",
)

CONSISTENCY_PATTERNS = (
    r"matches?\s+all\s+examples?",
    r"fits?\s+every\s+example",
    r"fits?\s+all\s+examples?",
    r"holds?\s+for\s+all\s+examples?",
    r"consistent\s+(?:with|across)\s+(?:all\s+)?examples?",
    r"verified\s+(?:on|against|across)\s+(?:all\s+)?examples?",
    r"all\s+examples?\s+(?:match|agree|check|pass|verified)",
    r"every\s+example\s+(?:matches|fits|agrees|passes)",
    r"works?\s+for\s+all\s+examples?",
    r"satisfies\s+all\s+examples?",
    r"VER\s*[:=]?\s*(?:PASS|OK|YES|MATCH)",
)


def _count_per_bit_mentions(trace: str) -> int:
    """Count explicit per-bit references like 'bit[0]', 'bit 3', 'bit_5', 'position 7'."""
    patterns = [
        r"\bbit\s*\[\s*[0-7]\s*\]",        # bit[0]
        r"\bbit\s+[0-7]\b",                # bit 0
        r"\bbit_?[0-7]\b",                 # bit_0, bit0
        r"\bposition\s+[0-7]\b",           # position 0
        r"\bpos\s*\[\s*[0-7]\s*\]",        # pos[0]
        r"\bB[0-7]\s*[:=]",                # B0:, B7=
        r"\bout\s*\[\s*[0-7]\s*\]",        # out[0]
        r"\boutput\s*\[\s*[0-7]\s*\]",     # output[0]
        r"\bcolumn\s+[0-7]\b",             # column 0
        r"\bcol\s*\[\s*[0-7]\s*\]",        # col[0]
    ]
    total = 0
    for p in patterns:
        total += len(re.findall(p, trace, flags=re.IGNORECASE))
    return total


def _count_gate_mentions(trace: str) -> int:
    # Count occurrences of each gate/kind word. Case-sensitive for the ALL-CAPS
    # canonical labels, case-insensitive for free-text "identity", "not", etc.
    total = 0
    canonical = ["IDENTITY", "CONSTANT", "AND", "NAND", "OR", "NOR", "XOR", "XNOR", "NOT", "MAJORITY"]
    for word in canonical:
        # require word boundary; match plain uppercase token OR lowercase variant
        total += len(re.findall(rf"\b{word}\b", trace))
        total += len(re.findall(rf"\b{word.lower()}\b", trace))
    # "input bit", "input[", "input_bit"
    total += len(re.findall(r"\binput\s+bit\b", trace, flags=re.IGNORECASE))
    total += len(re.findall(r"\binput\s*\[", trace, flags=re.IGNORECASE))
    total += len(re.findall(r"\binput_bit\b", trace, flags=re.IGNORECASE))
    return total


def _consistency_language(trace: str) -> bool:
    for pat in CONSISTENCY_PATTERNS:
        if re.search(pat, trace, flags=re.IGNORECASE):
            return True
    return False


def _rule_applied(trace: str, target: Optional[str]) -> bool:
    if not target:
        return False
    if target not in trace:
        return False
    application_markers = ("apply", "result", "target", "compute")
    trace_lower = trace.lower()
    return any(m in trace_lower for m in application_markers)


def _hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def grade_chain(prompt: str, expected: str, trace: str, boxed: str) -> dict:
    trace = trace or ""
    boxed = boxed or ""
    trace_lower = trace.lower()

    pairs, target = parse_prompt(prompt or "")
    true_answer = None
    try:
        true_answer = solve_ground_truth(pairs, target) if target else None
    except Exception:
        true_answer = None

    steps = []

    # ---- Step 1: identify_type ----------------------------------------------------
    type_hits = [m for m in TYPE_MARKERS if m in trace_lower]
    passed = len(type_hits) > 0
    steps.append({
        "name": "identify_type",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "detail": f"type_markers={len(type_hits)}" + (f" ({type_hits[0]!r})" if type_hits else ""),
    })

    # ---- Step 2: parse_examples ---------------------------------------------------
    example_inputs = [p[0] for p in pairs]
    present = sum(1 for inp in example_inputs if inp in trace)
    passed = present >= 3
    steps.append({
        "name": "parse_examples",
        "passed": passed,
        "score": (min(present, len(example_inputs)) / max(len(example_inputs), 1)) if example_inputs else 0.0,
        "detail": f"example_inputs_in_trace={present}/{len(example_inputs)}",
    })

    # ---- Step 3: per_bit_tried (THE KEY METRIC) -----------------------------------
    per_bit_count = _count_per_bit_mentions(trace)
    passed = per_bit_count >= 4
    steps.append({
        "name": "per_bit_tried",
        "passed": passed,
        "score": min(per_bit_count / 8.0, 1.0),
        "detail": f"per_bit count={per_bit_count} (need >=4)",
    })

    # ---- Step 4: rule_derived -----------------------------------------------------
    gate_count = _count_gate_mentions(trace)
    passed = gate_count >= 4
    steps.append({
        "name": "rule_derived",
        "passed": passed,
        "score": min(gate_count / 8.0, 1.0),
        "detail": f"gate_keyword count={gate_count} (need >=4)",
    })

    # ---- Step 5: consistency_ok ---------------------------------------------------
    has_language = _consistency_language(trace)
    answer_matches_truth = (true_answer is not None and boxed.strip() == true_answer)
    passed = has_language or answer_matches_truth
    if answer_matches_truth:
        detail = "answer==ground_truth (definitive pass)"
        score = 1.0
    elif has_language:
        detail = "verification language found"
        score = 1.0
    else:
        detail = "no verification language, answer != ground_truth"
        score = 0.0
    steps.append({
        "name": "consistency_ok",
        "passed": passed,
        "score": score,
        "detail": detail,
    })

    # ---- Step 6: rule_applied -----------------------------------------------------
    passed = _rule_applied(trace, target)
    target_in = target is not None and target in trace
    app_marker = any(m in trace_lower for m in ("apply", "result", "target", "compute"))
    steps.append({
        "name": "rule_applied",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "detail": f"target_in_trace={target_in}, application_markers={app_marker}",
    })

    # ---- Step 7: valid_binary -----------------------------------------------------
    stripped = boxed.strip()
    is_valid_binary = bool(re.fullmatch(r"[01]{8}", stripped))
    if is_valid_binary:
        passed = True
        if true_answer is not None:
            dist = _hamming(stripped, true_answer)
            score = 1.0  # valid format is a pass; we still report hamming in detail
            detail = f"valid_binary, hamming={dist}/8"
        else:
            score = 1.0
            detail = "valid_binary, ground_truth unknown"
    else:
        passed = False
        score = 0.0
        if true_answer is not None and stripped:
            # Try to compute hamming against truth if both are 8 chars
            if len(stripped) == 8:
                dist = _hamming(stripped, true_answer)
                score = max(0.0, 1.0 - dist / 8.0)
                detail = f"not pure [01]{{8}} ({stripped!r}), hamming={dist}/8"
            else:
                detail = f"not pure [01]{{8}} ({stripped!r})"
        else:
            detail = f"not pure [01]{{8}} ({stripped!r})"
    steps.append({
        "name": "valid_binary",
        "passed": passed,
        "score": score,
        "detail": detail,
    })

    # ---- Step 8: boxed_format -----------------------------------------------------
    boxed_in_trace = bool(re.search(r"\\boxed\s*\{", trace))
    passed = boxed_in_trace and bool(stripped)
    steps.append({
        "name": "boxed_format",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "detail": f"\\boxed{{}}_in_trace={boxed_in_trace}, boxed_nonempty={bool(stripped)}",
    })

    # ---- Aggregate ----------------------------------------------------------------
    pass_count = sum(1 for s in steps if s["passed"])
    fail_count = len(steps) - pass_count
    chain_score = pass_count / len(steps)

    first_failure = None
    for s in steps:
        if not s["passed"]:
            first_failure = s["name"]
            break

    return {
        "type": "binary",
        "steps": steps,
        "first_failure": first_failure,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "chain_score": chain_score,
    }


if __name__ == "__main__":
    test_prompt = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.

Here are some examples of input -> output:
00000000 -> 11111111
11111111 -> 00000000
10101010 -> 01010101
01010101 -> 10101010

Now, determine the output for: 11110000"""
    trace = """PIPELINE: per-bit scan.
bit[0]: NOT of input[0]. matches all examples.
bit[1]: NOT of input[1].
bit[2]: NOT of input[2].
bit[3]: NOT of input[3].
Apply to target 11110000 → result 00001111.
\\boxed{00001111}"""
    r = grade_chain(test_prompt, "00001111", trace, "00001111")
    print("chain_score:", r["chain_score"])
    for s in r["steps"]:
        print(f"  {'OK' if s['passed'] else '--'} {s['name']:<18} {s['score']:.2f}  {s['detail']}")
