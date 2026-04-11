"""Binary (8-bit manipulation) verifier.

Ground truth derivation:
- Parse input -> output example pairs (each is an 8-bit binary string)
- Try to classify per-bit: for each output bit position k, check if output[k] is:
  * a constant (0 or 1)
  * a copy of some input bit (identity/shift)
  * NOT of some input bit
  * a 2-input gate (AND, OR, XOR, NAND, NOR, XNOR) over two input bits
  * None of the above → "complex" (3+ input or majority/choice)

The model's per-bit gate only needs to be consistent across all examples.
If we find a valid per-bit rule for all 8 bits, we can compute the ground-truth
answer for the target and use it to grade. Otherwise we fall back to string match.

Failure modes:
- correct
- lazy_constant        — answer = all 0s or all 1s (and wrong)
- lazy_copy            — answer = input verbatim (and wrong)
- lazy_not             — answer = NOT(input) (and wrong)
- wrong_2gate          — classifiable but model picked wrong gate
- complex_gates        — ground truth needs 3+ input gates (known hard case)
- no_boxed / format_bad
"""
import re
from typing import Optional, List, Tuple


# ----------------------------------------------------------------------------
# Prompt parsing
# ----------------------------------------------------------------------------
def parse_prompt(prompt: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    pairs = []
    target = None
    for line in prompt.split('\n'):
        line = line.strip()
        m = re.match(r'^([01]{8})\s*->\s*([01]{8})\s*$', line)
        if m:
            pairs.append((m.group(1), m.group(2)))
            continue
        m2 = re.search(r'determine the output for[:\s]+([01]{8})', line, re.IGNORECASE)
        if m2:
            target = m2.group(1)
    return pairs, target


# ----------------------------------------------------------------------------
# Per-bit gate classification
# ----------------------------------------------------------------------------
GATES = {
    'AND':  lambda a, b: a & b,
    'OR':   lambda a, b: a | b,
    'XOR':  lambda a, b: a ^ b,
    'NAND': lambda a, b: 1 - (a & b),
    'NOR':  lambda a, b: 1 - (a | b),
    'XNOR': lambda a, b: 1 - (a ^ b),
}


def classify_bit(examples: List[Tuple[str, str]], k: int) -> Optional[dict]:
    """Find a consistent rule for output bit k across all examples.

    Returns a dict describing the rule, or None if no simple rule fits.
    """
    out_bits = [int(out[k]) for _, out in examples]

    # Constant
    if all(b == out_bits[0] for b in out_bits):
        return {"kind": "CONSTANT", "value": out_bits[0]}

    # Identity: output[k] == input[j] for some j
    for j in range(8):
        if all(int(inp[j]) == out_bits[i] for i, (inp, _) in enumerate(examples)):
            return {"kind": "IDENTITY", "j": j}
        if all(1 - int(inp[j]) == out_bits[i] for i, (inp, _) in enumerate(examples)):
            return {"kind": "NOT", "j": j}

    # 2-input gates
    for j1 in range(8):
        for j2 in range(j1 + 1, 8):
            for gname, gfn in GATES.items():
                if all(gfn(int(inp[j1]), int(inp[j2])) == out_bits[i]
                       for i, (inp, _) in enumerate(examples)):
                    return {"kind": "2GATE", "gate": gname, "j1": j1, "j2": j2}

    # 3-input majority / choice (hard but common)
    for j1 in range(8):
        for j2 in range(j1 + 1, 8):
            for j3 in range(j2 + 1, 8):
                # majority(a,b,c) = (a & b) | (a & c) | (b & c)
                def maj(a, b, c):
                    return (a & b) | (a & c) | (b & c)
                if all(maj(int(inp[j1]), int(inp[j2]), int(inp[j3])) == out_bits[i]
                       for i, (inp, _) in enumerate(examples)):
                    return {"kind": "MAJORITY", "j1": j1, "j2": j2, "j3": j3}

    return None


def solve_ground_truth(examples: List[Tuple[str, str]], target: str) -> Optional[str]:
    """Derive the true answer via per-bit classification. Returns None if unsolvable
    with simple gates."""
    if not examples or not target:
        return None
    rules = []
    for k in range(8):
        rule = classify_bit(examples, k)
        if rule is None:
            return None
        rules.append(rule)

    # Apply rules to target
    out = []
    tbits = [int(c) for c in target]
    for rule in rules:
        if rule["kind"] == "CONSTANT":
            out.append(rule["value"])
        elif rule["kind"] == "IDENTITY":
            out.append(tbits[rule["j"]])
        elif rule["kind"] == "NOT":
            out.append(1 - tbits[rule["j"]])
        elif rule["kind"] == "2GATE":
            out.append(GATES[rule["gate"]](tbits[rule["j1"]], tbits[rule["j2"]]))
        elif rule["kind"] == "MAJORITY":
            a = tbits[rule["j1"]]; b = tbits[rule["j2"]]; c = tbits[rule["j3"]]
            out.append((a & b) | (a & c) | (b & c))
    return ''.join(str(x) for x in out)


# ----------------------------------------------------------------------------
# Grading
# ----------------------------------------------------------------------------
def detect_lazy_pattern(answer: str, target: str) -> Optional[str]:
    if answer == '0' * 8:
        return "lazy_constant_0"
    if answer == '1' * 8:
        return "lazy_constant_1"
    if answer == target:
        return "lazy_copy"
    if answer == ''.join('1' if c == '0' else '0' for c in target):
        return "lazy_not"
    return None


def verify(prompt: str, expected: str, trace: str, boxed: str) -> dict:
    pairs, target = parse_prompt(prompt)
    true_answer = solve_ground_truth(pairs, target) if target else None
    is_solvable_simple = true_answer is not None

    correct = (boxed.strip() == str(expected).strip())

    # Classify ground-truth difficulty
    rules = []
    if pairs:
        for k in range(8):
            r = classify_bit(pairs, k)
            rules.append(r["kind"] if r else "COMPLEX")
    difficulty = "SIMPLE" if all(r in ("CONSTANT", "IDENTITY", "NOT", "2GATE", "MAJORITY") for r in rules) else "HARD"

    if correct:
        failure_mode = "correct"
    elif not boxed:
        failure_mode = "no_boxed"
    elif not re.fullmatch(r'[01]{8}', boxed.strip()):
        failure_mode = "format_bad"
    elif difficulty == "HARD":
        failure_mode = "complex_gates"
    else:
        lazy = detect_lazy_pattern(boxed.strip(), target or '')
        if lazy:
            failure_mode = lazy
        else:
            failure_mode = "wrong_2gate"

    return {
        "correct": correct,
        "failure_mode": failure_mode,
        "ground_truth": {
            "n_examples": len(pairs),
            "target": target,
            "true_answer": true_answer,
            "difficulty": difficulty,
            "rules_per_bit": rules,
        },
        "model_facts": {
            "boxed": boxed,
        },
        "quality_flags": {
            "solvable_with_simple_gates": is_solvable_simple,
            "answer_matches_truth": (true_answer is not None and boxed.strip() == true_answer),
        },
    }


if __name__ == '__main__':
    test_prompt = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.

Here are some examples of input -> output:
00000000 -> 11111111
11111111 -> 00000000
10101010 -> 01010101
01010101 -> 10101010

Now, determine the output for: 11110000"""
    r = verify(test_prompt, "00001111", "<trace>", "00001111")
    print("simple NOT test:", r["correct"], r["failure_mode"], r["ground_truth"]["rules_per_bit"][:3])
