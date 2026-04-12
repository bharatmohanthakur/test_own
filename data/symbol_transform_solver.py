"""Symbol_transform (CIPHER-DIGIT equation) solver.

Per Donald Galliano III: operator char at pos 2 is COSMETIC. All examples in a
puzzle share ONE combo (order × op × fmt). The challenge is cracking the
symbol→digit permutation jointly with identifying the combo.

Algorithm:
  1. Parse examples: (operand pair 1, operator, operand pair 2) = result_string
  2. Extract all distinct symbols → must map 1:1 to digits 0-9 (at most 10 symbols)
  3. For each combo in 47 candidates:
     a. For each permutation of symbols → digits (or DFS with propagation):
        - Decode operand pairs into numeric
        - Apply combo
        - Encode result with the candidate map
        - Check it matches the given result string
     b. If all examples consistent → found (combo, map)
  4. Apply the combo + map to the target

Optimization: instead of trying 10! permutations, use equation-level constraint
propagation. For each combo hypothesis, fix the operand digits from one example
and check consistency on others.

Test against all 823 CIPHER-DIGIT rows in train.csv.
"""
import csv
import itertools
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tracking"))
from train_loader import load_train

ORDERS = ["AB_CD", "BA_DC"]
OPS = ["add", "sub", "mul", "cat", "rcat", "mul_add1", "mul_sub1", "addm1", "add1"]
FMTS = ["raw", "rev", "abs", "rev_abs", "zpad4"]


def parse_prompt(prompt: str):
    pairs = []
    target = None
    for line in prompt.split("\n"):
        line = line.strip()
        if "=" in line and "example" not in line.lower() and "determine" not in line.lower():
            left, _, right = line.partition("=")
            left = left.strip()
            right = right.strip()
            if len(left) == 5:
                pairs.append((left, right))
        m = re.search(r"determine the (?:result|output) for[:\s]+(.{5})\s*$", line, re.IGNORECASE)
        if m:
            target = m.group(1)
    return pairs, target


def apply_op(s1: str, s2: str, op: str, fmt: str):
    """Apply op to decoded digit strings s1, s2. Return result string or None."""
    if not (s1.isdigit() and s2.isdigit()):
        return None
    a, b = int(s1), int(s2)
    if op == "add":      result = a + b
    elif op == "sub":    result = a - b
    elif op == "mul":    result = a * b
    elif op == "cat":    return s1 + s2 if fmt == "raw" else (s2 + s1 if fmt == "rev" else None)
    elif op == "rcat":   return s2 + s1 if fmt == "raw" else (s1 + s2 if fmt == "rev" else None)
    elif op == "mul_add1": result = a * b + 1
    elif op == "mul_sub1": result = a * b - 1
    elif op == "addm1":  result = a + b - 1
    elif op == "add1":   result = a + b + 1
    else: return None

    if fmt == "raw":     return str(result)
    if fmt == "rev":     return str(result)[::-1]
    if fmt == "abs":     return str(abs(result))
    if fmt == "rev_abs": return str(abs(result))[::-1]
    if fmt == "zpad4":
        if result < 0: return None
        return str(result).zfill(4)
    return None


def extract_operands(left: str, order: str):
    """Return (pair1_chars, pair2_chars) based on order."""
    c0, c1, _, c3, c4 = left
    if order == "AB_CD":
        return c0 + c1, c3 + c4
    else:  # BA_DC
        return c1 + c0, c4 + c3


def try_solve_puzzle(pairs, target):
    """Try every (order, op, fmt) combo. For each, search for a symbol→digit map
    consistent with all 4+ examples. Return (combo, map, target_answer) or None."""
    if not pairs or target is None:
        return None

    # All distinct symbols appearing as OPERANDS (LHS positions 0,1,3,4) across examples + target
    operand_syms = set()
    for left, _ in pairs:
        for pos in (0, 1, 3, 4):
            operand_syms.add(left[pos])
    for pos in (0, 1, 3, 4):
        operand_syms.add(target[pos])
    # Also add RHS symbols (they must appear in the digit map too — results are encoded)
    rhs_syms = set()
    for _, right in pairs:
        rhs_syms.update(right)

    all_syms = sorted(operand_syms | rhs_syms)
    if len(all_syms) > 10:
        return None  # Too many symbols — not a digit map

    for order in ORDERS:
        for op in OPS:
            for fmt in FMTS:
                result = search_map(all_syms, pairs, target, order, op, fmt)
                if result is not None:
                    return (order, op, fmt, result)
    return None


def search_map(all_syms, pairs, target, order, op, fmt):
    """Smart search: start by enumerating operand digits of the FIRST example
    (≤ 4 operand chars → ≤ 5040 assignments). For each, compute the expected
    result string, match it against the RHS symbols to extend the map, then
    verify on remaining examples."""
    if not pairs:
        return None

    first_left, first_right = pairs[0]
    p1_chars, p2_chars = extract_operands(first_left, order)
    all_first_chars = list(dict.fromkeys(p1_chars + p2_chars))  # unique in order
    n_first = len(all_first_chars)

    # Enumerate permutations of n_first chars over 10 digits
    for digits in itertools.permutations(range(10), n_first):
        assignment = dict(zip(all_first_chars, digits))
        # Decode operand pair 1 and 2
        s1 = "".join(str(assignment[c]) for c in p1_chars)
        s2 = "".join(str(assignment[c]) for c in p2_chars)
        got = apply_op(s1, s2, op, fmt)
        if got is None:
            continue
        if len(got) != len(first_right):
            continue
        # Skip negatives (can't be encoded)
        if not got.lstrip("-").isdigit() or got.startswith("-"):
            continue
        # Extend map from RHS
        extended = dict(assignment)
        consistent = True
        for i, ch in enumerate(first_right):
            d = int(got[i])
            if ch in extended:
                if extended[ch] != d:
                    consistent = False
                    break
            else:
                # Check digit not used by a DIFFERENT symbol
                if d in extended.values():
                    consistent = False
                    break
                extended[ch] = d
        if not consistent:
            continue
        # Verify on remaining examples (and any new syms they introduce)
        full_map = _verify_and_extend(extended, pairs[1:], order, op, fmt)
        if full_map is None:
            continue
        # Compute target answer
        target_answer = compute_target_safe(full_map, target, order, op, fmt)
        if target_answer is not None:
            return (full_map, target_answer)
    return None


def _verify_and_extend(assignment, remaining_pairs, order, op, fmt):
    """For each remaining pair, verify or extend the map. Return extended
    assignment if all consistent, else None."""
    cur = dict(assignment)
    for left, right in remaining_pairs:
        p1, p2 = extract_operands(left, order)
        # Check operand chars are all assigned; if not, try all digit options for unassigned
        unassigned = [c for c in set(p1 + p2) if c not in cur]
        if unassigned:
            # Try digit assignments for unassigned operand chars
            remaining_digits = [d for d in range(10) if d not in cur.values()]
            found = False
            for perm in itertools.permutations(remaining_digits, len(unassigned)):
                trial = dict(cur)
                for c, d in zip(unassigned, perm):
                    trial[c] = d
                s1 = "".join(str(trial[c]) for c in p1)
                s2 = "".join(str(trial[c]) for c in p2)
                got = apply_op(s1, s2, op, fmt)
                if got is None or len(got) != len(right):
                    continue
                if got.startswith("-") or not got.isdigit():
                    continue
                # Check/extend RHS
                trial2 = dict(trial)
                ok = True
                for i, ch in enumerate(right):
                    d = int(got[i])
                    if ch in trial2:
                        if trial2[ch] != d:
                            ok = False
                            break
                    else:
                        if d in trial2.values():
                            ok = False
                            break
                        trial2[ch] = d
                if ok:
                    cur = trial2
                    found = True
                    break
            if not found:
                return None
        else:
            s1 = "".join(str(cur[c]) for c in p1)
            s2 = "".join(str(cur[c]) for c in p2)
            got = apply_op(s1, s2, op, fmt)
            if got is None or len(got) != len(right):
                return None
            if got.startswith("-") or not got.isdigit():
                return None
            for i, ch in enumerate(right):
                d = int(got[i])
                if ch in cur:
                    if cur[ch] != d:
                        return None
                else:
                    if d in cur.values():
                        return None
                    cur[ch] = d
    return cur


def compute_target_safe(assignment, target, order, op, fmt):
    p1, p2 = extract_operands(target, order)
    missing = [c for c in set(p1 + p2) if c not in assignment]
    if missing:
        return None
    return compute_target(assignment, target, order, op, fmt)


def _partial_check(assignment, pairs, order, op, fmt):
    """Fast fail if any PAIR has all operand symbols assigned but doesn't match.
    Skip pairs where operands or result symbols are not yet assigned."""
    for left, right in pairs:
        p1, p2 = extract_operands(left, order)
        all_chars = set(p1 + p2 + right)
        if not all(c in assignment for c in all_chars):
            continue
        s1 = "".join(str(assignment[c]) for c in p1)
        s2 = "".join(str(assignment[c]) for c in p2)
        got = apply_op(s1, s2, op, fmt)
        expected = "".join(str(assignment[c]) for c in right)
        if got != expected:
            return False
    return True


def compute_target(assignment, target, order, op, fmt):
    p1, p2 = extract_operands(target, order)
    s1 = "".join(str(assignment[c]) for c in p1)
    s2 = "".join(str(assignment[c]) for c in p2)
    got = apply_op(s1, s2, op, fmt)
    if got is None:
        return None
    # Encode back: need to map digit → symbol
    digit_to_sym = {d: s for s, d in assignment.items()}
    encoded = ""
    for ch in got:
        d = int(ch)
        if d not in digit_to_sym:
            return None  # Can't encode — missing symbol
        encoded += digit_to_sym[d]
    return encoded


def detect_subtype(pairs):
    for left, _ in pairs:
        ops = [left[0], left[1], left[3], left[4]]
        if not all(c.isdigit() for c in ops):
            return "CIPHER-DIGIT"
    return "SYMBOL-DIGIT"


def main():
    _, by_type, _ = load_train()
    eq_rows = by_type["equation"]

    cipher_rows = []
    for row in eq_rows:
        pairs, target = parse_prompt(row["prompt"])
        if detect_subtype(pairs) == "CIPHER-DIGIT" and target is not None:
            cipher_rows.append((row, pairs, target))

    print(f"Total CIPHER-DIGIT rows: {len(cipher_rows)}")
    print("Attempting to solve...")

    solved = 0
    correct = 0
    combo_counts = Counter()
    import time
    start = time.time()

    LIMIT = 100  # Start with small batch to measure speed
    for i, (row, pairs, target) in enumerate(cipher_rows[:LIMIT]):
        t0 = time.time()
        result = try_solve_puzzle(pairs, target)
        dt = time.time() - t0
        if result is not None:
            order, op, fmt, (mapping, target_answer) = result
            solved += 1
            combo_counts[(order, op, fmt)] += 1
            if target_answer == row["answer"]:
                correct += 1
        if (i + 1) % 10 == 0 or dt > 1.0:
            status = "✓" if result and target_answer == row["answer"] else ("~" if result else "✗")
            print(f"  [{i+1}/{LIMIT}] {status} {dt:.1f}s (solved={solved}, correct={correct})")
        if time.time() - start > 120:
            print("  TIMEOUT after 120s — cutting off")
            break

    print(f"\n=== RESULTS ===")
    print(f"Scanned: {i+1}")
    print(f"Solved (found combo+map): {solved}")
    print(f"Correct on target: {correct}")
    print(f"Elapsed: {time.time()-start:.1f}s")
    print(f"\nTop combos:")
    for combo, count in combo_counts.most_common(10):
        print(f"  {combo}: {count}")


if __name__ == "__main__":
    main()
