"""Clean Z3-based solver for CIPHER-DIGIT equation_transform puzzles.

Key insight: operator char at LHS[2] is cosmetic (per Donald Galliano). All
examples in a puzzle share ONE operation (combo). Non-operator chars (pos 0, 1,
3, 4 on LHS + all RHS chars) form a bijective map to digits 0-9.

Uses Z3's Distinct constraint to enforce bijection. Tries a compact combo set
and returns on first sat with a verified target answer.
"""
import csv
import re
import sys
import time
from collections import Counter
from pathlib import Path

from z3 import Int, Solver, Distinct, sat

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tracking"))
from train_loader import load_train


ORDERS = [
    (0, 1, 3, 4),  # AB_CD
    (1, 0, 4, 3),  # BA_DC
    (0, 1, 4, 3),  # AB_DC
    (1, 0, 3, 4),  # BA_CD
]
OPS = ["add", "sub", "rsub", "mul", "cat", "rcat", "mul_add1", "mul_sub1", "addm1", "add1", "sub1", "rsub1"]
FMTS = ["raw", "rev"]


def parse_prompt(prompt):
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


def detect_cipher_digit(pairs):
    for left, _ in pairs:
        for pos in (0, 1, 3, 4):
            if not left[pos].isdigit():
                return True
    return False


def solve_one(pairs, target, time_ms=2000):
    """Return (predicted_answer, (order, op, fmt), map) or (None, None, None)."""
    if not pairs or target is None:
        return None, None, None

    # Collect all non-operator chars (operand positions + all RHS)
    chars = set()
    for left, right in pairs:
        for pos in (0, 1, 3, 4):
            chars.add(left[pos])
        chars.update(right)
    for pos in (0, 1, 3, 4):
        chars.add(target[pos])
    chars = sorted(chars)

    if len(chars) > 10:
        return None, None, None  # Can't map injectively

    for order in ORDERS:
        ai, aj, bi, bj = order
        for op_name in OPS:
            for fmt in FMTS:
                s = Solver()
                s.set("timeout", time_ms)
                d = {c: Int(f"d_{ord(c)}") for c in chars}
                for v in d.values():
                    s.add(v >= 0, v <= 9)
                if len(chars) <= 10:
                    s.add(Distinct(list(d.values())))

                bad = False
                for left, right in pairs:
                    a = d[left[ai]] * 10 + d[left[aj]]
                    b = d[left[bi]] * 10 + d[left[bj]]
                    if op_name == "add":   res = a + b
                    elif op_name == "sub": res = a - b
                    elif op_name == "rsub":res = b - a
                    elif op_name == "mul": res = a * b
                    elif op_name == "cat": res = a * 100 + b
                    elif op_name == "rcat":res = b * 100 + a
                    elif op_name == "mul_add1": res = a * b + 1
                    elif op_name == "mul_sub1": res = a * b - 1
                    elif op_name == "addm1":    res = a + b - 1
                    elif op_name == "add1":     res = a + b + 1
                    elif op_name == "sub1":     res = a - b - 1
                    elif op_name == "rsub1":    res = b - a - 1

                    # Build RHS expected value from cipher chars
                    rhs_seq = right[::-1] if fmt == "rev" else right
                    if not rhs_seq:
                        bad = True
                        break
                    rhs_val = 0
                    for c in rhs_seq:
                        rhs_val = rhs_val * 10 + d[c]
                    s.add(res == rhs_val)
                if bad:
                    continue

                if s.check() != sat:
                    continue
                m = s.model()
                dmap = {c: m[d[c]].as_long() for c in chars}

                # Compute target answer
                a = dmap[target[ai]] * 10 + dmap[target[aj]]
                b = dmap[target[bi]] * 10 + dmap[target[bj]]
                if op_name == "add":   qr = a + b
                elif op_name == "sub": qr = a - b
                elif op_name == "rsub":qr = b - a
                elif op_name == "mul": qr = a * b
                elif op_name == "cat": qr = a * 100 + b
                elif op_name == "rcat":qr = b * 100 + a
                elif op_name == "mul_add1": qr = a * b + 1
                elif op_name == "mul_sub1": qr = a * b - 1
                elif op_name == "addm1":    qr = a + b - 1
                elif op_name == "add1":     qr = a + b + 1
                elif op_name == "sub1":     qr = a - b - 1
                elif op_name == "rsub1":    qr = b - a - 1
                if qr < 0:
                    continue
                inv = {v: k for k, v in dmap.items()}
                digit_str = str(qr)
                out = ""
                ok = True
                for digit in digit_str:
                    dval = int(digit)
                    if dval not in inv:
                        ok = False
                        break
                    out += inv[dval]
                if not ok:
                    continue
                if fmt == "rev":
                    out = out[::-1]
                return out, (order, op_name, fmt), dmap
    return None, None, None


def main():
    _, by_type, _ = load_train()
    eqs = by_type["equation"]
    cd_rows = []
    for row in eqs:
        pairs, target = parse_prompt(row["prompt"])
        if target and detect_cipher_digit(pairs):
            cd_rows.append((row, pairs, target))
    print(f"CIPHER-DIGIT rows: {len(cd_rows)}")

    correct = wrong = nosol = 0
    combos = Counter()
    t0 = time.time()
    LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    for i, (row, pairs, target) in enumerate(cd_rows[:LIMIT]):
        pred, combo, _ = solve_one(pairs, target)
        if pred is None:
            nosol += 1
        elif pred == row["answer"].strip():
            correct += 1
            combos[combo] += 1
        else:
            wrong += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{LIMIT}] correct={correct} wrong={wrong} nosol={nosol} elapsed={time.time()-t0:.0f}s")
    print(f"\nFinal: correct={correct}, wrong={wrong}, nosol={nosol}, time={time.time()-t0:.1f}s")
    print(f"Top combos: {combos.most_common(5)}")


if __name__ == "__main__":
    main()
