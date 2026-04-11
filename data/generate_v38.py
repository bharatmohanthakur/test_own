"""Generate v38 training data — THK-inspired binary algorithm + Kh0a everything else.

Key innovation (from discussion 690307, Tong Hui Kang 1st place writeup):
Binary puzzles have STRUCTURED transformation rules. The output bits can usually
be matched by a sequence of (op, i_base+k, j_base+k) for k=0..7 rather than
independent rules per bit. This sequence match captures shifts and rotations
directly.

Algorithm:
  1. For each (op, i_base, j_base) triple:
     For k = 0..7: check if op(input[(i_base+k)%8], input[(j_base+k)%8])
     matches output bit k in ALL examples. Find the LONGEST run that matches.
  2. After finding the best sequence match, cover remaining bits with
     per-bit rules (Identity/NOT/Constant + 2-input gates).
  3. Generate a CoT trace showing:
     - Data tables (input, output)
     - The winning sequence + op name
     - Per-bit fallback rules for uncovered bits
     - Application to target

v38 composition:
  - 1400 binary (THK-style traces, verified-correct only)
  - 700 cipher from Kh0a (strong at 92%)
  - 400 equation from Kh0a (strong at 70%)
  - 150 each gravity/unit/roman (saturated)
  - Total: ~2800

Training config: same as v37 (Kh0a's). Expected score: 0.75-0.82 (binary from
20% → 60-85%, everything else maintained).

Usage: python3 generate_v38.py → data/training_v38/training_v38.jsonl
"""
import csv
import json
import random
import re
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tracking"))
from train_loader import load_train

TRAIN_CSV = ROOT / "data" / "train.csv"
V37_JSONL = ROOT / "data" / "training_v37" / "training_v37.jsonl"
OUT_DIR = ROOT / "data" / "training_v38"
OUT_JSONL = OUT_DIR / "training_v38.jsonl"
META_JSON = OUT_DIR / "dataset-metadata.json"

# ---------------------------------------------------------------------------
# Binary: THK-style sequence-matching solver
# ---------------------------------------------------------------------------

# Operation implementations — all take ints a, b ∈ {0,1}
OPS = {
    "AND":   lambda a, b: a & b,
    "OR":    lambda a, b: a | b,
    "XOR":   lambda a, b: a ^ b,
    "AND_N": lambda a, b: a & (1 - b),      # AND-NOT (non-commutative)
    "OR_N":  lambda a, b: a | (1 - b),
    "XOR_N": lambda a, b: a ^ (1 - b),
    "NAND":  lambda a, b: 1 - (a & b),
    "NOR":   lambda a, b: 1 - (a | b),
    "XNOR":  lambda a, b: 1 - (a ^ b),
}

UNARY = ["I", "N", "C0", "C1"]  # Identity(j), NOT(j), Constant 0, 1


def parse_binary_prompt(prompt: str):
    pairs = []
    target = None
    for line in prompt.split("\n"):
        line = line.strip()
        m = re.match(r"^([01]{8})\s*->\s*([01]{8})\s*$", line)
        if m:
            pairs.append((m.group(1), m.group(2)))
            continue
        m2 = re.search(r"determine the output for[:\s]+([01]{8})", line, re.IGNORECASE)
        if m2:
            target = m2.group(1)
    return pairs, target


def eval_unary(rule_kind, j, inp):
    if rule_kind == "I":  return int(inp[j])
    if rule_kind == "N":  return 1 - int(inp[j])
    if rule_kind == "C0": return 0
    if rule_kind == "C1": return 1
    raise ValueError(rule_kind)


def check_unary_rule_all(rule_kind, j, examples, k):
    """Does Unary(j) produce output bit k for ALL examples?"""
    return all(eval_unary(rule_kind, j, inp) == int(out[k]) for inp, out in examples)


def check_binary_rule_all(op_name, i, j, examples, k):
    """Does Op(input[i], input[j]) produce output bit k for ALL examples?"""
    fn = OPS[op_name]
    for inp, out in examples:
        if fn(int(inp[i]), int(inp[j])) != int(out[k]):
            return False
    return True


def find_sequence(examples):
    """Try to find the longest RUN [k_start, k_end] of consecutive output bits
    that all match the same (op, i_base+delta, j_base+delta) with delta = k - k_start.

    Returns (best_run, rule_info) where rule_info is a dict describing the pattern.
    """
    best = None  # (length, start, end, op, i_base, j_base)

    for op_name in OPS:
        for i_base in range(8):
            for j_base in range(8):
                if i_base == j_base:
                    continue
                # Find runs where Op(inp[(i_base+delta) % 8], inp[(j_base+delta) % 8]) == out[k_start + delta]
                for k_start in range(8):
                    delta = 0
                    while k_start + delta < 8:
                        i = (i_base + delta) % 8
                        j = (j_base + delta) % 8
                        k = k_start + delta
                        if not check_binary_rule_all(op_name, i, j, examples, k):
                            break
                        delta += 1
                    length = delta
                    if length >= 2 and (best is None or length > best[0]):
                        best = (length, k_start, k_start + length - 1, op_name, i_base, j_base)
    return best


def solve_bit_fallback(examples, k):
    """Find ANY rule that produces output bit k for all examples. Priority:
    Constant → Identity → NOT → binary op (first valid)."""
    # Constant
    for kind in ("C0", "C1"):
        if check_unary_rule_all(kind, 0, examples, k):
            return {"kind": kind, "name": kind}
    # Identity / NOT
    for j in range(8):
        if check_unary_rule_all("I", j, examples, k):
            return {"kind": "I", "j": j, "name": f"I{j}"}
        if check_unary_rule_all("N", j, examples, k):
            return {"kind": "N", "j": j, "name": f"N{j}"}
    # Binary ops
    for op_name in OPS:
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                if check_binary_rule_all(op_name, i, j, examples, k):
                    return {"kind": op_name, "i": i, "j": j, "name": f"{op_name}({i},{j})"}
    return None


def solve_all_bits_hybrid(examples):
    """Hybrid THK-style solver:
    1. Find best sequence (covers 2+ consecutive bits)
    2. Fill remaining bits with per-bit fallback

    Returns (sequence_info, per_bit_rules_list) or None if unsolvable.
    """
    seq = find_sequence(examples)
    rules = [None] * 8
    if seq:
        length, start, end, op_name, i_base, j_base = seq
        for delta in range(length):
            k = start + delta
            i = (i_base + delta) % 8
            j = (j_base + delta) % 8
            rules[k] = {
                "kind": op_name,
                "i": i,
                "j": j,
                "name": f"{op_name}({i},{j})",
                "from_seq": True,
            }
    for k in range(8):
        if rules[k] is None:
            r = solve_bit_fallback(examples, k)
            if r is None:
                return None
            rules[k] = r
    return seq, rules


def apply_rule_target(rule, inp):
    kind = rule["kind"]
    if kind == "C0": return 0
    if kind == "C1": return 1
    if kind == "I":  return int(inp[rule["j"]])
    if kind == "N":  return 1 - int(inp[rule["j"]])
    fn = OPS[kind]
    return fn(int(inp[rule["i"]]), int(inp[rule["j"]]))


def apply_solution(rules, inp):
    return "".join(str(apply_rule_target(r, inp)) for r in rules)


# ---------------------------------------------------------------------------
# Trace formatting — THK-inspired, concise
# ---------------------------------------------------------------------------
def format_bit_trace(examples, target, seq, rules, final_answer):
    lines = [
        "I need to find the bit-manipulation rule. The output has 8 bits; I'll",
        "derive each output bit by scanning (op, i, j) triples and looking for",
        "a sequence where consecutive output bits share a +1/+1 index progression",
        "on the same operation. Any bits not covered by a sequence get an",
        "independent per-bit rule.",
        "",
        "Input/Output table:",
    ]
    for inp, out in examples:
        lines.append(f"  {inp} -> {out}")
    lines.append("")

    if seq:
        length, start, end, op_name, i_base, j_base = seq
        lines.append(
            f"Scan found sequence: op={op_name} starting at (i={i_base}, j={j_base}),"
            f" covering output bits [{start}..{end}] with +1/+1 index progression."
        )
        lines.append("Verification against each example:")
        for inp, out in examples[:4]:
            parts = []
            for delta in range(length):
                i = (i_base + delta) % 8
                j = (j_base + delta) % 8
                a, b = int(inp[i]), int(inp[j])
                v = OPS[op_name](a, b)
                parts.append(f"bit[{start + delta}]={v}")
            lines.append(f"  {inp} -> expected {out[start:end+1]}, got {''.join(p.split('=')[1] for p in parts)}  ({'✓' if ''.join(p.split('=')[1] for p in parts) == out[start:end+1] else '✗'})")
        lines.append("")
    else:
        lines.append("No sequence match found — using independent per-bit rules.")
        lines.append("")

    lines.append("Per-bit rules (all bits):")
    for k, rule in enumerate(rules):
        src = " (from sequence)" if rule.get("from_seq") else ""
        lines.append(f"  bit[{k}] = {rule['name']}{src}")
    lines.append("")

    lines.append(f"Applying rules to target {target}:")
    for k, rule in enumerate(rules):
        v = apply_rule_target(rule, target)
        if rule["kind"] in ("C0", "C1"):
            desc = f"{rule['name']} = {v}"
        elif rule["kind"] in ("I", "N"):
            src = f"input[{rule['j']}]={int(target[rule['j']])}"
            op_str = "NOT " if rule["kind"] == "N" else ""
            desc = f"{rule['name']} = {op_str}{src} = {v}"
        else:
            a, b = int(target[rule["i"]]), int(target[rule["j"]])
            desc = f"{rule['name']} = {rule['kind']}({a}, {b}) = {v}"
        lines.append(f"  bit[{k}] = {desc}")
    lines.append("")
    lines.append(f"Concatenating: {final_answer}")

    body = "\n".join(lines)
    return f"<think>\n{body}\n</think>\n\n\\boxed{{{final_answer}}}"


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------
def build_binary_examples(target_count=1400):
    _, by_type, _ = load_train()
    rows = by_type["binary"]
    random.seed(38)
    random.shuffle(rows)

    solved = []
    scanned = 0
    for row in rows:
        scanned += 1
        examples, target = parse_binary_prompt(row["prompt"])
        if not examples or target is None or len(examples) < 5:
            continue
        out = solve_all_bits_hybrid(examples)
        if out is None:
            continue
        seq, rules = out
        predicted = apply_solution(rules, target)
        if predicted != row["answer"]:
            continue
        solved.append((row, examples, target, seq, rules))
        if len(solved) >= target_count:
            break
        if scanned % 200 == 0:
            print(f"  scanned {scanned}, solved {len(solved)}")

    print(f"Binary solved: {len(solved)}/{scanned}")
    seq_count = sum(1 for _, _, _, s, _ in solved if s is not None)
    print(f"  with sequence match: {seq_count}")

    out_list = []
    for row, examples, target, seq, rules in solved:
        user = row["prompt"].rstrip() + "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
        assistant = format_bit_trace(examples, target, seq, rules, row["answer"])
        out_list.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "_meta": {
                "source": "v38_thk_sequence_solver",
                "puzzle_type": "binary",
                "rule_kinds": list({r["kind"] for r in rules}),
                "difficulty": "THK_SEQUENCE",
                "is_unique_rule": False,
            },
        })
    return out_list


def load_other_from_v37():
    """Keep Kh0a's cipher/equation/gravity/unit/roman from v37."""
    kept = []
    with open(V37_JSONL) as f:
        for line in f:
            row = json.loads(line)
            if row["_meta"]["puzzle_type"] != "binary":
                kept.append(row)
    return kept


def main():
    random.seed(38)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building THK-style binary examples...")
    binaries = build_binary_examples(target_count=1400)

    print("\nLoading Kh0a non-binary from v37...")
    others = load_other_from_v37()
    print(f"  v37 non-binary: {len(others)}")

    all_examples = binaries + others
    random.shuffle(all_examples)

    with open(OUT_JSONL, "w") as f:
        for row in all_examples:
            f.write(json.dumps(row) + "\n")

    META_JSON.write_text(json.dumps({
        "title": "Nemotron Training V38",
        "id": "bharatmohan/nemotron-training-v38",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))

    src_counts = Counter(e["_meta"]["source"] for e in all_examples)
    type_counts = Counter(e["_meta"]["puzzle_type"] for e in all_examples)
    print(f"\n=== v38 composition ===")
    print(f"Total: {len(all_examples)}")
    print(f"By source: {dict(src_counts)}")
    print(f"By type:   {dict(type_counts)}")
    print(f"Output: {OUT_JSONL}")


if __name__ == "__main__":
    main()
