"""Generate v36 training data — adopts Tong Hui Kang (1st place, 0.85) bit_manipulation algorithm.

Research insight (Apr 11 2026): THK's algorithm solves ~85% of binary puzzles vs
the tracking verifier's ~62% and v34c's 25% model performance. The trick is
searching a MUCH larger gate space per output bit:

  - Identity(j), NOT(j) for j=0..7
  - Constant 0, Constant 1
  - AND(i,j), OR(i,j), XOR(i,j) for all pairs
  - AND-NOT(i,j) = AND(input[i], NOT(input[j]))  — NON-commutative
  - OR-NOT(i,j), XOR-NOT(i,j) — NON-commutative

v36 data composition:
  - v34c cipher/gravity/unit/roman (340 verbatim — don't touch saturated types)
  - v35's 13 verifier-derived equation examples
  - NEW: ~150 binary examples using THK-style expanded gate search
  - Total: ~503

Config follows Kh0a's public 0.73 notebook (research Apr 11):
  - 11 target modules (embed_tokens, lm_head added)
  - max_seq_len 4096 (we keep 4096 not 3500 since traces can be longer)
  - LR=1e-4, 2 epochs

Usage: python3 generate_v36.py → data/training_v36/training_v36.jsonl
"""
import csv
import json
import random
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tracking"))
from train_loader import load_train

TRAIN_CSV = ROOT / "data" / "train.csv"
V34C_JSONL = ROOT / "data" / "training_v34c" / "training_v34c.jsonl"
V35_JSONL = ROOT / "data" / "training_v35" / "training_v35.jsonl"
OUT_DIR = ROOT / "data" / "training_v36"
OUT_JSONL = OUT_DIR / "training_v36.jsonl"
META_JSON = OUT_DIR / "dataset-metadata.json"


# ---------------------------------------------------------------------------
# THK-style gate search: enumerate all valid rules for each output bit
# ---------------------------------------------------------------------------
def enumerate_rules(examples, k):
    """Return list of candidate rules for output bit k, each as dict with
    'kind', 'args', and 'name' (short string for trace)."""
    rules = []
    out_bits = [int(out[k]) for _, out in examples]

    # Constant 0 / 1
    if all(b == 0 for b in out_bits):
        rules.append({"kind": "C", "val": 0, "name": "C0"})
    if all(b == 1 for b in out_bits):
        rules.append({"kind": "C", "val": 1, "name": "C1"})

    # Identity and NOT
    for j in range(8):
        if all(int(inp[j]) == out_bits[i] for i, (inp, _) in enumerate(examples)):
            rules.append({"kind": "I", "j": j, "name": f"I{j}"})
        if all(1 - int(inp[j]) == out_bits[i] for i, (inp, _) in enumerate(examples)):
            rules.append({"kind": "N", "j": j, "name": f"N{j}"})

    # 2-input: AND, OR, XOR (commutative), AND-NOT/OR-NOT/XOR-NOT (non-commutative)
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            i_vals = [int(inp[i]) for inp, _ in examples]
            j_vals = [int(inp[j]) for inp, _ in examples]
            # commutative: only i < j
            if i < j:
                if all((a & b) == out_bits[n] for n, (a, b) in enumerate(zip(i_vals, j_vals))):
                    rules.append({"kind": "AND", "i": i, "j": j, "name": f"AND({i},{j})"})
                if all((a | b) == out_bits[n] for n, (a, b) in enumerate(zip(i_vals, j_vals))):
                    rules.append({"kind": "OR", "i": i, "j": j, "name": f"OR({i},{j})"})
                if all((a ^ b) == out_bits[n] for n, (a, b) in enumerate(zip(i_vals, j_vals))):
                    rules.append({"kind": "XOR", "i": i, "j": j, "name": f"XOR({i},{j})"})
            # non-commutative
            if all((a & (1 - b)) == out_bits[n] for n, (a, b) in enumerate(zip(i_vals, j_vals))):
                rules.append({"kind": "ANDN", "i": i, "j": j, "name": f"ANDN({i},{j})"})
            if all((a | (1 - b)) == out_bits[n] for n, (a, b) in enumerate(zip(i_vals, j_vals))):
                rules.append({"kind": "ORN", "i": i, "j": j, "name": f"ORN({i},{j})"})
            if all((a ^ (1 - b)) == out_bits[n] for n, (a, b) in enumerate(zip(i_vals, j_vals))):
                rules.append({"kind": "XORN", "i": i, "j": j, "name": f"XORN({i},{j})"})
    return rules


def apply_rule(rule, inp):
    if rule["kind"] == "C":
        return rule["val"]
    if rule["kind"] == "I":
        return int(inp[rule["j"]])
    if rule["kind"] == "N":
        return 1 - int(inp[rule["j"]])
    a = int(inp[rule["i"]])
    b = int(inp[rule["j"]])
    if rule["kind"] == "AND":
        return a & b
    if rule["kind"] == "OR":
        return a | b
    if rule["kind"] == "XOR":
        return a ^ b
    if rule["kind"] == "ANDN":
        return a & (1 - b)
    if rule["kind"] == "ORN":
        return a | (1 - b)
    if rule["kind"] == "XORN":
        return a ^ (1 - b)
    raise ValueError(rule)


def pick_simplest(rules):
    """Priority order: Constant > Identity > NOT > AND > OR > XOR > AND-NOT > OR-NOT > XOR-NOT."""
    priority = {"C": 0, "I": 1, "N": 2, "AND": 3, "OR": 4, "XOR": 5, "ANDN": 6, "ORN": 7, "XORN": 8}
    return sorted(rules, key=lambda r: priority[r["kind"]])[0]


def solve_all_bits(examples):
    """Return list of 8 rules (one per output bit) or None if any bit is unsolvable."""
    solution = []
    for k in range(8):
        rules = enumerate_rules(examples, k)
        if not rules:
            return None
        solution.append(pick_simplest(rules))
    return solution


def apply_solution(solution, inp):
    return "".join(str(apply_rule(r, inp)) for r in solution)


# ---------------------------------------------------------------------------
# Parse binary prompt (copy from tracking/verifiers/binary.py pattern)
# ---------------------------------------------------------------------------
def parse_binary_prompt(prompt: str):
    """Extract [(inp8, out8)] pairs and target inp8 from prompt."""
    import re
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


# ---------------------------------------------------------------------------
# Format a THK-style training trace — concise version
# ---------------------------------------------------------------------------
def format_trace(examples, target, solution, final_answer):
    lines = [
        "I need to find the bit-manipulation rule. The output has 8 bits; I'll derive the rule for each output bit independently by scanning candidate operations until one matches every example.",
        "",
        "Candidate operations per bit: Constant 0/1, Identity(j), NOT(j), AND(i,j), OR(i,j), XOR(i,j), AND-NOT(i,j)=i AND NOT(j), OR-NOT(i,j), XOR-NOT(i,j).",
        "",
        "Examples:",
    ]
    for inp, out in examples:
        lines.append(f"  {inp} -> {out}")
    lines.append("")
    lines.append("Per-bit derivation (showing the rule that matches all examples):")
    for k, rule in enumerate(solution):
        verify_parts = []
        for inp, out in examples[:3]:
            got = apply_rule(rule, inp)
            verify_parts.append(f"{inp}→{got}")
        verify_str = ", ".join(verify_parts) + f" ... (all {len(examples)} examples match)"
        lines.append(f"  bit[{k}] = {rule['name']}  — verify: {verify_str}")
    lines.append("")
    lines.append(f"Applying the 8 rules to target input {target}:")
    for k, rule in enumerate(solution):
        val = apply_rule(rule, target)
        if rule["kind"] == "C":
            desc = f"{rule['name']} = {rule['val']}"
        elif rule["kind"] in ("I", "N"):
            src = f"input[{rule['j']}]={int(target[rule['j']])}"
            op = "NOT " if rule["kind"] == "N" else ""
            desc = f"{rule['name']} = {op}{src} = {val}"
        else:
            a = int(target[rule["i"]])
            b = int(target[rule["j"]])
            op_map = {
                "AND": f"{a} AND {b}",
                "OR": f"{a} OR {b}",
                "XOR": f"{a} XOR {b}",
                "ANDN": f"{a} AND NOT({b})",
                "ORN": f"{a} OR NOT({b})",
                "XORN": f"{a} XOR NOT({b})",
            }
            desc = f"{rule['name']} = {op_map[rule['kind']]} = {val}"
        lines.append(f"  bit[{k}] = {desc}")
    lines.append("")
    lines.append(f"Concatenating: {final_answer}")
    trace = "\n".join(lines)
    return f"<think>\n{trace}\n</think>\n\n\\boxed{{{final_answer}}}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_binary_examples(target_count=50):
    _, by_type, _ = load_train()
    binary_rows = by_type["binary"]
    random.seed(36)
    random.shuffle(binary_rows)

    solved = []
    for row in binary_rows:
        examples, target = parse_binary_prompt(row["prompt"])
        if not examples or target is None or len(examples) < 5:
            continue
        solution = solve_all_bits(examples)
        if solution is None:
            continue
        predicted = apply_solution(solution, target)
        if predicted != row["answer"]:
            continue
        solved.append((row, examples, target, solution))
        if len(solved) >= target_count:
            break

    rule_kind_counts = Counter()
    for _, _, _, sol in solved:
        for r in sol:
            rule_kind_counts[r["kind"]] += 1
    print(f"Binary solved: {len(solved)}/{len(binary_rows)}")
    print(f"Rule kind distribution (across 8*n bits): {dict(rule_kind_counts)}")

    out = []
    for row, examples, target, solution in solved:
        user = row["prompt"].rstrip() + "\nPlease put your final answer inside `\\boxed{}`."
        assistant = format_trace(examples, target, solution, row["answer"])
        out.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "_meta": {
                "source": "v36_thk_binary_solver",
                "puzzle_type": "binary",
                "rule_kinds": list({r["kind"] for r in solution}),
                "difficulty": "THK_EXPANDED_GATES",
                "is_unique_rule": False,
            },
        })
    return out


def load_kept_from_v34c():
    """v34c cipher/gravity/unit/roman (saturated) — 340 examples."""
    kept = []
    with open(V34C_JSONL) as f:
        for line in f:
            row = json.loads(line)
            ptype = row.get("_meta", {}).get("puzzle_type", "")
            if ptype in ("cipher", "gravity", "unit", "roman"):
                kept.append(row)
    print(f"v34c kept (cipher/gravity/unit/roman): {len(kept)}")
    return kept


def load_equation_from_v35():
    """v35's 13 verifier-derived equation examples."""
    kept = []
    if not V35_JSONL.exists():
        print("v35 not yet generated; skipping equation")
        return []
    with open(V35_JSONL) as f:
        for line in f:
            row = json.loads(line)
            if row.get("_meta", {}).get("puzzle_type") == "equation":
                kept.append(row)
    print(f"v35 equation: {len(kept)}")
    return kept


def main():
    random.seed(36)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    saturated = load_kept_from_v34c()
    equations = load_equation_from_v35()
    binaries = build_binary_examples(target_count=50)

    all_examples = saturated + equations + binaries
    random.shuffle(all_examples)

    with open(OUT_JSONL, "w") as f:
        for row in all_examples:
            f.write(json.dumps(row) + "\n")

    META_JSON.write_text(json.dumps({
        "title": "Nemotron Training V36",
        "id": "bharatmohan/nemotron-training-v36",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))

    src_counts = Counter(e["_meta"]["source"] for e in all_examples)
    type_counts = Counter(e["_meta"].get("puzzle_type", "?") for e in all_examples)
    print(f"\n=== v36 composition ===")
    print(f"Total: {len(all_examples)}")
    print(f"By source: {dict(src_counts)}")
    print(f"By type:   {dict(type_counts)}")
    print(f"Output: {OUT_JSONL}")


if __name__ == "__main__":
    main()
