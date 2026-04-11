"""Generate v34 training data.

Strategy (from tracking system v30/v31 analysis):
1. v22's cipher/gravity/unit/roman examples are strong → reuse verbatim (340 examples)
2. v22's equation examples are weak but not poisonous → keep (79 examples)
3. v22's binary examples teach "I identify the pattern" hallucination → REPLACE
   Generate 150 new binary examples with explicit per-bit derivation using the
   tracking/verifiers/binary.py solver. Only include rows where the solver's answer
   matches the train.csv ground truth (815 candidates available).

No Donald-style hard labels (LEN:/TABLE:/VOCAB fill:/ANS:) — they teach v30's lying VER
pattern. Keep the v22 verbose free-form style throughout.

Usage: python3 generate_v34.py → writes data/training_v34/training_v34.jsonl
"""
import csv
import json
import random
import sys
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "tracking"))
from verifiers.binary import parse_prompt as parse_binary, classify_bit, solve_ground_truth, GATES


def enumerate_bit_rules(examples, k):
    """Enumerate ALL valid simple rules for output bit k across examples.

    Returns a list of rule dicts (same schema as classify_bit). Used to decide
    whether the per-bit rule is UNIQUE (exactly one valid rule) vs ambiguous.
    """
    out_bits = [int(out[k]) for _, out in examples]
    rules = []

    # Constant
    if all(b == out_bits[0] for b in out_bits):
        rules.append({"kind": "CONSTANT", "value": out_bits[0]})

    # Identity / NOT
    for j in range(8):
        if all(int(inp[j]) == out_bits[i] for i, (inp, _) in enumerate(examples)):
            rules.append({"kind": "IDENTITY", "j": j})
        if all(1 - int(inp[j]) == out_bits[i] for i, (inp, _) in enumerate(examples)):
            rules.append({"kind": "NOT", "j": j})

    # 2-input gates
    for j1 in range(8):
        for j2 in range(j1 + 1, 8):
            for gname, gfn in GATES.items():
                if all(gfn(int(inp[j1]), int(inp[j2])) == out_bits[i]
                       for i, (inp, _) in enumerate(examples)):
                    rules.append({"kind": "2GATE", "gate": gname, "j1": j1, "j2": j2})

    # 3-input majority
    def maj(a, b, c):
        return (a & b) | (a & c) | (b & c)
    for j1 in range(8):
        for j2 in range(j1 + 1, 8):
            for j3 in range(j2 + 1, 8):
                if all(maj(int(inp[j1]), int(inp[j2]), int(inp[j3])) == out_bits[i]
                       for i, (inp, _) in enumerate(examples)):
                    rules.append({"kind": "MAJORITY", "j1": j1, "j2": j2, "j3": j3})

    return rules


import argparse

TRAIN_CSV = Path(__file__).parent / "train.csv"
V22_PATH = Path(__file__).parent / "training_v22" / "training_v22.jsonl"

SUFFIX = "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

# Defaults match the v34 baseline
DEFAULT_N_BINARY = 150
DEFAULT_MIN_UNIQUE_BITS = 0  # 0 = no ambiguity filter
DEFAULT_NAME = "v34"
RANDOM_SEED = 42


def classify(p: str) -> str:
    fl = p.split("\n")[0].lower()
    if "bit manipulation" in fl or "8-bit" in fl: return "binary"
    if "roman" in fl or "numeral" in fl: return "roman"
    if "unit" in fl or "measurement" in fl: return "unit"
    if "gravitational" in fl: return "gravity"
    if "encryption" in fl or "cipher" in fl: return "cipher"
    if "transformation rules" in fl: return "equation"
    return "other"


# ----------------------------------------------------------------------------
# Binary trace generator
# ----------------------------------------------------------------------------
BIT_ORDINAL = ["bit[0] (leftmost)", "bit[1]", "bit[2]", "bit[3]",
               "bit[4]", "bit[5]", "bit[6]", "bit[7] (rightmost)"]


def rule_explanation(rule: dict, k: int) -> str:
    """Human-readable per-bit rule explanation."""
    if rule["kind"] == "CONSTANT":
        return f"bit[{k}]: always {rule['value']} across all examples → CONSTANT {rule['value']}"
    if rule["kind"] == "IDENTITY":
        return f"bit[{k}]: equals input bit[{rule['j']}] in every example → IDENTITY(input[{rule['j']}])"
    if rule["kind"] == "NOT":
        return f"bit[{k}]: equals NOT(input bit[{rule['j']}]) in every example → NOT(input[{rule['j']}])"
    if rule["kind"] == "2GATE":
        return (f"bit[{k}]: output bit matches input[{rule['j1']}] {rule['gate']} input[{rule['j2']}] "
                f"in every example → {rule['gate']}(input[{rule['j1']}], input[{rule['j2']}])")
    if rule["kind"] == "MAJORITY":
        return (f"bit[{k}]: output bit matches MAJORITY(input[{rule['j1']}], input[{rule['j2']}], "
                f"input[{rule['j3']}]) in every example → MAJORITY")
    return f"bit[{k}]: complex"


def apply_rule_verbose(rule: dict, target: str, k: int) -> tuple:
    """Apply rule to target input, return (result_bit, computation_string)."""
    tbits = [int(c) for c in target]
    if rule["kind"] == "CONSTANT":
        return rule["value"], f"bit[{k}] = {rule['value']} (constant)"
    if rule["kind"] == "IDENTITY":
        v = tbits[rule["j"]]
        return v, f"bit[{k}] = input[{rule['j']}] = {v}"
    if rule["kind"] == "NOT":
        v = 1 - tbits[rule["j"]]
        return v, f"bit[{k}] = NOT(input[{rule['j']}]) = NOT({tbits[rule['j']]}) = {v}"
    if rule["kind"] == "2GATE":
        a, b = tbits[rule["j1"]], tbits[rule["j2"]]
        v = GATES[rule["gate"]](a, b)
        return v, f"bit[{k}] = {rule['gate']}(input[{rule['j1']}], input[{rule['j2']}]) = {rule['gate']}({a}, {b}) = {v}"
    if rule["kind"] == "MAJORITY":
        a, b, c = tbits[rule["j1"]], tbits[rule["j2"]], tbits[rule["j3"]]
        v = (a & b) | (a & c) | (b & c)
        return v, f"bit[{k}] = MAJORITY({a}, {b}, {c}) = {v}"
    return 0, f"bit[{k}] = unknown"


def generate_binary_trace(prompt: str, expected: str) -> str:
    """Produce a verbose per-bit derivation trace + \\boxed{answer}."""
    pairs, target = parse_binary(prompt)
    if not pairs or not target:
        return None
    rules = [classify_bit(pairs, k) for k in range(8)]
    if any(r is None for r in rules):
        return None
    # Solve against ground truth
    predicted = solve_ground_truth(pairs, target)
    if predicted != expected:
        return None

    lines = ["<think>"]
    lines.append("I need to find the bit-manipulation rule. The output has 8 bits; I'll derive a rule "
                 "for each output bit independently by checking what's consistent across all examples.")
    lines.append("")
    lines.append("Examples:")
    for inp, out in pairs:
        lines.append(f"  {inp} -> {out}")
    lines.append("")
    lines.append("Per-bit analysis (for each output bit, find the simplest rule consistent with every example):")
    for k, rule in enumerate(rules):
        lines.append(f"  {rule_explanation(rule, k)}")
    lines.append("")
    lines.append(f"Applying the derived rules to target input {target}:")
    result_bits = []
    for k, rule in enumerate(rules):
        b, expl = apply_rule_verbose(rule, target, k)
        lines.append(f"  {expl}")
        result_bits.append(str(b))
    lines.append("")
    lines.append(f"Concatenating bits: {''.join(result_bits)}")
    lines.append("</think>")
    lines.append("")
    lines.append(f"\\boxed{{{''.join(result_bits)}}}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Main generation
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate Nemotron training data variants")
    ap.add_argument("--n-binary", type=int, default=DEFAULT_N_BINARY,
                    help="Number of new binary examples to generate (default 150)")
    ap.add_argument("--min-unique-bits", type=int, default=DEFAULT_MIN_UNIQUE_BITS,
                    help="Min number of bits (0-8) where the per-bit rule must be UNIQUELY correct. "
                         "0=no filter; 4=keep examples where >=4 bits are unambiguous; 8=fully unique")
    ap.add_argument("--name", default=DEFAULT_NAME,
                    help="Variant name (controls output dir + dataset id, e.g. 'v34', 'v34c', 'v34_strict')")
    ap.add_argument("--prefer-rules", default=None,
                    help="Comma-sep list of rule kinds to prefer (e.g. '2GATE,MAJORITY' to focus on hard rules). "
                         "If set, only keep binary examples where AT LEAST ONE bit uses one of these rules.")
    ap.add_argument("--keep-v22-binary", action="store_true",
                    help="Also keep v22's original binary examples (default: drop them)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    random.seed(args.seed)

    out_dir = Path(__file__).parent / f"training_{args.name}"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / f"training_{args.name}.jsonl"

    print(f"=== Generating training data variant: {args.name} ===")
    print(f"  n_binary={args.n_binary}")
    print(f"  min_unique_bits={args.min_unique_bits}")
    print(f"  prefer_rules={args.prefer_rules}")
    print(f"  keep_v22_binary={args.keep_v22_binary}")
    print(f"  output={out_path}")
    print()

    prefer_rules_set = None
    if args.prefer_rules:
        prefer_rules_set = set(args.prefer_rules.split(","))

    # 1. Load v22 base, split by type
    print(f"Loading v22 base from {V22_PATH}...")
    v22_by_type = defaultdict(list)
    with open(V22_PATH) as f:
        for line in f:
            ex = json.loads(line)
            t = classify(ex["messages"][0]["content"])
            v22_by_type[t].append(ex)
    print(f"v22 breakdown: {{t: len(items) for t, items in v22_by_type.items()}}")
    for t in sorted(v22_by_type):
        print(f"  {t}: {len(v22_by_type[t])}")

    # 2. Keep cipher, gravity, unit, roman, equation as-is (tag as v22_base)
    output = []
    keep_types = ["cipher", "gravity", "unit", "roman", "equation"]
    if args.keep_v22_binary:
        keep_types.append("binary")
    for t in keep_types:
        for ex in v22_by_type.get(t, []):
            tagged = dict(ex)  # shallow copy preserves messages
            tagged["_meta"] = {
                "source": "v22_base",
                "puzzle_type": t,
                "rule_kinds": [],
                "difficulty": "UNKNOWN",
                "is_unique_rule": False,
            }
            output.append(tagged)
    print(f"Kept {len(output)} v22 examples ({', '.join(keep_types)})")

    # 3. Generate new binary examples from the solver
    print(f"\nGenerating up to {args.n_binary} new binary examples from train.csv...")
    with open(TRAIN_CSV) as f:
        train_rows = list(csv.DictReader(f))
    binary_rows = [r for r in train_rows if classify(r["prompt"]) == "binary"]
    print(f"Binary rows in train.csv: {len(binary_rows)}")

    random.shuffle(binary_rows)
    generated = 0
    skipped_unsolvable = 0
    skipped_ambiguous = 0
    skipped_rules = 0
    difficulty_hit = Counter()

    for r in binary_rows:
        if generated >= args.n_binary:
            break
        trace = generate_binary_trace(r["prompt"], r["answer"])
        if trace is None:
            skipped_unsolvable += 1
            continue

        # Derive per-bit rule metadata for attribution
        pairs, target = parse_binary(r["prompt"])
        rules = [classify_bit(pairs, k) for k in range(8)]
        rule_kinds = [rule["kind"] if rule else "COMPLEX" for rule in rules]
        all_rule_sets = [enumerate_bit_rules(pairs, k) for k in range(8)]
        is_unique_rule = all(len(rs) == 1 for rs in all_rule_sets)

        # FILTER 1: ambiguity threshold
        if args.min_unique_bits > 0:
            unique_bit_count = sum(1 for rs in all_rule_sets if len(rs) == 1)
            if unique_bit_count < args.min_unique_bits:
                skipped_ambiguous += 1
                continue

        # FILTER 2: prefer specific rule kinds (e.g., HARD cases like 2GATE/MAJORITY)
        if prefer_rules_set:
            row_kinds = set(rule_kinds)
            if not (row_kinds & prefer_rules_set):
                skipped_rules += 1
                continue

        user_content = r["prompt"]
        if not user_content.rstrip().endswith("\\boxed{}`."):
            user_content = user_content.rstrip() + SUFFIX

        example = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": trace},
            ],
            "_meta": {
                "source": f"{args.name}_binary_solver",
                "puzzle_type": "binary",
                "rule_kinds": rule_kinds,
                "difficulty": "SIMPLE",
                "is_unique_rule": is_unique_rule,
                "unique_bit_count": sum(1 for rs in all_rule_sets if len(rs) == 1),
            },
        }
        output.append(example)
        generated += 1

        for rule in rules:
            if rule:
                difficulty_hit[rule["kind"]] += 1

    print(f"Generated {generated} new binary examples")
    print(f"  skipped: {skipped_unsolvable} unsolvable, {skipped_ambiguous} too-ambiguous, {skipped_rules} no preferred rule")
    print(f"  rule kinds used: {dict(difficulty_hit)}")

    # 4. Shuffle and save
    random.shuffle(output)
    with open(out_path, "w") as f:
        for ex in output:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(output)} examples to {out_path}")
    print(f"Final type mix:")
    final_mix = Counter()
    for ex in output:
        final_mix[classify(ex["messages"][0]["content"])] += 1
    for t in sorted(final_mix):
        print(f"  {t}: {final_mix[t]}")

    # Write dataset-metadata.json for kaggle CLI upload
    meta_path = out_dir / "dataset-metadata.json"
    if not meta_path.exists():
        meta_path.write_text(
            f'{{\n  "title": "Nemotron Training {args.name.upper()}",\n'
            f'  "id": "bharatmohan/nemotron-training-{args.name}",\n'
            f'  "licenses": [{{"name": "CC0-1.0"}}]\n}}\n'
        )
        print(f"Wrote dataset-metadata.json (run: kaggle datasets create -p {out_dir})")


if __name__ == "__main__":
    main()
