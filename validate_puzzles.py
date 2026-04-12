#!/usr/bin/env python3
"""
Validate gravity, unit_conversion, and numeral_system puzzles from train.csv.
Sample 85 examples per type (255 total), verify answers, generate CoT.
Save verified examples to training_data/verified_gravity_unit_numeral.jsonl
"""

import csv
import json
import os
import re
import random

random.seed(42)

DATA_PATH = "/Users/bharat/Downloads/kaggle/data/train.csv"
OUTPUT_PATH = "/Users/bharat/Downloads/kaggle/training_data/verified_gravity_unit_numeral.jsonl"

# ============================================================
# 1. Load and classify examples
# ============================================================
gravity_rows = []
unit_rows = []
numeral_rows = []

with open(DATA_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompt = row['prompt']
        pl = prompt.lower()
        if 'numeral system' in pl:
            numeral_rows.append(row)
        elif 'unit conversion' in pl:
            unit_rows.append(row)
        elif 'gravitational' in pl:
            gravity_rows.append(row)

print(f"Total gravity: {len(gravity_rows)}")
print(f"Total unit_conversion: {len(unit_rows)}")
print(f"Total numeral_system: {len(numeral_rows)}")

# Sample 85 from each
gravity_sample = random.sample(gravity_rows, min(85, len(gravity_rows)))
unit_sample = random.sample(unit_rows, min(85, len(unit_rows)))
numeral_sample = random.sample(numeral_rows, min(85, len(numeral_rows)))

# ============================================================
# 2. Validation functions
# ============================================================

def parse_gravity(prompt):
    """Parse gravity prompt: extract (t, d) pairs and target t."""
    # Extract example pairs: "For t = X s, distance = Y m"
    pairs = re.findall(r'For t = ([\d.]+)\s*s,\s*distance = ([\d.]+)\s*m', prompt)
    pairs = [(float(t), float(d)) for t, d in pairs]

    # Extract target t: "for t = X s" at the end
    target_match = re.search(r'determine the falling distance for t = ([\d.]+)\s*s', prompt)
    if not target_match:
        return None, None
    target_t = float(target_match.group(1))
    return pairs, target_t

def validate_gravity(prompt, answer_str):
    """Validate gravity puzzle. Returns (is_valid, computed_answer, details)."""
    pairs, target_t = parse_gravity(prompt)
    if pairs is None or len(pairs) == 0:
        return False, None, "Could not parse prompt"

    # Calculate g from each pair: g = 2d / t^2
    g_values = []
    for t, d in pairs:
        if t == 0:
            continue
        g = 2.0 * d / (t * t)
        g_values.append(g)

    if not g_values:
        return False, None, "No valid g values"

    avg_g = sum(g_values) / len(g_values)

    # Compute distance for target t
    computed_d = 0.5 * avg_g * target_t * target_t
    computed_d_rounded = round(computed_d, 2)

    try:
        expected = float(answer_str)
    except ValueError:
        return False, computed_d_rounded, f"Answer not numeric: {answer_str}"

    diff = abs(computed_d_rounded - expected)
    is_valid = diff <= 0.02

    details = f"g_values={[round(g,4) for g in g_values]}, avg_g={round(avg_g,4)}, target_t={target_t}, computed={computed_d_rounded}, expected={expected}, diff={round(diff,4)}"
    return is_valid, computed_d_rounded, details

def generate_gravity_cot(prompt, answer_str):
    """Generate step-by-step CoT for gravity puzzle."""
    pairs, target_t = parse_gravity(prompt)
    g_values = []
    steps = []

    steps.append("I need to find the gravitational constant g from the given observations, then compute the distance for the target time.")
    steps.append("")
    steps.append("Using d = 0.5 * g * t^2, I can solve for g = 2d / t^2 for each observation:")
    steps.append("")

    for i, (t, d) in enumerate(pairs, 1):
        g = 2.0 * d / (t * t)
        g_values.append(g)
        steps.append(f"Observation {i}: t = {t}s, d = {d}m")
        steps.append(f"  g = 2 * {d} / ({t})^2 = {2*d} / {round(t*t, 4)} = {round(g, 4)} m/s^2")
        steps.append("")

    avg_g = sum(g_values) / len(g_values)
    steps.append(f"Average g = ({' + '.join([str(round(g, 4)) for g in g_values])}) / {len(g_values)}")
    steps.append(f"Average g = {round(sum(g_values), 4)} / {len(g_values)} = {round(avg_g, 4)} m/s^2")
    steps.append("")

    computed_d = 0.5 * avg_g * target_t * target_t
    steps.append(f"Now compute distance for t = {target_t}s:")
    steps.append(f"d = 0.5 * {round(avg_g, 4)} * ({target_t})^2")
    steps.append(f"d = 0.5 * {round(avg_g, 4)} * {round(target_t*target_t, 4)}")
    steps.append(f"d = {round(computed_d, 2)}")

    cot = "<think>\n" + "\n".join(steps) + "\n</think>\n\n\\boxed{" + answer_str + "}"
    return cot

def parse_unit_conversion(prompt):
    """Parse unit_conversion prompt: extract (input, output) pairs and target value."""
    # Pattern: "X m becomes Y" or "X m becomes Y"
    pairs = re.findall(r'([\d.]+)\s*m?\s*becomes\s*([\d.]+)', prompt)
    pairs = [(float(x), float(y)) for x, y in pairs]

    # Target: "convert the following measurement: X m"
    target_match = re.search(r'convert the following measurement:\s*([\d.]+)', prompt)
    if not target_match:
        return None, None
    target = float(target_match.group(1))
    return pairs, target

def validate_unit_conversion(prompt, answer_str):
    """Validate unit_conversion puzzle."""
    pairs, target = parse_unit_conversion(prompt)
    if pairs is None or len(pairs) == 0:
        return False, None, "Could not parse prompt"

    # Calculate conversion factor from each pair: factor = output / input
    factors = []
    for inp, out in pairs:
        if inp == 0:
            continue
        factor = out / inp
        factors.append(factor)

    if not factors:
        return False, None, "No valid factors"

    avg_factor = sum(factors) / len(factors)
    computed = round(avg_factor * target, 2)

    try:
        expected = float(answer_str)
    except ValueError:
        return False, computed, f"Answer not numeric: {answer_str}"

    diff = abs(computed - expected)
    is_valid = diff <= 0.02

    details = f"factors={[round(f,6) for f in factors]}, avg_factor={round(avg_factor,6)}, target={target}, computed={computed}, expected={expected}, diff={round(diff,4)}"
    return is_valid, computed, details

def generate_unit_cot(prompt, answer_str):
    """Generate step-by-step CoT for unit_conversion puzzle."""
    pairs, target = parse_unit_conversion(prompt)
    factors = []
    steps = []

    steps.append("I need to determine the conversion factor from the given examples, then apply it to the target measurement.")
    steps.append("")
    steps.append("Computing factor = output / input for each example:")
    steps.append("")

    for i, (inp, out) in enumerate(pairs, 1):
        factor = out / inp
        factors.append(factor)
        steps.append(f"Example {i}: {inp} -> {out}, factor = {out} / {inp} = {round(factor, 6)}")

    steps.append("")
    avg_factor = sum(factors) / len(factors)
    steps.append(f"Average factor = {round(avg_factor, 6)}")
    steps.append("")

    computed = avg_factor * target
    steps.append(f"Converting target measurement {target}:")
    steps.append(f"Result = {round(avg_factor, 6)} * {target} = {round(computed, 2)}")

    cot = "<think>\n" + "\n".join(steps) + "\n</think>\n\n\\boxed{" + answer_str + "}"
    return cot

def int_to_roman(num):
    """Convert integer to Roman numeral string."""
    values = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]
    result = ''
    for val, sym in values:
        while num >= val:
            result += sym
            num -= val
    return result

def parse_numeral_system(prompt):
    """Parse numeral_system prompt: extract target number."""
    # "write the number X in the Wonderland numeral system"
    target_match = re.search(r'write the number (\d+) in the Wonderland numeral system', prompt)
    if not target_match:
        return None
    return int(target_match.group(1))

def validate_numeral_system(prompt, answer_str):
    """Validate numeral_system puzzle."""
    target = parse_numeral_system(prompt)
    if target is None:
        return False, None, "Could not parse target number"

    computed = int_to_roman(target)
    is_valid = computed == answer_str.strip()

    details = f"target={target}, computed={computed}, expected={answer_str.strip()}"
    return is_valid, computed, details

def generate_numeral_cot(prompt, answer_str):
    """Generate step-by-step CoT for numeral_system puzzle."""
    target = parse_numeral_system(prompt)

    # Extract example pairs from prompt for reference
    examples = re.findall(r'(\d+)\s*->\s*([A-Z]+)', prompt)

    steps = []
    steps.append("I need to convert the number to Roman numerals using standard rules.")
    steps.append("")
    steps.append("First, let me verify the mapping from the given examples:")
    for num_str, roman in examples:
        steps.append(f"  {num_str} -> {roman} (standard Roman: {int_to_roman(int(num_str))})")
    steps.append("")
    steps.append("The examples confirm standard Roman numeral conversion rules.")
    steps.append("")

    # Show step-by-step conversion
    steps.append(f"Now converting {target} to Roman numerals:")
    steps.append("")

    values = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]

    remaining = target
    roman_parts = []
    for val, sym in values:
        if remaining >= val:
            count = remaining // val
            roman_parts.append(sym * count)
            steps.append(f"{remaining} >= {val}: write {sym} x {count} = {sym * count}, remainder = {remaining - val * count}")
            remaining -= val * count

    if remaining == 0 and roman_parts:
        result = ''.join(roman_parts)
        steps.append(f"")
        steps.append(f"Result: {''.join(roman_parts)} = {result}")

    cot = "<think>\n" + "\n".join(steps) + "\n</think>\n\n\\boxed{" + answer_str.strip() + "}"
    return cot

# ============================================================
# 3. Validate all samples
# ============================================================

verified = []
failures = []
stats = {'gravity': {'total': 0, 'verified': 0},
         'unit_conversion': {'total': 0, 'verified': 0},
         'numeral_system': {'total': 0, 'verified': 0}}

# --- Gravity ---
print("\n" + "="*60)
print("GRAVITY VALIDATION")
print("="*60)
for row in gravity_sample:
    stats['gravity']['total'] += 1
    is_valid, computed, details = validate_gravity(row['prompt'], row['answer'])
    if is_valid:
        stats['gravity']['verified'] += 1
        cot = generate_gravity_cot(row['prompt'], row['answer'])
        user_content = row['prompt'] + '\nPlease put your final answer inside \\boxed{}.'
        verified.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": cot}
            ]
        })
    else:
        failures.append(('gravity', row['id'], details))

print(f"Verified: {stats['gravity']['verified']}/{stats['gravity']['total']}")

# --- Unit Conversion ---
print("\n" + "="*60)
print("UNIT CONVERSION VALIDATION")
print("="*60)
for row in unit_sample:
    stats['unit_conversion']['total'] += 1
    is_valid, computed, details = validate_unit_conversion(row['prompt'], row['answer'])
    if is_valid:
        stats['unit_conversion']['verified'] += 1
        cot = generate_unit_cot(row['prompt'], row['answer'])
        user_content = row['prompt'] + '\nPlease put your final answer inside \\boxed{}.'
        verified.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": cot}
            ]
        })
    else:
        failures.append(('unit_conversion', row['id'], details))

print(f"Verified: {stats['unit_conversion']['verified']}/{stats['unit_conversion']['total']}")

# --- Numeral System ---
print("\n" + "="*60)
print("NUMERAL SYSTEM VALIDATION")
print("="*60)
for row in numeral_sample:
    stats['numeral_system']['total'] += 1
    is_valid, computed, details = validate_numeral_system(row['prompt'], row['answer'])
    if is_valid:
        stats['numeral_system']['verified'] += 1
        cot = generate_numeral_cot(row['prompt'], row['answer'])
        user_content = row['prompt'] + '\nPlease put your final answer inside \\boxed{}.'
        verified.append({
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": cot}
            ]
        })
    else:
        failures.append(('numeral_system', row['id'], details))

print(f"Verified: {stats['numeral_system']['verified']}/{stats['numeral_system']['total']}")

# ============================================================
# 4. Save results
# ============================================================
with open(OUTPUT_PATH, 'w') as f:
    for item in verified:
        f.write(json.dumps(item) + '\n')

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
total_verified = sum(s['verified'] for s in stats.values())
total_sampled = sum(s['total'] for s in stats.values())
print(f"Total verified: {total_verified}/{total_sampled}")
for ptype, s in stats.items():
    print(f"  {ptype}: {s['verified']}/{s['total']}")

print(f"\nSaved {total_verified} verified examples to {OUTPUT_PATH}")

if failures:
    print(f"\n{'='*60}")
    print(f"FAILURES ({len(failures)} total)")
    print(f"{'='*60}")
    for ptype, pid, details in failures:
        print(f"  [{ptype}] id={pid}: {details}")
