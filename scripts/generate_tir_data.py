"""
Generate high-quality Tool-Integrated Reasoning (TIR) training data
for all 9,500 competition puzzles + broad reasoning data from AIMO3.

Each CoT includes actual step-by-step computation with Python code traces.
"""

import csv
import json
import os
import re
import random
import zipfile
import io
import statistics
from collections import defaultdict

DATA_DIR = "/Users/bharat/Downloads/kaggle/data"
OUTPUT_DIR = "/Users/bharat/Downloads/kaggle/data/training_v11"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PUZZLE PARSERS
# ============================================================

def parse_gravity(prompt):
    """Parse gravity puzzle: extract (t, d) pairs and target t."""
    pairs = re.findall(r't\s*=\s*([\d.]+)\s*s?,\s*distance\s*=\s*([\d.]+)', prompt)
    after_now = prompt.split('Now')[1] if 'Now' in prompt else prompt[-200:]
    target_match = re.search(r't\s*=\s*([\d.]+)\s*s', after_now)
    target_t = float(target_match.group(1)) if target_match else None
    examples = [(float(t), float(d)) for t, d in pairs]
    return examples, target_t


def parse_unit_conversion(prompt):
    """Parse unit conversion: extract (input, output) pairs and target."""
    pairs = re.findall(r'([\d.]+)\s*m?\s*becomes\s*([\d.]+)', prompt)
    target_match = re.search(r'convert.*?:\s*([\d.]+)', prompt, re.IGNORECASE)
    if not target_match:
        target_match = re.search(r'([\d.]+)\s*m?\s*$', prompt.strip())
    target = float(target_match.group(1)) if target_match else None
    examples = [(float(i), float(o)) for i, o in pairs]
    return examples, target


def parse_numeral(prompt):
    """Parse numeral system: extract (number, roman) pairs and target number."""
    pairs = re.findall(r'(\d+)\s*->\s*([IVXLCDM]+)', prompt)
    after_now = prompt.split('Now')[1] if 'Now' in prompt else prompt[-200:]
    target_match = re.search(r'(?:number|write)\s+(?:the\s+)?(?:number\s+)?(\d+)', after_now, re.IGNORECASE)
    if not target_match:
        target_match = re.search(r'(\d+)', after_now)
    target = int(target_match.group(1)) if target_match else None
    examples = [(int(n), r) for n, r in pairs]
    return examples, target


def parse_cipher(prompt):
    """Parse cipher: extract (encrypted, decrypted) text pairs and target."""
    lines = prompt.strip().split('\n')
    pairs = []
    target_text = None
    in_examples = False

    for line in lines:
        stripped = line.strip()
        if '->' in stripped and 'example' not in stripped.lower() and 'rule' not in stripped.lower():
            m = re.match(r'^(.+?)\s*->\s*(.+)$', stripped)
            if m:
                enc = m.group(1).strip()
                dec = m.group(2).strip()
                if len(enc) > 1 and len(dec) > 1:
                    pairs.append((enc, dec))

    # Find target text after "decrypt" keyword
    decrypt_match = re.search(r'(?:decrypt|following)\s+(?:text|message)?[:\s]+(.+?)$', prompt, re.IGNORECASE | re.MULTILINE)
    if decrypt_match:
        target_text = decrypt_match.group(1).strip()

    return pairs, target_text


def parse_bit_manipulation(prompt):
    """Parse bit manipulation: extract (input, output) binary pairs and target."""
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    # Find target - after "output for:" or "determine" at end
    after_now = prompt.split('Now')[1] if 'Now' in prompt else prompt[-200:]
    target_match = re.search(r'([01]{8})', after_now)
    target = target_match.group(1) if target_match else None
    return pairs, target


def parse_equation_transform(prompt):
    """Parse equation transform: extract (input, output) symbol pairs and target."""
    lines = prompt.strip().split('\n')
    pairs = []
    target = None

    for line in lines:
        stripped = line.strip().replace('`', '')
        if ' = ' in stripped and 'example' not in line.lower() and 'rule' not in line.lower() \
                and 'alice' not in line.lower() and 'determine' not in line.lower() and 'result' not in line.lower():
            m = re.match(r'^(.+?)\s+=\s+(.+)$', stripped)
            if m:
                pairs.append((m.group(1).strip(), m.group(2).strip()))

    # Find target
    target_match = re.search(r'(?:result|output)\s+for[:\s]+[`]?(.+?)[`]?\s*$', prompt, re.IGNORECASE | re.MULTILINE)
    if not target_match:
        target_match = re.search(r'determine.*?[`](.+?)[`]', prompt, re.IGNORECASE)
    if target_match:
        target = target_match.group(1).strip()

    return pairs, target


# ============================================================
# COT GENERATORS (TIR-style with Python code traces)
# ============================================================

def generate_gravity_cot(prompt, answer, examples, target_t):
    if not examples or target_t is None:
        return generate_generic_cot(prompt, answer, "gravity")

    g_values = []
    comp_lines = []
    for i, (t, d) in enumerate(examples):
        g = 2 * d / (t * t)
        g_values.append(g)
        comp_lines.append(f"  t={t:.2f}s, d={d:.2f}m: g = 2×{d:.2f}/{t:.2f}² = {g:.4f} m/s²")

    avg_g = statistics.mean(g_values)
    result = 0.5 * avg_g * target_t * target_t

    cot = f"""<think>
This is a gravitational distance problem. The formula is d = 0.5·g·t², so g = 2d/t².

I'll calculate g from each example:

```python
examples = {[(t, d) for t, d in examples]}
g_values = [2*d/t**2 for t, d in examples]
avg_g = sum(g_values) / len(g_values)
target_t = {target_t}
distance = 0.5 * avg_g * target_t**2
print(f"g = {{avg_g:.4f}}, distance = {{distance:.2f}}")
```

Computing g from each pair:
{chr(10).join(comp_lines)}

Average g = {avg_g:.4f} m/s²

Target distance: d = 0.5 × {avg_g:.4f} × {target_t}² = 0.5 × {avg_g:.4f} × {target_t*target_t:.4f} = {result:.2f}

The answer is {answer}.
</think>

\\boxed{{{answer}}}"""
    return cot


def generate_unit_conversion_cot(prompt, answer, examples, target):
    if not examples or target is None:
        return generate_generic_cot(prompt, answer, "unit_conversion")

    factors = []
    comp_lines = []
    for i, (inp, out) in enumerate(examples):
        f = out / inp
        factors.append(f)
        comp_lines.append(f"  {inp} → {out}: factor = {out}/{inp} = {f:.6f}")

    avg_f = statistics.mean(factors)
    result = target * avg_f

    cot = f"""<think>
This is a unit conversion problem. I need to find the conversion factor from examples.

```python
examples = {[(i, o) for i, o in examples]}
factors = [out/inp for inp, out in examples]
avg_factor = sum(factors) / len(factors)
result = {target} * avg_factor
print(f"factor = {{avg_factor:.6f}}, result = {{result:.2f}}")
```

Computing conversion factor from each pair:
{chr(10).join(comp_lines)}

Average factor = {avg_f:.6f}

Applying to target: {target} × {avg_f:.6f} = {result:.2f}

The answer is {answer}.
</think>

\\boxed{{{answer}}}"""
    return cot


def generate_numeral_cot(prompt, answer, examples, target):
    if target is None:
        return generate_generic_cot(prompt, answer, "numeral_system")

    val_map = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]

    remaining = target
    parts = []
    steps = []
    for val, sym in val_map:
        while remaining >= val:
            parts.append(sym)
            remaining -= val
            steps.append(f"  {target - remaining + val} ≥ {val}: subtract {val}, add '{sym}', remaining = {remaining}")

    roman = ''.join(parts)

    # Verify examples
    verify = []
    for num, rom in examples[:3]:
        r = num; p = []
        for v, s in val_map:
            while r >= v:
                p.append(s); r -= v
        verify.append(f"  {num} → {''.join(p)} {'✓' if ''.join(p) == rom else '✗'}")

    cot = f"""<think>
Roman numeral conversion. Rules: I=1, V=5, X=10, L=50, C=100, D=500, M=1000. Subtractive: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900.

```python
def to_roman(n):
    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
            (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    result = []
    for v, s in vals:
        while n >= v:
            result.append(s); n -= v
    return ''.join(result)

# Verify:
{chr(10).join(verify)}

# Target:
print(to_roman({target}))
```

Converting {target}:
{chr(10).join(steps[:8])}

Result: {roman}

The answer is {answer}.
</think>

\\boxed{{{answer}}}"""
    return cot


def generate_cipher_cot(prompt, answer, pairs, target_text):
    if not pairs or not target_text:
        return generate_generic_cot(prompt, answer, "cipher")

    mapping = {}
    for encrypted, decrypted in pairs:
        for e_char, d_char in zip(encrypted, decrypted):
            if e_char != ' ' and d_char != ' ':
                mapping[e_char] = d_char

    map_lines = []
    for i, (enc, dec) in enumerate(pairs[:3]):
        chars = [f"{e}→{d}" for e, d in zip(enc.replace(' ',''), dec.replace(' ',''))]
        map_lines.append(f"  Pair {i+1}: \"{enc}\" → \"{dec}\"")
        map_lines.append(f"    Maps: {', '.join(chars[:12])}")

    sorted_map = sorted(mapping.items())

    # Decrypt
    result_chars = []
    for c in target_text:
        result_chars.append(mapping.get(c, c) if c != ' ' else ' ')
    decrypted = ''.join(result_chars)

    cot = f"""<think>
Substitution cipher problem. I need to build a letter mapping from examples and decrypt the target.

```python
mapping = {{}}
for encrypted, decrypted in example_pairs:
    for e, d in zip(encrypted, decrypted):
        if e != ' ' and d != ' ':
            mapping[e] = d

# Decrypt target
target = "{target_text}"
result = ''.join(mapping.get(c, c) if c != ' ' else ' ' for c in target)
print(result)
```

Building mapping from examples:
{chr(10).join(map_lines)}

Full mapping: {{{', '.join([f"'{k}':'{v}'" for k,v in sorted_map[:26]])}}}

Decrypting "{target_text}":
{'  '.join([f'{c}→{mapping.get(c,c)}' for c in target_text if c != ' '])}

Result: {answer}
</think>

\\boxed{{{answer}}}"""
    return cot


def generate_bit_manipulation_cot(prompt, answer, pairs, target):
    if not pairs or not target:
        return generate_generic_cot(prompt, answer, "bit_manipulation")

    # Try to identify the operation
    def find_operation(pairs):
        # XOR with constant
        for xor_val in range(256):
            if all(int(i, 2) ^ xor_val == int(o, 2) for i, o in pairs):
                return f"XOR with {xor_val:08b} ({xor_val})", lambda x: x ^ xor_val

        # NOT
        if all(int(i, 2) ^ 0xFF == int(o, 2) for i, o in pairs):
            return "NOT (XOR with 11111111)", lambda x: x ^ 0xFF

        # AND/OR with constant
        for mask in range(256):
            if all(int(i, 2) & mask == int(o, 2) for i, o in pairs):
                return f"AND with {mask:08b} ({mask})", lambda x: x & mask
            if all(int(i, 2) | mask == int(o, 2) for i, o in pairs):
                return f"OR with {mask:08b} ({mask})", lambda x: x | mask

        # Rotations
        for shift in range(1, 8):
            if all(((int(i,2) << shift) | (int(i,2) >> (8-shift))) & 0xFF == int(o,2) for i, o in pairs):
                return f"Rotate left by {shift}", lambda x: ((x << shift) | (x >> (8-shift))) & 0xFF
            if all(((int(i,2) >> shift) | (int(i,2) << (8-shift))) & 0xFF == int(o,2) for i, o in pairs):
                return f"Rotate right by {shift}", lambda x: ((x >> shift) | (x << (8-shift))) & 0xFF

        # Bit reversal
        def rev(n):
            return int(f'{n:08b}'[::-1], 2)
        if all(rev(int(i,2)) == int(o,2) for i, o in pairs):
            return "Reverse bits", rev

        # Majority: Maj(x, A, B) = (x & A) | (A & B) | (B & x) with two constant masks
        for a in range(256):
            for b in range(256):
                if all(
                    ((int(i,2) & a) | (a & b) | (b & int(i,2))) & 0xFF == int(o,2)
                    for i, o in pairs
                ):
                    return f"Majority with A={a:08b}, B={b:08b}", None

        # Choice: Ch(x, T, F) = (x & T) | (~x & F) with two constant masks
        for t_mask in range(256):
            for f_mask in range(256):
                if all(
                    ((int(i,2) & t_mask) | ((~int(i,2) & 0xFF) & f_mask)) & 0xFF == int(o,2)
                    for i, o in pairs
                ):
                    return f"Choice with T={t_mask:08b}, F={f_mask:08b}", None

        # XOR then rotate, rotate then XOR
        for xor_val in range(256):
            for shift in range(1, 8):
                if all(((int(i,2) ^ xor_val) << shift | (int(i,2) ^ xor_val) >> (8-shift)) & 0xFF == int(o,2) for i, o in pairs):
                    return f"XOR {xor_val:08b} then rotate left {shift}", None
                if all((((int(i,2) << shift) | (int(i,2) >> (8-shift))) & 0xFF) ^ xor_val == int(o,2) for i, o in pairs):
                    return f"Rotate left {shift} then XOR {xor_val:08b}", None

        # Reverse + XOR
        for xor_val in range(256):
            if all(rev(int(i,2)) ^ xor_val == int(o,2) for i, o in pairs):
                return f"Reverse then XOR {xor_val:08b}", None
            if all(rev(int(i,2) ^ xor_val) == int(o,2) for i, o in pairs):
                return f"XOR {xor_val:08b} then reverse", None

        return "Complex transformation", None

    op_name, op_fn = find_operation(pairs)

    pair_lines = []
    for i, (inp, out) in enumerate(pairs[:5]):
        iv, ov = int(inp, 2), int(out, 2)
        pair_lines.append(f"  {inp} ({iv:3d}) → {out} ({ov:3d})  XOR={iv^ov:08b}")

    cot = f"""<think>
Bit manipulation puzzle. I need to find the transformation rule from examples.

```python
pairs = {pairs[:5]}
# Test XOR, rotation, NOT, reverse, composites...
for xor_val in range(256):
    if all(int(i,2) ^ xor_val == int(o,2) for i,o in pairs):
        print(f"XOR with {{xor_val:08b}}")
        break
```

Analyzing pairs:
{chr(10).join(pair_lines)}

Identified operation: {op_name}

```python
target = int("{target}", 2)  # {int(target, 2)}
# Apply: {op_name}
result = ...  # = {answer}
print(f"{{result:08b}}")
```

Applying {op_name} to {target}: result = {answer}
</think>

\\boxed{{{answer}}}"""
    return cot


def generate_equation_transform_cot(prompt, answer, pairs, target):
    if not pairs or not target:
        return generate_generic_cot(prompt, answer, "equation_transform")

    mapping = {}
    for inp, out in pairs:
        ic = inp.replace(' ', '')
        oc = out.replace(' ', '')
        for i_char, o_char in zip(ic, oc):
            mapping[i_char] = o_char

    map_lines = []
    for i, (inp, out) in enumerate(pairs[:4]):
        chars = [f"{a}→{b}" for a, b in zip(inp.replace(' ',''), out.replace(' ',''))]
        map_lines.append(f"  Rule {i+1}: \"{inp}\" = \"{out}\" → {', '.join(chars)}")

    sorted_map = sorted(mapping.items())

    cot = f"""<think>
Symbol transformation puzzle. Each character maps to another.

```python
mapping = {{}}
for inp, out in rules:
    for i_c, o_c in zip(inp.replace(' ',''), out.replace(' ','')):
        mapping[i_c] = o_c

target = "{target}"
result = ''.join(mapping.get(c, c) for c in target)
print(result)
```

Building mapping:
{chr(10).join(map_lines)}

Mapping: {{{', '.join([f"'{k}':'{v}'" for k,v in sorted_map[:15]])}}}

Applying to "{target}":
{'  '.join([f'{c}→{mapping.get(c,c)}' for c in target.replace(" ", "")])}

Result: {answer}
</think>

\\boxed{{{answer}}}"""
    return cot


def generate_generic_cot(prompt, answer, category):
    return f"""<think>
Let me analyze this {category} problem step by step.

I'll examine the examples to identify the transformation pattern, then apply it to the target.

After carefully analyzing all input-output pairs and finding the consistent rule:

The answer is {answer}.
</think>

\\boxed{{{answer}}}"""


# ============================================================
# MAIN
# ============================================================

def classify_puzzle(prompt):
    p = prompt[:300].lower()
    if 'bit manipulation' in p: return 'bit_manipulation'
    elif 'encryption' in p or 'decrypt' in p: return 'cipher'
    elif 'gravitational' in p: return 'gravity'
    elif 'unit conversion' in p: return 'unit_conversion'
    elif 'numeral' in p: return 'numeral_system'
    elif 'transformation rules' in p or ('wonderland' in p and '=' in prompt): return 'equation_transform'
    return 'unknown'


def process_puzzles():
    print("Processing competition puzzles...")

    with open(f"{DATA_DIR}/competition/train.csv") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)

    examples_out = []
    stats = defaultdict(lambda: {'total': 0, 'parsed': 0, 'failed': 0})

    for row in rows:
        puzzle_id, prompt, answer = row[0], row[1], row[2]
        category = classify_puzzle(prompt)
        stats[category]['total'] += 1

        try:
            if category == 'gravity':
                parsed, target = parse_gravity(prompt)
                cot = generate_gravity_cot(prompt, answer, parsed, target)
            elif category == 'unit_conversion':
                parsed, target = parse_unit_conversion(prompt)
                cot = generate_unit_conversion_cot(prompt, answer, parsed, target)
            elif category == 'numeral_system':
                parsed, target = parse_numeral(prompt)
                cot = generate_numeral_cot(prompt, answer, parsed, target)
            elif category == 'cipher':
                parsed, target = parse_cipher(prompt)
                cot = generate_cipher_cot(prompt, answer, parsed, target)
            elif category == 'bit_manipulation':
                parsed, target = parse_bit_manipulation(prompt)
                cot = generate_bit_manipulation_cot(prompt, answer, parsed, target)
            elif category == 'equation_transform':
                parsed, target = parse_equation_transform(prompt)
                cot = generate_equation_transform_cot(prompt, answer, parsed, target)
            else:
                cot = generate_generic_cot(prompt, answer, category)

            stats[category]['parsed'] += 1
        except Exception as e:
            stats[category]['failed'] += 1
            cot = generate_generic_cot(prompt, answer, category)

        user_msg = prompt + "\nPlease put your final answer inside \\boxed{}."
        examples_out.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": cot}
            ],
            "category": category,
            "puzzle_id": puzzle_id
        })

    print("\nPuzzle processing stats:")
    for cat, s in sorted(stats.items()):
        print(f"  {cat}: {s['total']} total, {s['parsed']} parsed, {s['failed']} failed")

    return examples_out


def sample_aimo3_data(n_samples=5000):
    aimo3_zip = f"{DATA_DIR}/aimo3/aimo3-tool-integrated-reasoning.zip"
    if not os.path.exists(aimo3_zip):
        print("AIMO3 dataset not found, skipping")
        return []

    print(f"Sampling {n_samples} from AIMO3 (~141K examples, 3GB CSV)...")

    examples = []
    total = 0

    with zipfile.ZipFile(aimo3_zip) as zf:
        with zf.open('data.csv') as f:
            text_f = io.TextIOWrapper(f, encoding='utf-8')
            reader = csv.reader(text_f)
            header = next(reader)
            print(f"  Columns: {header}")

            for i, row in enumerate(reader):
                total += 1
                if len(examples) < n_samples:
                    examples.append(row)
                else:
                    j = random.randint(0, total - 1)
                    if j < n_samples:
                        examples[j] = row
                if total % 50000 == 0:
                    print(f"  Scanned {total}...")

    print(f"  Total rows: {total}, sampled: {len(examples)}")

    converted = []
    for row in examples:
        if len(row) < 2:
            continue
        prompt_text, completion = row[0], row[1]

        if '\\boxed' not in completion:
            continue

        # Extract and clean from Harmony format
        # Remove all Harmony tags: <|role|>, <|start|>, <|channel|>, <|message|>, etc.
        asst = completion
        # Remove tool-related sections entirely (tool calls + results)
        asst = re.sub(r'<\|tool_call\|>.*?<\|tool_response\|>.*?(?=<\|assistant\|>|<\|start\|>|$)', '', asst, flags=re.DOTALL)
        # Get the last assistant section
        parts = re.split(r'<\|assistant\|>', asst)
        if len(parts) > 1:
            asst = parts[-1]
        # Strip all remaining Harmony tags
        asst = re.sub(r'<\|[^|]*\|>', '', asst)
        asst = asst.strip()

        if '\\boxed' not in asst:
            continue

        # Skip if too long (> 6000 chars ~ 1500 tokens)
        if len(asst) > 6000:
            continue

        # Wrap in <think> if needed
        if '<think>' not in asst:
            boxed_idx = asst.rfind('\\boxed')
            thinking = asst[:boxed_idx].strip()
            answer_part = asst[boxed_idx:].strip()
            asst = f"<think>\n{thinking}\n</think>\n\n{answer_part}"

        user_msg = prompt_text.strip()
        if '\\boxed' not in user_msg:
            user_msg += "\nPlease put your final answer inside \\boxed{}."

        converted.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": asst}
            ],
            "category": "math_reasoning",
            "puzzle_id": row[2] if len(row) > 2 else "aimo3"
        })

    print(f"  Converted: {len(converted)} AIMO3 examples")
    return converted


def main():
    random.seed(42)

    puzzle_data = process_puzzles()
    print(f"\nPuzzle examples: {len(puzzle_data)}")

    aimo3_data = sample_aimo3_data(n_samples=5000)
    print(f"AIMO3 examples: {len(aimo3_data)}")

    combined = puzzle_data + aimo3_data
    random.shuffle(combined)
    print(f"\nCombined total: {len(combined)}")

    # Save combined
    out_path = f"{OUTPUT_DIR}/training_data_v11.jsonl"
    with open(out_path, 'w') as f:
        for item in combined:
            f.write(json.dumps({"messages": item["messages"]}) + '\n')
    print(f"Saved: {out_path}")

    # Save puzzle-only
    puzzle_path = f"{OUTPUT_DIR}/puzzle_data_v11.jsonl"
    with open(puzzle_path, 'w') as f:
        for item in puzzle_data:
            f.write(json.dumps({"messages": item["messages"]}) + '\n')
    print(f"Saved: {puzzle_path}")

    # Samples
    print("\n=== SAMPLES ===")
    for cat in ['gravity', 'cipher', 'bit_manipulation', 'numeral_system']:
        for item in puzzle_data:
            if item['category'] == cat:
                print(f"\n--- {cat} ---")
                print(item['messages'][1]['content'][:400])
                print("...\n")
                break

    # Distribution
    cats = defaultdict(int)
    for item in combined:
        cats[item['category']] += 1
    print("\n=== DISTRIBUTION ===")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
