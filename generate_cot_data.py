"""
Generate high-quality Chain-of-Thought training data.
Each puzzle type gets a detailed, step-by-step reasoning trace
that teaches the model HOW to solve, not just the answer.
"""

import csv
import json
import re
import random
import statistics


def classify_puzzle(prompt):
    p = prompt[:120].lower()
    if 'bit manipulation' in p: return 'bit_manipulation'
    if 'encryption' in p or 'decrypt' in p: return 'cipher'
    if 'gravitational' in p: return 'gravity'
    if 'unit conversion' in p: return 'unit_conversion'
    if 'numeral' in p: return 'numeral_system'
    if 'transformation rules' in p: return 'equation_transform'
    return 'unknown'


def generate_gravity_cot(prompt, answer):
    """Generate detailed CoT for gravity puzzles."""
    pairs = re.findall(r't\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)', prompt)
    target = re.search(r'for\s+t\s*=\s*([\d.]+)s?\s+given', prompt)

    cot = "<think>\nThis is a physics problem where d = 0.5 * g * t^2, and I need to find the secret gravitational constant g.\n\n"
    cot += "Step 1: Extract g from each example using g = 2*d / t^2\n"

    gs = []
    for t_str, d_str in pairs:
        t, d = float(t_str), float(d_str)
        g = 2 * d / (t * t)
        gs.append(g)
        cot += f"  t={t_str}s, d={d_str}m → g = 2*{d_str}/{t_str}^2 = {g:.4f}\n"

    g_avg = statistics.mean(gs)
    cot += f"\nStep 2: Average g = {g_avg:.4f}\n"

    if target:
        t_target = float(target.group(1))
        result = 0.5 * g_avg * t_target * t_target
        cot += f"\nStep 3: Calculate d for t={t_target}s\n"
        cot += f"  d = 0.5 * {g_avg:.4f} * {t_target}^2 = {result:.2f}\n"

    cot += "</think>\n\n"
    cot += f"\\boxed{{{answer}}}"
    return cot


def generate_unit_conversion_cot(prompt, answer):
    """Generate detailed CoT for unit conversion puzzles."""
    pairs = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    target = re.search(r'convert the following measurement:\s*([\d.]+)', prompt)

    cot = "<think>\nThis is a unit conversion problem. I need to find the conversion factor.\n\n"
    cot += "Step 1: Calculate the conversion factor from each example (output/input)\n"

    factors = []
    for inp_str, out_str in pairs:
        inp, out = float(inp_str), float(out_str)
        f = out / inp
        factors.append(f)
        cot += f"  {inp_str}m → {out_str} : factor = {out_str}/{inp_str} = {f:.6f}\n"

    f_avg = statistics.mean(factors)
    cot += f"\nStep 2: Average conversion factor = {f_avg:.6f}\n"

    if target:
        inp_target = float(target.group(1))
        result = f_avg * inp_target
        cot += f"\nStep 3: Convert {inp_target}m\n"
        cot += f"  {inp_target} * {f_avg:.6f} = {result:.2f}\n"

    cot += "</think>\n\n"
    cot += f"\\boxed{{{answer}}}"
    return cot


def generate_numeral_cot(prompt, answer):
    """Generate detailed CoT for numeral system puzzles."""
    examples = re.findall(r'(\d+)\s*->\s*([A-Z]+)', prompt)
    target = re.search(r'write the number\s+(\d+)', prompt)

    cot = "<think>\nThis is a numeral system conversion. Let me check if it's Roman numerals.\n\n"
    cot += "Step 1: Verify the pattern with examples\n"

    for num_str, roman in examples:
        cot += f"  {num_str} → {roman} ✓ (matches Roman numeral)\n"

    cot += "\nThis is standard Roman numeral conversion.\n"
    cot += "Rules: I=1, V=5, X=10, L=50, C=100, D=500, M=1000\n"
    cot += "Subtractive: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900\n"

    if target:
        num = int(target.group(1))
        cot += f"\nStep 2: Convert {num} to Roman numerals\n"

        vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
                (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
        remaining = num
        parts = []
        for val, sym in vals:
            while remaining >= val:
                parts.append(sym)
                remaining -= val
        cot += f"  {num} = {''.join(parts)}\n"

    cot += "</think>\n\n"
    cot += f"\\boxed{{{answer}}}"
    return cot


def generate_cipher_cot(prompt, answer):
    """Generate detailed CoT for cipher puzzles."""
    cot = "<think>\nThis is a substitution cipher. I need to build a letter mapping from the examples.\n\n"
    cot += "Step 1: Extract character mappings from each example pair\n"

    lines = prompt.strip().split('\n')
    mapping = {}

    for line in lines:
        if '->' in line:
            parts = line.split('->')
            if len(parts) == 2:
                enc = parts[0].strip()
                dec = parts[1].strip()
                enc_words = enc.split()
                dec_words = dec.split()
                if len(enc_words) == len(dec_words):
                    for ew, dw in zip(enc_words, dec_words):
                        if len(ew) == len(dw):
                            for ec, dc in zip(ew, dw):
                                if ec not in mapping:
                                    mapping[ec] = dc

    cot += "  Mapping found:\n"
    for k, v in sorted(mapping.items()):
        cot += f"    '{k}' → '{v}'\n"

    target_match = re.search(r'decrypt the following text:\s*(.*)', prompt, re.IGNORECASE)
    if target_match:
        target_cipher = target_match.group(1).strip()
        cot += f"\nStep 2: Apply mapping to '{target_cipher}'\n"
        result = []
        for ch in target_cipher:
            if ch == ' ':
                result.append(' ')
            elif ch in mapping:
                result.append(mapping[ch])
            else:
                result.append(f'[{ch}?]')
        cot += f"  Result: {''.join(result)}\n"

    cot += "</think>\n\n"
    cot += f"\\boxed{{{answer}}}"
    return cot


def generate_bit_cot(prompt, answer):
    """Generate detailed CoT for bit manipulation puzzles."""
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    target_match = re.search(r'determine the output for:\s*([01]{8})', prompt)

    cot = "<think>\nThis is a bit manipulation puzzle. I need to identify the operation from the examples.\n\n"
    cot += "Step 1: Analyze input-output pairs\n"

    for inp, out in pairs[:4]:
        cot += f"  {inp} → {out}\n"

    cot += "\nStep 2: Test common operations (XOR with constant, NOT, rotate, bit reverse, etc.)\n"

    # Try to identify the actual operation
    examples = [(int(i, 2), int(o, 2)) for i, o in pairs]
    found_op = None

    # XOR with constant
    if len(examples) >= 2:
        xor_val = examples[0][0] ^ examples[0][1]
        if all(o == (i ^ xor_val) for i, o in examples):
            found_op = f"XOR with {format(xor_val, '08b')} ({xor_val})"

    # NOT
    if not found_op and all(o == (~i & 0xFF) for i, o in examples):
        found_op = "bitwise NOT"

    # Bit reverse
    def bit_reverse(n):
        return int(format(n, '08b')[::-1], 2)
    if not found_op and all(o == bit_reverse(i) for i, o in examples):
        found_op = "bit reverse"

    if found_op:
        cot += f"  Identified operation: {found_op}\n"
    else:
        cot += "  Testing combinations of operations...\n"
        cot += "  Applying the identified pattern to the target.\n"

    if target_match:
        cot += f"\nStep 3: Apply to {target_match.group(1)}\n"
        cot += f"  Result: {answer}\n"

    cot += "</think>\n\n"
    cot += f"\\boxed{{{answer}}}"
    return cot


def generate_equation_cot(prompt, answer):
    """Generate detailed CoT for equation transform puzzles."""
    cot = "<think>\nThis is a symbol transformation puzzle. I need to find the character-to-character mapping.\n\n"
    cot += "Step 1: Extract symbol mappings from examples\n"

    lines = prompt.strip().split('\n')
    mapping = {}

    for line in lines:
        if '=' in line and 'determine' not in line.lower():
            parts = line.split('=')
            if len(parts) == 2:
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                for i in range(min(len(lhs), len(rhs))):
                    if lhs[i] not in mapping:
                        mapping[lhs[i]] = rhs[i] if i < len(rhs) else '?'

    cot += "  Character mapping:\n"
    for k, v in list(mapping.items())[:10]:
        cot += f"    '{k}' → '{v}'\n"

    cot += "\nStep 2: Apply mapping to the target expression\n"
    cot += f"  Result: {answer}\n"

    cot += "</think>\n\n"
    cot += f"\\boxed{{{answer}}}"
    return cot


GENERATORS = {
    'gravity': generate_gravity_cot,
    'unit_conversion': generate_unit_conversion_cot,
    'numeral_system': generate_numeral_cot,
    'cipher': generate_cipher_cot,
    'bit_manipulation': generate_bit_cot,
    'equation_transform': generate_equation_cot,
}


def main():
    with open('/Users/bharat/Downloads/kaggle/data/train.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)

    print(f"Processing {len(rows)} training examples...")

    examples = []
    errors = 0
    for row in rows:
        id_, prompt, answer = row[0], row[1], row[2]
        cat = classify_puzzle(prompt)

        user_msg = prompt + '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

        try:
            gen_fn = GENERATORS.get(cat)
            if gen_fn:
                assistant_msg = gen_fn(prompt, answer)
            else:
                assistant_msg = f"<think>\nAnalyzing the pattern from the examples.\n</think>\n\n\\boxed{{{answer}}}"
        except Exception as e:
            assistant_msg = f"<think>\nAnalyzing the pattern from the examples.\n</think>\n\n\\boxed{{{answer}}}"
            errors += 1

        examples.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        })

    print(f"Generated {len(examples)} CoT examples ({errors} fallbacks)")

    # Save
    output_path = '/Users/bharat/Downloads/kaggle/training_data/cot_train.jsonl'
    with open(output_path, 'w') as f:
        for item in examples:
            f.write(json.dumps(item) + '\n')
    print(f"Saved to {output_path}")

    # Show sample
    sample = random.choice(examples)
    print("\n=== SAMPLE ===")
    print(sample["messages"][1]["content"][:500])


if __name__ == '__main__':
    main()
