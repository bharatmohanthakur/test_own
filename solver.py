"""
NVIDIA Nemotron Model Reasoning Challenge — Deterministic Solvers
Solves 4 of 6 puzzle types algorithmically. Remaining 2 need LLM.
"""

import csv
import re
import statistics
from collections import Counter


# =============================================================================
# CLASSIFIER
# =============================================================================

def classify_puzzle(prompt):
    p = prompt[:120].lower()
    if 'bit manipulation' in p:
        return 'bit_manipulation'
    if 'encryption' in p or 'decrypt' in p:
        return 'cipher'
    if 'gravitational' in p:
        return 'gravity'
    if 'unit conversion' in p:
        return 'unit_conversion'
    if 'numeral' in p:
        return 'numeral_system'
    if 'transformation rules' in p:
        return 'equation_transform'
    return 'unknown'


# =============================================================================
# SOLVER: GRAVITY  (d = 0.5 * g * t^2)
# =============================================================================

def solve_gravity(prompt):
    # Extract example pairs
    pairs = re.findall(r't\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)', prompt)
    if not pairs:
        return None

    # Compute g from each example: g = 2*d / t^2
    gs = []
    for t_str, d_str in pairs:
        t, d = float(t_str), float(d_str)
        if t > 0:
            gs.append(2 * d / (t * t))
    if not gs:
        return None
    g = statistics.median(gs)

    # Extract target t
    target = re.search(r'for\s+t\s*=\s*([\d.]+)s?\s+given', prompt)
    if not target:
        return None
    t_target = float(target.group(1))

    result = 0.5 * g * t_target * t_target
    return f"{result:.2f}"


# =============================================================================
# SOLVER: UNIT CONVERSION  (output = factor * input)
# =============================================================================

def solve_unit_conversion(prompt):
    # Extract example pairs
    pairs = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    if not pairs:
        return None

    # Compute conversion factor
    factors = []
    for inp_str, out_str in pairs:
        inp, out = float(inp_str), float(out_str)
        if inp > 0:
            factors.append(out / inp)
    if not factors:
        return None
    factor = statistics.median(factors)

    # Extract target
    target = re.search(r'convert the following measurement:\s*([\d.]+)', prompt)
    if not target:
        return None
    inp_target = float(target.group(1))

    result = factor * inp_target
    return f"{result:.2f}"


# =============================================================================
# SOLVER: NUMERAL SYSTEM (currently all seem to be Roman numerals)
# =============================================================================

def int_to_roman(num):
    vals = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]
    result = ''
    for val, sym in vals:
        while num >= val:
            result += sym
            num -= val
    return result


def solve_numeral(prompt):
    # Check if examples look like Roman numerals
    examples = re.findall(r'(\d+)\s*->\s*([A-Z]+)', prompt)
    if not examples:
        return None

    # Verify they're Roman numerals
    is_roman = all(all(c in 'IVXLCDM' for c in rom) for _, rom in examples)
    if not is_roman:
        return None

    # Extract target number
    target = re.search(r'write the number\s+(\d+)', prompt)
    if not target:
        return None

    num = int(target.group(1))
    return int_to_roman(num)


# =============================================================================
# SOLVER: CIPHER (substitution cipher)
# =============================================================================

def solve_cipher(prompt):
    # Extract encrypted -> decrypted pairs
    lines = prompt.strip().split('\n')
    mapping = {}  # encrypted_char -> decrypted_char

    example_lines = []
    target_line = None

    for line in lines:
        line = line.strip()
        if '->' in line:
            parts = line.split('->')
            if len(parts) == 2:
                encrypted = parts[0].strip()
                decrypted = parts[1].strip()
                example_lines.append((encrypted, decrypted))
        elif line.lower().startswith('now, decrypt'):
            match = re.search(r'decrypt the following text:\s*(.*)', line, re.IGNORECASE)
            if match:
                target_line = match.group(1).strip()

    if not example_lines or not target_line:
        return None

    # Build character mapping from examples
    for encrypted, decrypted in example_lines:
        enc_words = encrypted.split()
        dec_words = decrypted.split()
        if len(enc_words) != len(dec_words):
            continue
        for ew, dw in zip(enc_words, dec_words):
            if len(ew) != len(dw):
                continue
            for ec, dc in zip(ew, dw):
                if ec in mapping:
                    if mapping[ec] != dc:
                        pass  # conflict — keep first mapping
                else:
                    mapping[ec] = dc

    # Decrypt target
    result = []
    for ch in target_line:
        if ch == ' ':
            result.append(' ')
        elif ch in mapping:
            result.append(mapping[ch])
        else:
            result.append(ch)  # unmapped — keep as is
    return ''.join(result)


# =============================================================================
# SOLVER: BIT MANIPULATION (brute force common operations)
# =============================================================================

def solve_bit_manipulation(prompt):
    # Extract input -> output pairs
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    if not pairs:
        return None

    # Extract target
    target_match = re.search(r'determine the output for:\s*([01]{8})', prompt)
    if not target_match:
        return None
    target = int(target_match.group(1), 2)

    examples = [(int(i, 2), int(o, 2)) for i, o in pairs]

    # Try common single operations
    candidates = []

    # NOT
    if all(o == (~i & 0xFF) for i, o in examples):
        candidates.append(~target & 0xFF)

    # XOR with constant
    for xor_val in range(256):
        if all(o == (i ^ xor_val) for i, o in examples):
            candidates.append(target ^ xor_val)

    # Bit reverse
    def bit_reverse(n):
        return int(format(n, '08b')[::-1], 2)
    if all(o == bit_reverse(i) for i, o in examples):
        candidates.append(bit_reverse(target))

    # Rotate left by k
    for k in range(1, 8):
        def rot_left(n, k=k):
            return ((n << k) | (n >> (8 - k))) & 0xFF
        if all(o == rot_left(i) for i, o in examples):
            candidates.append(rot_left(target))

    # Rotate right by k
    for k in range(1, 8):
        def rot_right(n, k=k):
            return ((n >> k) | (n << (8 - k))) & 0xFF
        if all(o == rot_right(i) for i, o in examples):
            candidates.append(rot_right(target))

    # Shift left by k (with mask)
    for k in range(1, 8):
        if all(o == ((i << k) & 0xFF) for i, o in examples):
            candidates.append((target << k) & 0xFF)

    # Shift right by k
    for k in range(1, 8):
        if all(o == (i >> k) for i, o in examples):
            candidates.append(target >> k)

    # NOT then XOR
    for xor_val in range(256):
        if all(o == ((~i & 0xFF) ^ xor_val) for i, o in examples):
            candidates.append((~target & 0xFF) ^ xor_val)

    # Bit reverse then XOR
    for xor_val in range(256):
        if all(o == (bit_reverse(i) ^ xor_val) for i, o in examples):
            candidates.append(bit_reverse(target) ^ xor_val)

    # Rotate then XOR
    for k in range(1, 8):
        for xor_val in range(256):
            def rot_left(n, k=k):
                return ((n << k) | (n >> (8 - k))) & 0xFF
            if all(o == (rot_left(i) ^ xor_val) for i, o in examples):
                candidates.append(rot_left(target) ^ xor_val)
                break

    # XOR then rotate
    for xor_val in range(256):
        for k in range(1, 8):
            def rot_left(n, k=k):
                return ((n << k) | (n >> (8 - k))) & 0xFF
            if all(o == rot_left(i ^ xor_val) for i, o in examples):
                candidates.append(rot_left(target ^ xor_val))
                break

    # Swap nibbles (upper 4 and lower 4 bits)
    def swap_nibbles(n):
        return ((n & 0x0F) << 4) | ((n & 0xF0) >> 4)
    if all(o == swap_nibbles(i) for i, o in examples):
        candidates.append(swap_nibbles(target))

    # Swap nibbles then XOR
    for xor_val in range(256):
        if all(o == (swap_nibbles(i) ^ xor_val) for i, o in examples):
            candidates.append(swap_nibbles(target) ^ xor_val)

    if candidates:
        # Return most common candidate
        c = Counter(candidates)
        return format(c.most_common(1)[0][0], '08b')

    return None


# =============================================================================
# MAIN SOLVER
# =============================================================================

def solve(prompt):
    cat = classify_puzzle(prompt)
    if cat == 'gravity':
        return solve_gravity(prompt), cat
    elif cat == 'unit_conversion':
        return solve_unit_conversion(prompt), cat
    elif cat == 'numeral_system':
        return solve_numeral(prompt), cat
    elif cat == 'cipher':
        return solve_cipher(prompt), cat
    elif cat == 'bit_manipulation':
        return solve_bit_manipulation(prompt), cat
    elif cat == 'equation_transform':
        return None, cat  # needs LLM
    return None, cat


# =============================================================================
# EVALUATE ON TRAINING DATA
# =============================================================================

if __name__ == '__main__':
    with open('/Users/bharat/Downloads/kaggle/train.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)

    results = {}
    for row in rows:
        id_, prompt, answer = row[0], row[1], row[2]
        prediction, cat = solve(prompt)

        if cat not in results:
            results[cat] = {'correct': 0, 'wrong': 0, 'unsolved': 0, 'total': 0, 'wrong_examples': []}
        results[cat]['total'] += 1

        if prediction is None:
            results[cat]['unsolved'] += 1
        elif prediction.strip() == answer.strip():
            results[cat]['correct'] += 1
        else:
            results[cat]['wrong'] += 1
            if len(results[cat]['wrong_examples']) < 3:
                results[cat]['wrong_examples'].append({
                    'id': id_,
                    'predicted': prediction,
                    'actual': answer,
                    'prompt_snippet': prompt[:200]
                })

    print("=" * 70)
    print(f"{'Category':<22} {'Correct':>8} {'Wrong':>8} {'Unsolved':>8} {'Total':>8} {'Acc%':>8}")
    print("=" * 70)
    total_correct = 0
    total_all = 0
    for cat in sorted(results.keys()):
        r = results[cat]
        acc = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
        print(f"{cat:<22} {r['correct']:>8} {r['wrong']:>8} {r['unsolved']:>8} {r['total']:>8} {acc:>7.1f}%")
        total_correct += r['correct']
        total_all += r['total']
    print("=" * 70)
    print(f"{'TOTAL':<22} {total_correct:>8} {'':>8} {'':>8} {total_all:>8} {total_correct/total_all*100:>7.1f}%")

    # Show wrong examples
    print("\n\nWRONG PREDICTIONS (samples):")
    for cat, r in sorted(results.items()):
        if r['wrong_examples']:
            print(f"\n--- {cat} ---")
            for ex in r['wrong_examples']:
                print(f"  ID={ex['id']}  predicted={ex['predicted']}  actual={ex['actual']}")
