"""
Generate synthetic training data for Nemotron Reasoning Challenge.
Creates puzzles matching the 6 competition types with known answers.
Output: JSONL with chat-format messages for SFT training.
"""

import json
import random
import re
import string
import math


# =============================================================================
# GRAVITY PUZZLES: d = 0.5 * g * t^2
# =============================================================================

def generate_gravity():
    g = random.uniform(5.0, 25.0)  # secret gravity constant
    num_examples = random.randint(3, 7)

    examples = []
    for _ in range(num_examples):
        t = round(random.uniform(0.5, 5.0), 2)
        d = round(0.5 * g * t * t, 2)
        examples.append((t, d))

    # Target
    t_target = round(random.uniform(0.5, 5.0), 2)
    d_target = round(0.5 * g * t_target * t_target, 2)

    prompt = "In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:\n"
    for t, d in examples:
        prompt += f"For t = {t}s, distance = {d} m\n"
    prompt += f"Now, determine the falling distance for t = {t_target}s given d = 0.5*g*t^2."

    answer = str(d_target)
    return prompt, answer


# =============================================================================
# UNIT CONVERSION: output = factor * input
# =============================================================================

def generate_unit_conversion():
    factor = random.uniform(0.3, 2.5)
    num_examples = random.randint(3, 7)

    examples = []
    for _ in range(num_examples):
        inp = round(random.uniform(5.0, 50.0), 2)
        out = round(factor * inp, 2)
        examples.append((inp, out))

    inp_target = round(random.uniform(5.0, 50.0), 2)
    out_target = round(factor * inp_target, 2)

    prompt = "In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:\n"
    for inp, out in examples:
        prompt += f"{inp} m becomes {out}\n"
    prompt += f"Now, convert the following measurement: {inp_target} m"

    answer = str(out_target)
    return prompt, answer


# =============================================================================
# NUMERAL SYSTEM (Roman numerals)
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


def generate_numeral():
    num_examples = random.randint(3, 6)

    used = set()
    examples = []
    for _ in range(num_examples):
        n = random.randint(1, 99)
        while n in used:
            n = random.randint(1, 99)
        used.add(n)
        examples.append((n, int_to_roman(n)))

    target = random.randint(1, 99)
    while target in used:
        target = random.randint(1, 99)

    prompt = "In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:\n"
    for n, r in examples:
        prompt += f"{n} -> {r}\n"
    prompt += f"Now, write the number {target} in the Wonderland numeral system."

    answer = int_to_roman(target)
    return prompt, answer


# =============================================================================
# CIPHER (substitution cipher)
# =============================================================================

WORDS = [
    "alice", "rabbit", "queen", "king", "castle", "garden", "potion", "wizard",
    "dragon", "knight", "princess", "forest", "mirror", "sword", "bridge",
    "mountain", "river", "tower", "door", "secret", "magical", "golden",
    "silver", "ancient", "dark", "bright", "wise", "brave", "the",
    "discovers", "creates", "follows", "reads", "watches", "chases",
    "imagines", "dreams", "draws", "studies", "opens", "finds",
    "near", "under", "inside", "behind", "above", "beyond",
    "cat", "dog", "bird", "mouse", "turtle", "book", "map", "key",
    "student", "teacher", "hatter", "valley", "palace", "treasure",
]


def generate_cipher():
    # Create random substitution cipher (letter -> letter)
    alphabet = list(string.ascii_lowercase)
    shuffled = alphabet[:]
    random.shuffle(shuffled)
    encrypt_map = dict(zip(alphabet, shuffled))

    def encrypt_word(word):
        return ''.join(encrypt_map.get(c, c) for c in word)

    num_examples = random.randint(3, 7)
    examples = []
    for _ in range(num_examples):
        num_words = random.randint(2, 6)
        phrase_words = random.sample(WORDS, num_words)
        plaintext = ' '.join(phrase_words)
        ciphertext = ' '.join(encrypt_word(w) for w in phrase_words)
        examples.append((ciphertext, plaintext))

    # Target
    num_target_words = random.randint(2, 4)
    # Only use words whose letters all appear in the examples
    used_letters = set()
    for ct, pt in examples:
        for c in pt.replace(' ', ''):
            used_letters.add(c)

    eligible_words = [w for w in WORDS if all(c in used_letters for c in w)]
    if len(eligible_words) < num_target_words:
        eligible_words = WORDS

    target_words = random.sample(eligible_words, min(num_target_words, len(eligible_words)))
    target_plain = ' '.join(target_words)
    target_cipher = ' '.join(encrypt_word(w) for w in target_words)

    prompt = "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:\n"
    for ct, pt in examples:
        prompt += f"{ct} -> {pt}\n"
    prompt += f"Now, decrypt the following text: {target_cipher}"

    answer = target_plain
    return prompt, answer


# =============================================================================
# BIT MANIPULATION
# =============================================================================

def generate_bit_manipulation():
    # Pick a random bit operation
    op_type = random.choice([
        'xor_const', 'not', 'rotate_left', 'rotate_right',
        'bit_reverse', 'xor_then_rotate', 'rotate_then_xor',
        'not_then_xor', 'swap_nibbles', 'swap_nibbles_xor',
    ])

    xor_val = random.randint(0, 255)
    rot_k = random.randint(1, 7)

    def rot_left(n, k):
        return ((n << k) | (n >> (8 - k))) & 0xFF

    def rot_right(n, k):
        return ((n >> k) | (n << (8 - k))) & 0xFF

    def bit_reverse(n):
        return int(format(n, '08b')[::-1], 2)

    def swap_nibbles(n):
        return ((n & 0x0F) << 4) | ((n & 0xF0) >> 4)

    def apply_op(n):
        if op_type == 'xor_const':
            return n ^ xor_val
        elif op_type == 'not':
            return ~n & 0xFF
        elif op_type == 'rotate_left':
            return rot_left(n, rot_k)
        elif op_type == 'rotate_right':
            return rot_right(n, rot_k)
        elif op_type == 'bit_reverse':
            return bit_reverse(n)
        elif op_type == 'xor_then_rotate':
            return rot_left(n ^ xor_val, rot_k)
        elif op_type == 'rotate_then_xor':
            return rot_left(n, rot_k) ^ xor_val
        elif op_type == 'not_then_xor':
            return (~n & 0xFF) ^ xor_val
        elif op_type == 'swap_nibbles':
            return swap_nibbles(n)
        elif op_type == 'swap_nibbles_xor':
            return swap_nibbles(n) ^ xor_val
        return n

    num_examples = random.randint(6, 10)
    used = set()
    examples = []
    for _ in range(num_examples):
        inp = random.randint(0, 255)
        while inp in used:
            inp = random.randint(0, 255)
        used.add(inp)
        out = apply_op(inp)
        examples.append((format(inp, '08b'), format(out, '08b')))

    target_inp = random.randint(0, 255)
    while target_inp in used:
        target_inp = random.randint(0, 255)
    target_out = format(apply_op(target_inp), '08b')

    prompt = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:\n"""
    for inp, out in examples:
        prompt += f"{inp} -> {out}\n"
    prompt += f"\nNow, determine the output for: {format(target_inp, '08b')}"

    answer = target_out
    return prompt, answer


# =============================================================================
# EQUATION TRANSFORM
# =============================================================================

def generate_equation_transform():
    # Create a secret character mapping for symbols
    symbols = list("!@#$%^&*()_+-=[]{}|;':\",./<>?`~")
    random.shuffle(symbols)

    # Pick a subset for mapping
    num_symbols = random.randint(8, 15)
    src_symbols = symbols[:num_symbols]
    dst_symbols = symbols[num_symbols:2*num_symbols] if len(symbols) >= 2*num_symbols else random.sample(symbols, num_symbols)

    char_map = dict(zip(src_symbols, dst_symbols))

    def apply_transform(s):
        # Apply character substitution
        return ''.join(char_map.get(c, c) for c in s)

    num_examples = random.randint(3, 6)
    examples = []
    for _ in range(num_examples):
        # Generate random equation-like string
        length = random.randint(3, 6)
        lhs = ''.join(random.choice(src_symbols) for _ in range(length))
        rhs = apply_transform(lhs)
        examples.append((lhs, rhs))

    # Target
    target_len = random.randint(3, 5)
    target_lhs = ''.join(random.choice(src_symbols) for _ in range(target_len))
    target_rhs = apply_transform(target_lhs)

    prompt = "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:\n"
    for lhs, rhs in examples:
        prompt += f"{lhs} = {rhs}\n"
    prompt += f"Now, determine the result for: {target_lhs}"

    answer = target_rhs
    return prompt, answer


# =============================================================================
# FORMAT AS CHAT FOR SFT
# =============================================================================

def format_chat(prompt, answer):
    """Format as chat message with \boxed{} answer for SFT training."""
    user_msg = prompt + '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

    # Create a reasoning chain then boxed answer
    assistant_msg = f"Let me analyze the pattern step by step.\n\nAfter examining the examples carefully, I can determine the answer.\n\n\\boxed{{{answer}}}"

    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


# =============================================================================
# ALSO FORMAT REAL TRAINING DATA
# =============================================================================

def load_real_training_data():
    """Load real competition training data and format for SFT."""
    import csv
    examples = []
    with open('/Users/bharat/Downloads/kaggle/data/train.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            prompt, answer = row[1], row[2]
            examples.append(format_chat(prompt, answer))
    return examples


# =============================================================================
# MAIN
# =============================================================================

GENERATORS = {
    'gravity': generate_gravity,
    'unit_conversion': generate_unit_conversion,
    'numeral_system': generate_numeral,
    'cipher': generate_cipher,
    'bit_manipulation': generate_bit_manipulation,
    'equation_transform': generate_equation_transform,
}


def main():
    random.seed(42)

    # Generate synthetic data — 3000 per type = 18000 total
    synthetic = []
    per_type = 3000
    for cat, gen_fn in GENERATORS.items():
        print(f"Generating {per_type} {cat} examples...")
        for i in range(per_type):
            try:
                prompt, answer = gen_fn()
                synthetic.append(format_chat(prompt, answer))
            except Exception as e:
                pass

    print(f"\nGenerated {len(synthetic)} synthetic examples")

    # Load real training data
    real = load_real_training_data()
    print(f"Loaded {len(real)} real training examples")

    # Combine: real data (repeated 3x for emphasis) + synthetic
    all_data = real * 3 + synthetic
    random.shuffle(all_data)

    print(f"Total training examples: {len(all_data)}")

    # Save as JSONL
    output_path = '/Users/bharat/Downloads/kaggle/training_data/sft_train.jsonl'
    with open(output_path, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n')

    print(f"Saved to {output_path}")

    # Also save just the real data
    real_path = '/Users/bharat/Downloads/kaggle/training_data/real_train.jsonl'
    with open(real_path, 'w') as f:
        for item in real:
            f.write(json.dumps(item) + '\n')
    print(f"Real data saved to {real_path}")


if __name__ == '__main__':
    main()
