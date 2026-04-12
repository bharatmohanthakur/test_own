#!/usr/bin/env python3
"""Analyze quality of distilled CoT data."""
import json

fallback_count = 0
api_count = 0
type_counts = {}
total_len = 0
has_boxed = 0
has_think = 0

with open('/Users/bharat/Downloads/kaggle/training_data/distilled_cot_1000.jsonl') as f:
    for line in f:
        d = json.loads(line)
        content = d['messages'][1]['content']
        total_len += len(content)

        if '\\boxed{' in content:
            has_boxed += 1
        if '<think>' in content:
            has_think += 1

        # Detect puzzle type from user message
        user_msg = d['messages'][0]['content']
        if 'bit manipulation' in user_msg:
            t = 'bit_manipulation'
        elif 'encryption' in user_msg or 'decrypt' in user_msg:
            t = 'cipher'
        elif 'gravitational' in user_msg:
            t = 'gravity'
        elif 'unit conversion' in user_msg:
            t = 'unit_conversion'
        elif 'numeral system' in user_msg:
            t = 'numeral_system'
        elif 'transformation rules' in user_msg:
            t = 'equation_transform'
        else:
            t = 'unknown'
        type_counts[t] = type_counts.get(t, 0) + 1

        if len(content) < 200:
            fallback_count += 1
        else:
            api_count += 1

print('=== Output Quality Report ===')
print(f'Total examples: 1000')
print(f'Avg response length: {total_len/1000:.0f} chars')
print(f'Has \\boxed: {has_boxed}')
print(f'Has <think>: {has_think}')
print(f'Longer responses (likely API): {api_count}')
print(f'Short responses (possible fallback): {fallback_count}')
print(f'Type distribution:')
for t, c in sorted(type_counts.items()):
    print(f'  {t}: {c}')
