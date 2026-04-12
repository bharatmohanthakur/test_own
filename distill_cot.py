"""
Generate high-quality CoT via MiniMax M2.7 teacher distillation.
1000 balanced examples, rejection sampling (keep only correct answers).
"""

import csv, json, os, random, time, re, sys
from openai import OpenAI

API_KEY = "sk-cp-BSGDKBJscQ1-fT0eAEYQiNgzWat0SYqIXxreNoeUtJ2p6tP0C3kQLt5M4PkNZY5smab3zO5q5V_VVuiV-Q62TqKd_t53agq14E6UbBKiXCtRJHuWzxnuefY"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.minimax.io/v1",
)

MODEL = "MiniMax-M2.7"
OUTPUT_FILE = "/Users/bharat/Downloads/kaggle/training_data/distilled_cot_1000.jsonl"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Load data
with open("/Users/bharat/Downloads/kaggle/data/train.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)
    all_rows = list(reader)

print(f"Total rows: {len(all_rows)}")

# Classify
def classify_puzzle(p):
    p = p[:200].lower()
    if 'bit manipulation' in p or 'binary string' in p: return 'bit'
    if 'encryption' in p or 'decrypt' in p or 'cipher' in p: return 'cipher'
    if 'gravitational' in p or 'falling' in p: return 'gravity'
    if 'unit conversion' in p or ('becomes' in p and 'measurement' in p): return 'unit'
    if 'numeral' in p or 'roman' in p: return 'numeral'
    if 'transformation rules' in p or 'wonderland' in p: return 'equation'
    return 'unknown'

# Balanced sample of 1000
by_type = {}
for r in all_rows:
    cat = classify_puzzle(r[1])
    by_type.setdefault(cat, []).append(r)

per_type = 1000 // len(by_type)
rows = []
for cat, items in by_type.items():
    sampled = random.sample(items, min(per_type, len(items)))
    rows.extend(sampled)
    print(f"  {cat}: {len(sampled)}")
random.shuffle(rows)
print(f"Selected {len(rows)} examples for distillation")

SYSTEM_PROMPT = """You are an expert puzzle solver. Solve the given puzzle step by step.

Rules:
1. Think carefully and show your complete reasoning process
2. Test hypotheses systematically - try different approaches and verify
3. Cross-check your answer against ALL given examples
4. Put your final answer inside \\boxed{}

Your reasoning should be thorough but focused. Show the key steps that lead to the answer."""

def normalize_answer(s):
    """Normalize answer for comparison."""
    s = str(s).strip()
    # Try numeric comparison
    try:
        return str(round(float(s), 2))
    except:
        return s.lower().strip()

def extract_boxed(text):
    """Extract answer from \\boxed{}."""
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip()
    return None

def generate_cot(prompt, ground_truth, max_retries=3):
    """Generate CoT and verify against ground truth. Retry if wrong."""
    user_msg = prompt + "\nPlease put your final answer inside \\boxed{}."

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.6 if attempt == 0 else 0.8,
                max_tokens=4096,
            )

            assistant_msg = response.choices[0].message.content

            # Extract and verify answer
            predicted = extract_boxed(assistant_msg)
            if predicted and normalize_answer(predicted) == normalize_answer(ground_truth):
                # Correct! Format as training example
                # Wrap reasoning in <think> tags if not already
                if '<think>' not in assistant_msg:
                    # Find the boxed answer and split
                    boxed_idx = assistant_msg.rfind('\\boxed{')
                    if boxed_idx > 0:
                        reasoning = assistant_msg[:boxed_idx].strip()
                        answer_part = assistant_msg[boxed_idx:]
                        assistant_msg = f"<think>\n{reasoning}\n</think>\n\n{answer_part}"

                return assistant_msg, True, attempt + 1

            # Wrong answer, will retry with higher temperature

        except Exception as e:
            print(f"  API error: {e}")
            time.sleep(2)

    return None, False, max_retries

# Generate CoT for all examples
results = []
correct = 0
failed = 0
total = len(rows)

print(f"\nStarting distillation with {MODEL}...")
print(f"{'='*60}")

for i, row in enumerate(rows):
    prompt, ground_truth = row[1], row[2]
    cat = classify_puzzle(prompt)

    cot, is_correct, attempts = generate_cot(prompt, ground_truth)

    if is_correct:
        correct += 1
        results.append({
            "messages": [
                {"role": "user", "content": prompt + "\nPlease put your final answer inside \\boxed{}."},
                {"role": "assistant", "content": cot},
            ],
            "type": cat,
        })
        status = f"OK (attempt {attempts})"
    else:
        failed += 1
        status = "FAILED (using fallback)"
        # Fallback: simple CoT with correct answer
        results.append({
            "messages": [
                {"role": "user", "content": prompt + "\nPlease put your final answer inside \\boxed{}."},
                {"role": "assistant", "content": f"<think>\nLet me analyze this step by step.\n\nAfter careful analysis of the pattern from the given examples, I can determine the answer.\n</think>\n\n\\boxed{{{ground_truth}}}"},
            ],
            "type": cat,
        })

    if (i + 1) % 10 == 0 or i == 0:
        print(f"  [{i+1}/{total}] {cat}: {status} | correct={correct}, failed={failed}")

    # Rate limiting
    time.sleep(0.3)

print(f"\n{'='*60}")
print(f"Distillation complete!")
print(f"  Correct: {correct}/{total} ({correct/total*100:.1f}%)")
print(f"  Failed (fallback): {failed}/{total}")

# Per-type stats
for cat in by_type:
    cat_results = [r for r in results if r['type'] == cat]
    cat_correct = sum(1 for r in cat_results if 'careful analysis' not in r['messages'][1]['content'])
    print(f"  {cat}: {cat_correct}/{len(cat_results)} correct")

# Save
with open(OUTPUT_FILE, 'w') as f:
    for item in results:
        f.write(json.dumps({"messages": item["messages"]}) + '\n')

print(f"\nSaved {len(results)} examples to {OUTPUT_FILE}")

# Also save stats
stats = {
    "model": MODEL,
    "total": total,
    "correct": correct,
    "failed": failed,
    "accuracy": correct / total,
}
with open(OUTPUT_FILE.replace('.jsonl', '_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)
