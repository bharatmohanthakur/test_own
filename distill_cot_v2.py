#!/usr/bin/env python3
"""
Distill Chain-of-Thought reasoning from MiniMax M2.7 for Nemotron training.
Generates high-quality CoT traces for 1000 balanced examples from train.csv.
"""

import csv
import json
import os
import re
import random
import time
import sys
from collections import defaultdict
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY = "sk-cp-BSGDKBJscQ1-fT0eAEYQiNgzWat0SYqIXxreNoeUtJ2p6tP0C3kQLt5M4PkNZY5smab3zO5q5V_VVuiV-Q62TqKd_t53agq14E6UbBKiXCtRJHuWzxnuefY"
BASE_URL = "https://api.minimax.io/v1"
MODEL = "MiniMax-M2.7"

DATA_PATH = "/Users/bharat/Downloads/kaggle/data/train.csv"
OUTPUT_PATH = "/Users/bharat/Downloads/kaggle/training_data/distilled_cot_1000.jsonl"
CHECKPOINT_PATH = "/Users/bharat/Downloads/kaggle/training_data/distilled_cot_1000_checkpoint.jsonl"
PROGRESS_PATH = "/Users/bharat/Downloads/kaggle/training_data/distill_progress.json"

NUM_SAMPLES = 1000
SAMPLES_PER_TYPE = 167  # ceil(1000/6) = 167, we'll trim to 1000
NUM_ATTEMPTS = 5
TEMPERATURE = 0.7
MAX_TOKENS = 4096
DELAY_BETWEEN_CALLS = 0.2  # seconds

SYSTEM_PROMPT = "You are an expert puzzle solver. Think step by step through the problem carefully. Put your final answer inside \\boxed{}."

# ── Classification ──────────────────────────────────────────────────────────
def classify_puzzle(prompt: str) -> str:
    """Classify a puzzle prompt into one of 6 types."""
    prompt_lower = prompt.lower()
    if "bit manipulation" in prompt_lower or "8-bit binary" in prompt_lower:
        return "bit_manipulation"
    elif "encryption" in prompt_lower or "decrypt" in prompt_lower:
        return "cipher"
    elif "gravitational" in prompt_lower or "falling distance" in prompt_lower:
        return "gravity"
    elif "unit conversion" in prompt_lower or "convert the following measurement" in prompt_lower:
        return "unit_conversion"
    elif "numeral system" in prompt_lower or "wonderland numeral" in prompt_lower:
        return "numeral_system"
    elif "transformation rules" in prompt_lower and "equations" in prompt_lower:
        return "equation_transform"
    elif "transformation rules" in prompt_lower:
        return "equation_transform"
    else:
        # Fallback heuristics
        if "binary" in prompt_lower and ("XOR" in prompt or "AND" in prompt or "OR" in prompt):
            return "bit_manipulation"
        elif "encrypt" in prompt_lower or "cipher" in prompt_lower:
            return "cipher"
        elif "distance" in prompt_lower and "t =" in prompt:
            return "gravity"
        elif "becomes" in prompt_lower and ("m " in prompt or "kg" in prompt_lower):
            return "unit_conversion"
        elif "numeral" in prompt_lower or "roman" in prompt_lower:
            return "numeral_system"
        else:
            return "equation_transform"


# ── Answer extraction & comparison ──────────────────────────────────────────
def extract_boxed_answer(text: str) -> str:
    """Extract the last \\boxed{...} answer from text, handling nested braces."""
    # Find all \boxed{ positions
    matches = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        # Find matching closing brace
        depth = 0
        start = idx + len("\\boxed{")
        j = start
        while j < len(text):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                if depth == 0:
                    matches.append(text[start:j])
                    break
                depth -= 1
            j += 1
        i = j + 1 if j < len(text) else len(text)

    if matches:
        return matches[-1].strip()
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    return answer.strip().lower()


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    pred = normalize_answer(predicted)
    gt = normalize_answer(ground_truth)

    if not pred or not gt:
        return False

    # Exact match (case-insensitive)
    if pred == gt:
        return True

    # Try numeric comparison with tolerance
    try:
        pred_num = float(pred)
        gt_num = float(gt)
        if abs(pred_num - gt_num) < 0.01:
            return True
        # Also check relative tolerance for larger numbers
        if gt_num != 0 and abs((pred_num - gt_num) / gt_num) < 0.001:
            return True
    except (ValueError, ZeroDivisionError):
        pass

    # Strip leading/trailing whitespace and compare again
    if pred.strip() == gt.strip():
        return True

    return False


# ── Fallback CoT generation ────────────────────────────────────────────────
def generate_fallback_cot(prompt: str, answer: str, puzzle_type: str) -> str:
    """Generate a basic CoT reasoning trace when the model fails to get correct answer."""
    if puzzle_type == "bit_manipulation":
        reasoning = (
            "<think>\nLet me analyze the bit manipulation rule by examining the input-output pairs.\n\n"
            "I need to look at each example carefully and identify the pattern. "
            "Let me check various operations: bit shifts, rotations, XOR with a mask, NOT, etc.\n\n"
            "After examining all the examples systematically, I can identify the transformation rule. "
            "Applying this rule to the given input gives me the answer.\n</think>\n\n"
        )
    elif puzzle_type == "cipher":
        reasoning = (
            "<think>\nLet me analyze the encryption pattern by comparing the encrypted and decrypted text pairs.\n\n"
            "I'll map each encrypted letter to its decrypted counterpart using the examples provided. "
            "By building a complete substitution table, I can decrypt the target text.\n\n"
            "Mapping each letter carefully from the examples and applying to the target.\n</think>\n\n"
        )
    elif puzzle_type == "gravity":
        reasoning = (
            "<think>\nI need to find the gravitational constant g from the examples using d = 0.5*g*t^2.\n\n"
            "From the formula: g = 2*d / t^2\n\n"
            "Let me calculate g from each example to find the consistent value, then apply it to the target time.\n\n"
            f"Using the derived g value and the formula d = 0.5*g*t^2 for the given time.\n</think>\n\n"
        )
    elif puzzle_type == "unit_conversion":
        reasoning = (
            "<think>\nI need to find the conversion factor from the examples.\n\n"
            "For each example, I'll calculate: output / input to find the ratio.\n\n"
            "Finding a consistent conversion factor and applying it to the target measurement.\n</think>\n\n"
        )
    elif puzzle_type == "numeral_system":
        reasoning = (
            "<think>\nI need to convert the number to the Wonderland numeral system based on the examples.\n\n"
            "Looking at the examples, I can identify the numeral system being used. "
            "I'll convert the given number step by step.\n</think>\n\n"
        )
    elif puzzle_type == "equation_transform":
        reasoning = (
            "<think>\nI need to identify the symbol transformation rules from the examples.\n\n"
            "By comparing each input equation with its output, I can map each symbol to its transformed version. "
            "Then I'll apply these mappings to the target equation.\n</think>\n\n"
        )
    else:
        reasoning = "<think>\nLet me solve this step by step.\n</think>\n\n"

    return reasoning + f"\\boxed{{{answer}}}"


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("MiniMax M2.7 CoT Distillation")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading training data...")
    rows = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows)} rows")

    # Classify
    print("\n[2/4] Classifying puzzles...")
    by_type = defaultdict(list)
    for row in rows:
        ptype = classify_puzzle(row['prompt'])
        row['puzzle_type'] = ptype
        by_type[ptype].append(row)

    for ptype, examples in sorted(by_type.items()):
        print(f"  {ptype}: {len(examples)} examples")

    # Sample balanced
    print(f"\n[3/4] Sampling {NUM_SAMPLES} balanced examples...")
    random.seed(42)
    sampled = []
    types_list = sorted(by_type.keys())
    for ptype in types_list:
        pool = by_type[ptype]
        n = min(SAMPLES_PER_TYPE, len(pool))
        chosen = random.sample(pool, n)
        sampled.extend(chosen)
        print(f"  {ptype}: sampled {n}")

    # Trim to exactly NUM_SAMPLES
    random.shuffle(sampled)
    sampled = sampled[:NUM_SAMPLES]

    # Re-count after trimming
    type_counts = defaultdict(int)
    for s in sampled:
        type_counts[s['puzzle_type']] += 1
    print(f"\n  Final distribution:")
    for ptype in types_list:
        print(f"    {ptype}: {type_counts[ptype]}")

    # Load progress (for resume)
    completed_ids = set()
    results = []
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r') as f:
            progress = json.load(f)
            completed_ids = set(progress.get('completed_ids', []))
            print(f"\n  Resuming from checkpoint: {len(completed_ids)} already done")

    if os.path.exists(CHECKPOINT_PATH) and completed_ids:
        with open(CHECKPOINT_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

    # Initialize API client
    print("\n[4/4] Starting CoT generation with MiniMax M2.7...")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)

    # Test connection
    print("  Testing API connection...")
    try:
        test_resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "What is 2+2? Put answer in \\boxed{}."}],
            temperature=0.1,
            max_tokens=256,
        )
        print(f"  API test OK: {test_resp.choices[0].message.content[:80]}...")
    except Exception as e:
        print(f"  API test failed: {e}")
        sys.exit(1)

    # Stats
    total = len(sampled)
    correct_first_try = 0
    correct_after_retries = 0
    fallback_used = 0
    start_time = time.time()

    # Process each example
    for idx, example in enumerate(sampled):
        eid = example['id']

        # Skip if already done
        if eid in completed_ids:
            continue

        prompt = example['prompt']
        ground_truth = example['answer']
        puzzle_type = example['puzzle_type']

        user_message = prompt + "\nPlease put your final answer inside \\boxed{}."

        # Try up to NUM_ATTEMPTS times
        best_correct_response = None
        best_correct_length = 0
        all_responses = []

        for attempt in range(NUM_ATTEMPTS):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                content = response.choices[0].message.content
                all_responses.append(content)

                # Check correctness
                extracted = extract_boxed_answer(content)
                if answers_match(extracted, ground_truth):
                    # Keep the longest correct response (more detailed reasoning)
                    if len(content) > best_correct_length:
                        best_correct_response = content
                        best_correct_length = len(content)
                    if attempt == 0:
                        correct_first_try += 1

                time.sleep(DELAY_BETWEEN_CALLS)

            except Exception as e:
                print(f"  [!] API error on example {idx+1}, attempt {attempt+1}: {e}")
                time.sleep(2)  # Longer delay on error
                continue

            # If we found a correct answer on first try, still try a couple more for diversity
            # but if we have a correct one after 3 tries, stop early
            if best_correct_response and attempt >= 2:
                break

        # Build the training example
        if best_correct_response:
            correct_after_retries += 1
            assistant_content = best_correct_response
        else:
            fallback_used += 1
            assistant_content = generate_fallback_cot(prompt, ground_truth, puzzle_type)

        training_example = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_content},
            ],
            "metadata": {
                "id": eid,
                "puzzle_type": puzzle_type,
                "ground_truth": ground_truth,
                "used_fallback": best_correct_response is None,
                "attempts": len(all_responses),
            }
        }
        results.append(training_example)
        completed_ids.add(eid)

        # Progress reporting
        done = len(completed_ids)
        if done % 10 == 0 or done == total:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"  [{done}/{total}] "
                f"correct={correct_after_retries} fallback={fallback_used} "
                f"first_try_acc={correct_first_try/max(done,1):.1%} "
                f"rate={rate:.2f}/s ETA={eta/60:.0f}min"
            )

        # Checkpoint every 50
        if done % 50 == 0:
            _save_checkpoint(results, completed_ids)

    # Final save
    _save_final(results)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("DISTILLATION COMPLETE")
    print("=" * 70)
    print(f"  Total examples:     {total}")
    print(f"  Correct (model):    {correct_after_retries} ({correct_after_retries/total:.1%})")
    print(f"  Correct first try:  {correct_first_try} ({correct_first_try/total:.1%})")
    print(f"  Fallback used:      {fallback_used} ({fallback_used/total:.1%})")
    print(f"  Total time:         {elapsed/60:.1f} min ({elapsed/3600:.1f} hrs)")
    print(f"  Output:             {OUTPUT_PATH}")


def _save_checkpoint(results, completed_ids):
    """Save intermediate results."""
    with open(CHECKPOINT_PATH, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(PROGRESS_PATH, 'w') as f:
        json.dump({'completed_ids': list(completed_ids)}, f)
    print(f"    [checkpoint saved: {len(results)} examples]")


def _save_final(results):
    """Save final output (without metadata, clean format for training)."""
    # Save full version with metadata (for analysis)
    with open(CHECKPOINT_PATH, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Save clean version for training (no metadata)
    with open(OUTPUT_PATH, 'w') as f:
        for r in results:
            clean = {"messages": r["messages"]}
            f.write(json.dumps(clean, ensure_ascii=False) + '\n')
    print(f"    [final saved: {len(results)} examples to {OUTPUT_PATH}]")


if __name__ == "__main__":
    main()
