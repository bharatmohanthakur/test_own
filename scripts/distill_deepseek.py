"""
Distill CoT reasoning from DeepSeek R1 for competition puzzles.
Generates high-quality <think>...\boxed{} training examples.
Verifies each against ground truth — only keeps correct ones.
"""

import json, os, re, math, time, sys
from openai import OpenAI

# DeepSeek R1 API
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com"
)

MODEL = "deepseek-reasoner"  # DeepSeek R1

# Competition verify function
def verify(stored_answer, predicted):
    stored_answer = str(stored_answer).strip()
    predicted = str(predicted).strip()
    if not predicted:
        return False
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except:
        return predicted.lower() == stored_answer.lower()

def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else ""

SYSTEM_PROMPT = """You are a puzzle solver. Think step by step to solve the puzzle.
Show your complete reasoning process, then give your final answer inside \\boxed{}.
For example: \\boxed{your answer}"""

SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

def generate_cot(prompt, answer, category, max_retries=2):
    """Generate CoT from DeepSeek R1 and verify answer."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt + SUFFIX}
                ],
                max_tokens=4096,
                temperature=0.0,
            )

            content = response.choices[0].message.content
            reasoning = getattr(response.choices[0].message, 'reasoning_content', '') or ''

            # Build full response with <think> tags
            if reasoning:
                full_response = f"<think>\n{reasoning}\n</think>\n\n{content}"
            else:
                full_response = content

            # Extract and verify answer
            extracted = extract_boxed(full_response)
            is_correct = verify(answer, extracted)

            return {
                "content": full_response,
                "extracted": extracted,
                "correct": is_correct,
                "reasoning_len": len(reasoning),
                "total_len": len(full_response),
                "attempt": attempt + 1,
            }

        except Exception as e:
            print(f"  Error (attempt {attempt+1}): {e}")
            time.sleep(5)

    return None

def main():
    # Load prompts
    with open("/Users/bharat/Downloads/kaggle/data/distillation_prompts.json") as f:
        prompts = json.load(f)

    print(f"Loaded {len(prompts)} prompts")

    # Output file
    output_path = "/Users/bharat/Downloads/kaggle/data/distilled_r1.jsonl"

    # Resume from existing if present
    existing = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                ex = json.loads(line)
                existing.add(ex.get("id", ""))
        print(f"Resuming: {len(existing)} already done")

    correct_count = 0
    total_count = 0

    with open(output_path, "a") as out:
        for i, item in enumerate(prompts):
            if item["id"] in existing:
                continue

            total_count += 1
            cat = item["category"]

            print(f"[{i+1}/{len(prompts)}] {cat} id={item['id'][:8]}...", end=" ")

            result = generate_cot(item["prompt"], item["answer"], cat)

            if result is None:
                print("FAILED")
                continue

            if result["correct"]:
                correct_count += 1
                # Save as training example
                example = {
                    "id": item["id"],
                    "category": cat,
                    "messages": [
                        {"role": "user", "content": item["prompt"] + SUFFIX},
                        {"role": "assistant", "content": result["content"]}
                    ],
                    "answer": item["answer"],
                    "extracted": result["extracted"],
                    "reasoning_len": result["reasoning_len"],
                }
                out.write(json.dumps(example) + "\n")
                out.flush()
                print(f"✓ ({result['reasoning_len']} chars reasoning)")
            else:
                print(f"✗ got={result['extracted']} want={item['answer']}")

            # Rate limit
            time.sleep(0.5)

            # Progress report every 50
            if total_count % 50 == 0:
                acc = correct_count / total_count if total_count > 0 else 0
                print(f"\n--- Progress: {total_count}/{len(prompts)}, accuracy: {acc:.1%} ({correct_count} correct) ---\n")

    print(f"\n{'='*60}")
    print(f"Done! {correct_count}/{total_count} correct ({correct_count/max(total_count,1):.1%})")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("Set DEEPSEEK_API_KEY environment variable first")
        print("Get a key at https://platform.deepseek.com/")
        sys.exit(1)
    main()
