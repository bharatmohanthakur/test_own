"""
Parallel distillation from DeepSeek R1.
10 concurrent API calls for 10x speed.
"""

import json, os, re, math, time, sys, asyncio
from openai import AsyncOpenAI

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

MODEL = "deepseek-reasoner"
MAX_CONCURRENT = 10
SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

def verify(stored_answer, predicted):
    stored_answer = str(stored_answer).strip()
    predicted = str(predicted).strip()
    if not predicted: return False
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        return math.isclose(float(stored_answer), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except:
        return predicted.lower() == stored_answer.lower()

def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else ""

async def generate_one(item, semaphore):
    async with semaphore:
        try:
            r = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Solve the puzzle step by step. Put final answer in \\boxed{}."},
                    {"role": "user", "content": item["prompt"] + SUFFIX}
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            content = r.choices[0].message.content or ""
            reasoning = getattr(r.choices[0].message, 'reasoning_content', '') or ''

            if reasoning:
                full = f"<think>\n{reasoning}\n</think>\n\n{content}"
            else:
                full = content

            extracted = extract_boxed(full)
            correct = verify(item["answer"], extracted)

            status = "✓" if correct else "✗"
            print(f"  {status} {item['category']} id={item['id'][:8]} got={extracted[:20]} want={item['answer'][:20]} reasoning={len(reasoning)}", flush=True)

            if correct:
                return {
                    "id": item["id"],
                    "category": item["category"],
                    "messages": [
                        {"role": "user", "content": item["prompt"] + SUFFIX},
                        {"role": "assistant", "content": full}
                    ],
                    "answer": item["answer"],
                    "extracted": extracted,
                    "reasoning_len": len(reasoning),
                }
            return None

        except Exception as e:
            print(f"  ERROR {item['category']} id={item['id'][:8]}: {e}", flush=True)
            return None

async def main():
    with open("/Users/bharat/Downloads/kaggle/data/distillation_prompts.json") as f:
        prompts = json.load(f)

    # Load existing to skip
    output_path = "/Users/bharat/Downloads/kaggle/data/distilled_r1.jsonl"
    existing = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                existing.add(json.loads(line).get("id", ""))

    todo = [p for p in prompts if p["id"] not in existing]
    print(f"Total: {len(prompts)}, Already done: {len(existing)}, Todo: {len(todo)}", flush=True)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Process in batches
    BATCH = 20
    correct_total = 0

    for batch_start in range(0, len(todo), BATCH):
        batch = todo[batch_start:batch_start + BATCH]
        print(f"\n--- Batch {batch_start//BATCH + 1} ({batch_start+1}-{batch_start+len(batch)}/{len(todo)}) ---", flush=True)

        t0 = time.time()
        tasks = [generate_one(item, semaphore) for item in batch]
        results = await asyncio.gather(*tasks)

        correct = [r for r in results if r is not None]
        correct_total += len(correct)

        with open(output_path, "a") as f:
            for r in correct:
                f.write(json.dumps(r) + "\n")

        elapsed = time.time() - t0
        print(f"  Batch done: {len(correct)}/{len(batch)} correct, {elapsed:.0f}s, total correct: {correct_total}", flush=True)

    print(f"\nDONE! Total correct: {correct_total}/{len(todo)}", flush=True)

if __name__ == "__main__":
    if not API_KEY:
        print("Set DEEPSEEK_API_KEY"); sys.exit(1)
    asyncio.run(main())
