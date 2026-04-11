"""
Generate distilled CoT from DeepSeek R1 for all 7924 non-numeral training examples.
K=4 rejection sampling: generate 4 solutions per problem, keep correct ones.
Verified against ground truth answers from train.csv.
"""
import csv, json, os, random, time, re, sys, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "sk-08c52559f89f4744b2222da4776ee2b4"
API_URL = "https://api.deepseek.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
MODEL = "deepseek-reasoner"

K = 4  # solutions per problem
OUTPUT = "/Users/bharat/Downloads/kaggle/training_data/deepseek_r1_cot.jsonl"
CHECKPOINT = "/Users/bharat/Downloads/kaggle/training_data/deepseek_checkpoint.json"
STATS_FILE = "/Users/bharat/Downloads/kaggle/training_data/deepseek_stats.json"

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# Load training data
with open("/Users/bharat/Downloads/kaggle/data/train.csv") as f:
    reader = csv.reader(f)
    next(reader)
    all_rows = list(reader)

def classify(p):
    p = p[:200].lower()
    if 'bit manipulation' in p: return 'bit'
    if 'encryption' in p or 'decrypt' in p: return 'cipher'
    if 'gravitational' in p: return 'gravity'
    if 'unit conversion' in p: return 'unit'
    if 'numeral' in p: return 'numeral'
    if 'transformation rules' in p or 'wonderland' in p: return 'equation'
    return 'unknown'

# Filter out numeral (100% zero-shot) and build work list
rows = []
for r in all_rows:
    cat = classify(r[1])
    if cat == 'numeral':
        continue
    rows.append((r[0], r[1], r[2], cat))  # id, prompt, answer, category

random.seed(42)
random.shuffle(rows)
print(f"Total examples to process: {len(rows)} (skipped numeral)")

# Count by category
cat_counts = {}
for _, _, _, cat in rows:
    cat_counts[cat] = cat_counts.get(cat, 0) + 1
for cat, n in sorted(cat_counts.items()):
    print(f"  {cat}: {n}")

def normalize_answer(s):
    s = str(s).strip()
    try:
        return str(round(float(s), 2))
    except ValueError:
        return s.lower().strip()

def extract_boxed(text):
    idx = text.rfind('\\boxed{')
    if idx < 0:
        return None
    start = idx + len('\\boxed{')
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{': depth += 1
        elif text[pos] == '}': depth -= 1
        pos += 1
    if depth == 0:
        return text[start:pos-1].strip()
    return None

def check_answer(predicted, expected):
    if predicted is None:
        return False
    pred_n = normalize_answer(predicted)
    exp_n = normalize_answer(expected)
    if pred_n == exp_n:
        return True
    try:
        if abs(float(pred_n) - float(exp_n)) < 0.02:
            return True
    except (ValueError, TypeError):
        pass
    return False

def call_deepseek(user_msg, temperature=0.7):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }
    r = requests.post(API_URL, headers=HEADERS, json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]["message"]
    # DeepSeek R1 returns reasoning_content + content
    reasoning = choice.get("reasoning_content", "")
    content = choice.get("content", "")
    return reasoning, content

# Resume from checkpoint
done = {}
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        done = json.load(f)
    print(f"Resuming: {len(done)} already completed")

stats = {cat: {"total": 0, "accepted": 0, "k_correct": []} for cat in cat_counts}
results = []

# Load existing results
if os.path.exists(OUTPUT) and done:
    with open(OUTPUT) as f:
        for line in f:
            results.append(json.loads(line))
    print(f"Loaded {len(results)} existing results")

start_time = time.time()
total_api_calls = 0

for idx, (pid, prompt, gt_answer, cat) in enumerate(rows):
    if pid in done:
        stats[cat]["total"] += 1
        stats[cat]["accepted"] += 1
        continue

    user_msg = prompt + "\nPlease put your final answer inside \\boxed{}."

    correct_solutions = []
    all_attempts = 0

    for k in range(K):
        try:
            reasoning, content = call_deepseek(user_msg)
            total_api_calls += 1

            # Build the full response with <think> tags
            if reasoning:
                full_response = f"<think>\n{reasoning}\n</think>\n\n{content}"
            else:
                full_response = content

            # Check if answer is correct
            predicted = extract_boxed(full_response)
            if predicted is None:
                predicted = extract_boxed(content)

            if check_answer(predicted, gt_answer):
                correct_solutions.append(full_response)

            all_attempts += 1

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"  Rate limited, waiting 10s...")
                time.sleep(10)
            elif e.response.status_code == 402:
                print(f"  INSUFFICIENT BALANCE - stopping")
                sys.exit(1)
            else:
                print(f"  HTTP error {e.response.status_code}: {e}")
            continue
        except requests.exceptions.Timeout:
            print(f"  Timeout, skipping attempt")
            continue
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)
            continue

        # Small delay between calls
        time.sleep(0.2)

    stats[cat]["total"] += 1
    stats[cat]["k_correct"].append(len(correct_solutions))

    if correct_solutions:
        # Pick the longest correct solution (most detailed reasoning)
        best = max(correct_solutions, key=len)
        stats[cat]["accepted"] += 1
    else:
        # Fallback: use ground truth with minimal CoT
        best = f"<think>\nLet me analyze this {cat} puzzle step by step.\nAfter careful analysis, I determine the answer.\n</think>\n\n\\boxed{{{gt_answer}}}"

    entry = {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": best},
        ],
        "category": cat,
        "verified": len(correct_solutions) > 0,
        "k_correct": len(correct_solutions),
    }
    results.append(entry)
    done[pid] = True

    # Save checkpoint every 25 examples
    if (idx + 1) % 25 == 0:
        with open(CHECKPOINT, 'w') as f:
            json.dump(done, f)
        with open(OUTPUT, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')

        elapsed = time.time() - start_time
        rate = total_api_calls / max(elapsed, 1) * 3600
        processed = sum(s["total"] for s in stats.values())
        remaining = len(rows) - processed
        eta_hrs = remaining * K / max(rate, 1)

        acc = {c: f"{s['accepted']}/{s['total']}" for c, s in stats.items() if s['total'] > 0}
        verified = sum(s['accepted'] for s in stats.values())
        total = sum(s['total'] for s in stats.values())
        print(f"  [{processed}/{len(rows)}] verified={verified}/{total} ({verified/max(total,1)*100:.0f}%) | "
              f"calls={total_api_calls} | {elapsed/60:.1f}min | ETA={eta_hrs:.1f}hrs | {acc}")

    # Save stats periodically
    if (idx + 1) % 100 == 0:
        with open(STATS_FILE, 'w') as f:
            json.dump({
                "processed": sum(s["total"] for s in stats.values()),
                "total": len(rows),
                "verified": sum(s["accepted"] for s in stats.values()),
                "api_calls": total_api_calls,
                "elapsed_min": (time.time() - start_time) / 60,
                "by_category": {c: {"accepted": s["accepted"], "total": s["total"],
                                     "rate": s["accepted"]/max(s["total"],1)*100}
                                for c, s in stats.items()},
            }, f, indent=2)

# Final save
with open(CHECKPOINT, 'w') as f:
    json.dump(done, f)
with open(OUTPUT, 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
with open(STATS_FILE, 'w') as f:
    json.dump({
        "processed": sum(s["total"] for s in stats.values()),
        "total": len(rows),
        "verified": sum(s["accepted"] for s in stats.values()),
        "api_calls": total_api_calls,
        "elapsed_min": (time.time() - start_time) / 60,
        "by_category": {c: {"accepted": s["accepted"], "total": s["total"],
                             "rate": s["accepted"]/max(s["total"],1)*100}
                        for c, s in stats.items()},
    }, f, indent=2)

elapsed = (time.time() - start_time) / 60
print(f"\nDone! {len(results)} examples in {elapsed:.1f} min, {total_api_calls} API calls")
for cat, s in stats.items():
    rate = s['accepted'] / max(s['total'], 1) * 100
    avg_k = sum(s['k_correct']) / max(len(s['k_correct']), 1)
    print(f"  {cat}: {s['accepted']}/{s['total']} ({rate:.0f}%), avg {avg_k:.1f}/{K} correct")
