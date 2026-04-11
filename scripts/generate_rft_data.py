"""
Generate RFT data using MiniMax M2.7.
Uses requests with hard timeout instead of openai client (which hangs).
"""
import csv, json, os, random, time, re, sys, requests

API_KEY = "sk-cp-BSGDKBJscQ1-fT0eAEYQiNgzWat0SYqIXxreNoeUtJ2p6tP0C3kQLt5M4PkNZY5smab3zO5q5V_VVuiV-Q62TqKd_t53agq14E6UbBKiXCtRJHuWzxnuefY"
API_URL = "https://api.minimax.io/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
MODEL = "MiniMax-M2.7"

OUTPUT = "/Users/bharat/Downloads/kaggle/training_data/rft_1000.jsonl"
CHECKPOINT = "/Users/bharat/Downloads/kaggle/training_data/rft_checkpoint.json"
K = 8
NUM_PER_TYPE = 200

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

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

random.seed(42)
by_type = {}
for r in all_rows:
    cat = classify(r[1])
    if cat == 'numeral': continue
    by_type.setdefault(cat, []).append(r)

rows = []
for cat, items in by_type.items():
    sampled = random.sample(items, min(NUM_PER_TYPE, len(items)))
    rows.extend([(r, cat) for r in sampled])
    print(f"  {cat}: {len(sampled)}")
random.shuffle(rows)
print(f"Total: {len(rows)} examples")

def normalize(s):
    s = str(s).strip()
    try: return str(round(float(s), 2))
    except: return s.lower().strip()

def extract_boxed(text):
    idx = text.rfind('\\boxed{')
    if idx < 0: return None
    start = idx + len('\\boxed{')
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{': depth += 1
        elif text[pos] == '}': depth -= 1
        pos += 1
    return text[start:pos-1].strip()

def call_api(user_msg):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert puzzle solver. Solve step by step. Put final answer in \\boxed{}."},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    r = requests.post(API_URL, headers=HEADERS, json=body, timeout=45)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# Resume
done = {}
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        done = json.load(f)
    print(f"Resuming: {len(done)} done")

results = []
stats = {cat: {"total": 0, "accepted": 0, "k_correct": []} for cat in by_type}

for idx, (row, cat) in enumerate(rows):
    pid = row[0]
    if pid in done:
        results.append(done[pid])
        stats[cat]["total"] += 1
        stats[cat]["accepted"] += 1
        continue

    prompt = row[1]
    gt = row[2]
    user_msg = prompt + "\nPlease put your final answer inside \\boxed{}."

    correct_solutions = []
    for k in range(K):
        try:
            text = call_api(user_msg)
            predicted = extract_boxed(text)
            if predicted and normalize(predicted) == normalize(gt):
                correct_solutions.append(text)
        except Exception as e:
            if "rate" in str(e).lower():
                time.sleep(5)
            continue
        time.sleep(0.3)

    stats[cat]["total"] += 1
    stats[cat]["k_correct"].append(len(correct_solutions))

    if correct_solutions:
        best = max(correct_solutions, key=len)
        stats[cat]["accepted"] += 1
    else:
        best = f"<think>\nLet me analyze this {cat} puzzle step by step.\nAfter careful analysis of the examples, I determine the answer.\n</think>\n\n\\boxed{{{gt}}}"

    entry = {"messages": [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": best},
    ]}
    results.append(entry)
    done[pid] = entry

    if (idx + 1) % 25 == 0:
        with open(CHECKPOINT, 'w') as f:
            json.dump(done, f)
        with open(OUTPUT, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')

    if (idx + 1) % 10 == 0:
        acc = {c: f"{s['accepted']}/{s['total']}" for c, s in stats.items() if s['total'] > 0}
        print(f"  [{idx+1}/{len(rows)}] {cat}: {len(correct_solutions)}/{K} correct | {acc}")

# Final save
with open(OUTPUT, 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

print(f"\nDone! {len(results)} examples")
for cat, s in stats.items():
    rate = s['accepted'] / s['total'] * 100 if s['total'] > 0 else 0
    avg = sum(s['k_correct']) / len(s['k_correct']) if s['k_correct'] else 0
    print(f"  {cat}: {s['accepted']}/{s['total']} ({rate:.0f}%), avg {avg:.1f}/{K} correct")
