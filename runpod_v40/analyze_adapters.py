"""Analyze v38a and v42 adapters: run inference on 60 diverse prompts,
compare per-type accuracy, identify what each adapter learned/fails on."""
import csv, json, os, re, math, random, time, sys, collections

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = "/workspace/model"
TRAIN_CSV = "/workspace/data/train.csv"

def classify(p):
    p = p.split('\n')[0].lower()
    if 'bit manipulation' in p: return 'binary'
    if 'roman' in p or 'numeral' in p: return 'roman'
    if 'unit' in p or 'measurement' in p: return 'unit'
    if 'gravitational' in p: return 'gravity'
    if 'encryption' in p or 'cipher' in p: return 'cipher'
    if 'transformation rules' in p: return 'equation'
    return 'other'

def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else ""

def verify(stored, predicted):
    stored, predicted = str(stored).strip(), str(predicted).strip()
    if not predicted: return False
    if re.fullmatch(r'[01]+', stored): return predicted.lower() == stored.lower()
    try: return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except: return predicted.lower() == stored.lower()

# Load prompts — 10 per type
prompts_by_type = collections.defaultdict(list)
with open(TRAIN_CSV) as f:
    for row in csv.DictReader(f):
        t = classify(row['prompt'])
        prompts_by_type[t].append(row)

random.seed(60)
selected = []
for t in ['binary','cipher','equation','gravity','roman','unit']:
    random.shuffle(prompts_by_type[t])
    selected.extend(prompts_by_type[t][:10])
random.shuffle(selected)
print(f"Selected {len(selected)} prompts ({', '.join(f'{t}:10' for t in ['binary','cipher','equation','gravity','roman','unit'])})")

SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

def evaluate_adapter(adapter_path, adapter_name):
    print(f"\n{'='*60}")
    print(f"Evaluating: {adapter_name} ({adapter_path})")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        print(f"Adapter loaded from {adapter_path}")
    else:
        print(f"NO adapter — running base model")

    results = collections.defaultdict(lambda: {"correct": 0, "total": 0, "samples": []})
    t0 = time.time()

    for i, row in enumerate(selected):
        t = classify(row['prompt'])
        messages = [{"role": "user", "content": row['prompt'] + SUFFIX}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=3500).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1024, temperature=0.0,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_boxed(completion)
        correct = verify(row['answer'], pred)

        results[t]["total"] += 1
        if correct:
            results[t]["correct"] += 1
        results[t]["samples"].append({
            "expected": row['answer'],
            "predicted": pred,
            "correct": correct,
            "completion_len": len(completion),
        })

        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(selected)}] {time.time()-t0:.0f}s")

    # Print results
    print(f"\n--- {adapter_name} Results ({time.time()-t0:.0f}s) ---")
    total_correct = 0
    total_all = 0
    for t in ['binary','cipher','equation','gravity','roman','unit']:
        r = results[t]
        pct = r['correct']/r['total']*100 if r['total'] else 0
        total_correct += r['correct']
        total_all += r['total']
        # Show failures
        failures = [s for s in r['samples'] if not s['correct']]
        fail_info = f" failures: {[f['predicted'][:20] for f in failures[:3]]}" if failures else ""
        print(f"  {t:12s}: {r['correct']:2d}/{r['total']:2d} ({pct:5.1f}%){fail_info}")
    print(f"  {'TOTAL':12s}: {total_correct}/{total_all} ({total_correct/total_all*100:.1f}%)")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return dict(results)

# Find adapters
adapters = {}
for name, path in [
    ("v38a", "/workspace/v38a_adapter"),
    ("v42", "/workspace/v42_adapter"),
]:
    cfg = os.path.join(path, "adapter_config.json")
    if os.path.exists(cfg):
        adapters[name] = path
    else:
        print(f"Adapter {name} not found at {path}")

# Run evaluations
all_results = {}
for name, path in adapters.items():
    all_results[name] = evaluate_adapter(path, name)

# Compare
if len(all_results) == 2:
    names = list(all_results.keys())
    print(f"\n{'='*60}")
    print(f"COMPARISON: {names[0]} vs {names[1]}")
    print(f"{'='*60}")
    for t in ['binary','cipher','equation','gravity','roman','unit']:
        r0 = all_results[names[0]].get(t, {"correct":0,"total":0})
        r1 = all_results[names[1]].get(t, {"correct":0,"total":0})
        delta = r1['correct'] - r0['correct']
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {t:12s}: {names[0]}={r0['correct']}/{r0['total']}, {names[1]}={r1['correct']}/{r1['total']} {arrow}{abs(delta)}")

print("\nDONE")
