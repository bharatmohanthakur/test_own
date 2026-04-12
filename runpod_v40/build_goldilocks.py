"""Build Goldilocks GRPO data: run v38a on 500 prompts, 4 completions each.
Keep prompts where model gets 1-3 out of 4 correct (25-75% = learnable zone).
"""
import csv, json, os, re, math, random, time, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = "/workspace/model"
ADAPTER_PATH = "/workspace/v38a_adapter"
TRAIN_CSV = "/workspace/data/train.csv"
OUTPUT_PATH = "/workspace/goldilocks.jsonl"

NUM_PROMPTS = 500
NUM_COMPLETIONS = 4
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.9

print("Loading model + v38a adapter...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=False)
model.set_adapter("default")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
prompts = []
with open(TRAIN_CSV) as f:
    for row in csv.DictReader(f):
        prompts.append({"prompt": row["prompt"] + SUFFIX, "answer": row["answer"]})
random.seed(42)
random.shuffle(prompts)
prompts = prompts[:NUM_PROMPTS]

def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else ""

def verify(stored, predicted):
    stored, predicted = str(stored).strip(), str(predicted).strip()
    if not predicted: return False
    if re.fullmatch(r'[01]+', stored): return predicted.lower() == stored.lower()
    try: return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except: return predicted.lower() == stored.lower()

print(f"Generating {NUM_COMPLETIONS} completions for {NUM_PROMPTS} prompts...")
goldilocks = []
stats = {"easy": 0, "goldilocks": 0, "hard": 0, "total": 0}
t0 = time.time()

for i, p in enumerate(prompts):
    messages = [{"role": "user", "content": p["prompt"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    correct_count = 0
    for _ in range(NUM_COMPLETIONS):
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                                 do_sample=True, top_p=0.95, pad_token_id=tokenizer.pad_token_id)
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_boxed(completion)
        if verify(p["answer"], pred):
            correct_count += 1
    stats["total"] += 1
    if correct_count == 0 or correct_count == NUM_COMPLETIONS:
        bucket = "easy" if correct_count == NUM_COMPLETIONS else "hard"
        stats[bucket] += 1
    else:
        stats["goldilocks"] += 1
        goldilocks.append({
            "prompt": [{"role": "user", "content": p["prompt"]}],
            "answer": p["answer"],
            "correct_rate": correct_count / NUM_COMPLETIONS,
        })
    if (i + 1) % 25 == 0:
        elapsed = time.time() - t0
        print(f"  [{i+1}/{NUM_PROMPTS}] goldilocks={stats['goldilocks']}, easy={stats['easy']}, hard={stats['hard']}, {elapsed:.0f}s")

with open(OUTPUT_PATH, "w") as f:
    for g in goldilocks:
        f.write(json.dumps(g) + "\n")

print(f"\n=== Goldilocks Results ===")
print(f"Total: {stats['total']}, Easy: {stats['easy']}, Goldilocks: {stats['goldilocks']}, Hard: {stats['hard']}")
print(f"Saved {len(goldilocks)} Goldilocks prompts to {OUTPUT_PATH}")
print(f"Time: {(time.time()-t0)/60:.1f} min")
