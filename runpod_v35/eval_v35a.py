"""Run vLLM batched inference on 60 train.csv prompts with v35a + v34 adapters.

Saves the eval JSON locally on the pod for download. Uses the same parameters
as Kaggle eval: temperature=0.0, top_p=1.0, max_tokens=7680, max_model_len=8192,
FLASHINFER backend (Blackwell sm_120 compat).

Output: /workspace/eval_v35a.json — feeds directly into tracking/validator.py locally.
"""
import os, sys, json, re, csv, time, random
from collections import Counter, defaultdict

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

print("Importing vLLM...", flush=True)
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL = "/workspace/model"
ADAPTER_V35A = "/workspace/output_v35a/nemotron-lora-adapter"
ADAPTER_V34 = "/workspace/output_v34/nemotron-lora-adapter"
TRAIN_CSV = "/workspace/data/train.csv"
OUT_JSON = "/workspace/eval_v35a.json"

N_PER_TYPE = 10  # 60 total prompts
SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'


def classify(p: str) -> str:
    fl = p.split('\n')[0].lower()
    if 'bit manipulation' in fl or '8-bit' in fl: return 'binary'
    if 'roman' in fl or 'numeral' in fl: return 'roman'
    if 'unit' in fl or 'measurement' in fl: return 'unit'
    if 'gravitational' in fl: return 'gravity'
    if 'encryption' in fl or 'cipher' in fl: return 'cipher'
    if 'transformation rules' in fl: return 'equation'
    return 'other'


def wrap_prompt(text: str) -> str:
    if not text.endswith(SUFFIX.strip()):
        text = text + SUFFIX
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def extract_boxed(text: str) -> str:
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else ""


def verify(stored, predicted):
    import math
    stored = str(stored).strip()
    predicted = str(predicted).strip()
    if not predicted: return False
    if re.fullmatch(r'[01]+', stored):
        return predicted.lower() == stored.lower()
    try:
        return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored.lower()


# Load and stratified-sample 10 per type (seed=42 matches the earlier benchmark)
print(f"Loading {TRAIN_CSV}...")
with open(TRAIN_CSV) as f:
    rows = list(csv.DictReader(f))
print(f"Loaded {len(rows)} train rows")

random.seed(42)
by_type = defaultdict(list)
for r in rows:
    t = classify(r['prompt'])
    if t == 'other': continue
    by_type[t].append(r)

sampled = []
for t in sorted(by_type.keys()):
    items = by_type[t][:]
    random.shuffle(items)
    sampled.extend(items[:N_PER_TYPE])
print(f"Sampled {len(sampled)} prompts: {Counter(classify(r['prompt']) for r in sampled)}")

prompts = [wrap_prompt(r['prompt']) for r in sampled]

# Load model with both LoRAs registered
print("\nLoading vLLM (~2 min)...", flush=True)
t0 = time.time()
llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    enable_lora=True,
    max_loras=2,
    max_lora_rank=32,
    trust_remote_code=True,
    enforce_eager=True,
    attention_backend="FLASHINFER",  # Blackwell sm_120 compat
)
print(f"Model loaded in {time.time()-t0:.1f}s")

sp = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=7680,
    stop=["<|im_end|>", "<|endoftext|>"],
)

# Run v34 first (baseline)
print(f"\n=== Running v34 on {len(prompts)} prompts ===", flush=True)
t0 = time.time()
v34_out = llm.generate(prompts, sp, lora_request=LoRARequest("v34", 1, ADAPTER_V34))
print(f"v34 generated in {time.time()-t0:.1f}s")

# Run v35a
print(f"\n=== Running v35a on {len(prompts)} prompts ===", flush=True)
t0 = time.time()
v35a_out = llm.generate(prompts, sp, lora_request=LoRARequest("v35a", 2, ADAPTER_V35A))
print(f"v35a generated in {time.time()-t0:.1f}s")

# Build the rows-format that tracking/validator.py expects (full prompts!)
rows_out = []
for i, r in enumerate(sampled):
    v34_text = v34_out[i].outputs[0].text
    v35a_text = v35a_out[i].outputs[0].text
    v34_box = extract_boxed(v34_text)
    v35a_box = extract_boxed(v35a_text)
    rows_out.append({
        "type": classify(r['prompt']),
        "expected": r['answer'],
        "prompt": r['prompt'],  # FULL prompt, not preview
        "v34_text": v34_text,
        "v34_box": v34_box,
        "v34_ok": verify(r['answer'], v34_box),
        "v35a_text": v35a_text,
        "v35a_box": v35a_box,
        "v35a_ok": verify(r['answer'], v35a_box),
    })

# Quick top-line summary
print("\n=== TOP-LINE SUMMARY ===")
v34_correct = sum(r['v34_ok'] for r in rows_out)
v35a_correct = sum(r['v35a_ok'] for r in rows_out)
print(f"v34: {v34_correct}/60 = {v34_correct/60:.3f}")
print(f"v35a: {v35a_correct}/60 = {v35a_correct/60:.3f}")
print(f"delta: {(v35a_correct-v34_correct)/60:+.3f}")

print("\nPer-type:")
for t in sorted(by_type.keys()):
    rs = [r for r in rows_out if r['type']==t]
    a31 = sum(r['v34_ok'] for r in rs)
    a34 = sum(r['v35a_ok'] for r in rs)
    print(f"  {t:<10} v34={a31}/{len(rs)} v35a={a34}/{len(rs)}  delta={(a34-a31):+d}")

with open(OUT_JSON, 'w') as f:
    json.dump(rows_out, f, indent=2)
print(f"\nSaved {len(rows_out)} rows to {OUT_JSON}")
print("DONE")
