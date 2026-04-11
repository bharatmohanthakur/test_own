"""
GRPO v22 — Reinforcement Learning on top of best SFT adapter
Difficulty-aware data: Goldilocks zone prompts (30-70% accuracy)
Uses competition verify() as reward function
"""

import sys, os, subprocess, glob, json, random, time, zipfile, gc, re, math

# ============================================================
# 1. INSTALL FROM OFFLINE WHEELS
# ============================================================
print("Installing packages...")

PKG_DIR = None
for candidate in [
    "/kaggle/input/datasets/mayukh18/nemotron-packages/packages",
    "/kaggle/input/nemotron-packages/packages",
]:
    if os.path.exists(candidate):
        PKG_DIR = candidate
        break

if PKG_DIR:
    subprocess.run(
        f"pip install -q --no-index --find-links {PKG_DIR} "
        f"unsloth trl peft transformers datasets accelerate bitsandbytes",
        shell=True
    )
    for pattern in ["causal_conv1d*.whl", "mamba_ssm*.whl"]:
        wheels = sorted(glob.glob(os.path.join(PKG_DIR, "..", pattern)) +
                       glob.glob(os.path.join(PKG_DIR, pattern)))
        if wheels:
            subprocess.run(f"pip install -q {wheels[-1]}", shell=True)
else:
    print("ERROR: nemotron-packages not found!")
    sys.exit(1)

# ============================================================
# 2. IMPORTS
# ============================================================
import torch
import kagglehub
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset as HFDataset
import csv

print(f"Packages loaded. CUDA: {torch.cuda.is_available()}")

# ============================================================
# 3. CONFIG
# ============================================================
OUTPUT_DIR = "/kaggle/working"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "nemotron-lora-adapter")
os.makedirs(ADAPTER_DIR, exist_ok=True)

LORA_RANK = 32
LORA_ALPHA = 64
MAX_SEQ_LEN = 4096
NUM_GENERATIONS = 4       # completions per prompt (lower = less VRAM)
MAX_COMPLETION_LEN = 2048 # max tokens per completion
MAX_PROMPT_LEN = 512
LR = 5e-6                 # RL needs much lower LR than SFT
MAX_STEPS = 200            # ~200 steps is enough for GRPO
BATCH_SIZE = 1
GRAD_ACCUM = 4
RANDOM_SEED = 42
TEMPERATURE = 0.9          # need diversity for exploration

# ============================================================
# 4. LOAD COMPETITION DATA (prompts + answers for reward)
# ============================================================
print("Loading competition data...")

train_paths = glob.glob("/kaggle/input/competitions/*/train.csv") + \
              glob.glob("/kaggle/input/*/train.csv")
if not train_paths:
    print("ERROR: train.csv not found")
    sys.exit(1)

train_path = train_paths[0]
print(f"Using: {train_path}")

rows = []
with open(train_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Total problems: {len(rows)}")

# Classify puzzle type
def classify(prompt):
    p = prompt[:500].lower()
    if 'bit manipulation' in p or '8-bit binary' in p: return 'bit_manipulation'
    if 'encrypt' in p or 'cipher' in p or 'secret language' in p: return 'cipher'
    if 'gravitational' in p or 'gravity' in p: return 'gravity'
    if 'unit conversion' in p or 'secret unit' in p: return 'unit_conversion'
    if 'numeral' in p or 'roman' in p: return 'numeral'
    if 'transformation rules' in p: return 'equation'
    return 'other'

# Weight toward WEAK categories (Goldilocks zone after SFT)
# Base model: bit_manipulation 9.6%, equation 13%, cipher 36%, gravity 66%
# After SFT these should be ~30-70% = perfect for GRPO
CATEGORY_WEIGHTS = {
    'bit_manipulation': 150,
    'equation': 120,
    'cipher': 100,
    'gravity': 50,
    'unit_conversion': 40,
    'numeral': 20,
    'other': 20,
}

random.seed(RANDOM_SEED)
categorized = {}
for row in rows:
    cat = classify(row.get('prompt', ''))
    if cat not in categorized:
        categorized[cat] = []
    categorized[cat].append(row)

selected = []
for cat, target_n in CATEGORY_WEIGHTS.items():
    available = categorized.get(cat, [])
    if available:
        n = min(target_n, len(available))
        selected.extend(random.sample(available, n))

random.shuffle(selected)
print(f"Selected {len(selected)} problems for GRPO")
for cat, n in CATEGORY_WEIGHTS.items():
    actual = sum(1 for r in selected if classify(r.get('prompt','')) == cat)
    print(f"  {cat}: {actual}")

# Build dataset with prompts
BOXED_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

grpo_data = []
for row in selected:
    prompt = row.get('prompt', '')
    answer = row.get('answer', '')
    grpo_data.append({
        "prompt": [
            {"role": "user", "content": prompt + BOXED_SUFFIX}
        ],
        "answer": answer,
    })

dataset = HFDataset.from_list(grpo_data)
print(f"GRPO dataset: {len(dataset)} prompts")

# ============================================================
# 5. REWARD FUNCTION (competition verify())
# ============================================================
def verify(stored_answer, predicted):
    """Competition-exact reward function."""
    stored_answer = str(stored_answer).strip()
    predicted = str(predicted).strip()
    if not predicted:
        return False
    # Binary string exact match
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except:
        return predicted.lower() == stored_answer.lower()

def extract_boxed(text):
    """Extract answer from \\boxed{...}"""
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else ""

# Reward functions for GRPO
def correctness_reward(prompts, completions, answer, **kwargs):
    """Main reward: +2 if correct, 0 if wrong."""
    responses = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    rewards = []
    for resp, ans in zip(responses, answer):
        extracted = extract_boxed(resp)
        if verify(ans, extracted):
            rewards.append(2.0)
        else:
            rewards.append(0.0)
    return rewards

def format_reward(completions, **kwargs):
    """Reward for using \\boxed{} format."""
    responses = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    rewards = []
    for resp in responses:
        if '\\boxed{' in resp and '}' in resp.split('\\boxed{')[-1]:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def thinking_reward(completions, **kwargs):
    """Reward for showing reasoning before answer."""
    responses = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    rewards = []
    for resp in responses:
        if '<think>' in resp or len(resp) > 200:
            rewards.append(0.25)
        else:
            rewards.append(0.0)
    return rewards

# ============================================================
# 6. LOAD MODEL + SFT ADAPTER
# ============================================================
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
print(f"Model at: {MODEL_PATH}")

# TODO: Load best SFT adapter here if available
# For now, start from base model with fresh LoRA
# When we have a good SFT adapter, load it:
# model = FastLanguageModel.from_pretrained("path/to/sft/adapter", ...)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    dtype=None,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # GRPO needs left padding for generation

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,
    target_modules=["in_proj", "out_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=RANDOM_SEED,
)
model.print_trainable_parameters()

# ============================================================
# 7. GRPO TRAINING
# ============================================================
start_time = time.time()

training_args = GRPOConfig(
    output_dir=os.path.join(OUTPUT_DIR, "grpo_output"),
    learning_rate=LR,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=5,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_generations=NUM_GENERATIONS,
    max_prompt_length=MAX_PROMPT_LEN,
    max_completion_length=MAX_COMPLETION_LEN,
    max_grad_norm=0.1,
    temperature=TEMPERATURE,
    max_steps=MAX_STEPS,
    save_steps=MAX_STEPS,
    report_to="none",
    seed=RANDOM_SEED,
    # GRPO specific
    loss_type="grpo",
)

print(f"\n{'='*60}")
print(f"GRPO v22 — Reinforcement Learning")
print(f"{'='*60}")
print(f"  Prompts: {len(dataset)}")
print(f"  Generations per prompt: {NUM_GENERATIONS}")
print(f"  Max steps: {MAX_STEPS}")
print(f"  LR: {LR}")
print(f"  Rewards: correctness(+2) + format(+0.5) + thinking(+0.25)")
print(f"{'='*60}\n")

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        correctness_reward,
        format_reward,
        thinking_reward,
    ],
    args=training_args,
    train_dataset=dataset,
)

print("Starting GRPO training...")
trainer.train()

elapsed = (time.time() - start_time) / 60
print(f"\nGRPO training complete in {elapsed:.1f} min")

# ============================================================
# 8. SAVE & ZIP
# ============================================================
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

config_path = os.path.join(ADAPTER_DIR, "adapter_config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["inference_mode"] = True
    cfg["lora_dropout"] = 0.0
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in ["adapter_config.json", "adapter_model.safetensors"]:
        fpath = os.path.join(ADAPTER_DIR, fname)
        if os.path.exists(fpath):
            zf.write(fpath, fname)
            print(f"  Added {fname}")

print(f"\nsubmission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
print(f"Total: {elapsed:.1f} min")
print("DONE")
