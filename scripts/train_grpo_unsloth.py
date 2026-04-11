"""
GRPO on v9 SFT adapter (0.68) using Unsloth for 80% VRAM savings.
- Load base model + v9 adapter via Unsloth
- GRPO with competition puzzles as verifiable rewards
- Binary reward: correct \boxed{} = 1.0, wrong = 0.0
"""

import os, sys, csv, json, random, re, subprocess, zipfile
import torch

os.environ["WANDB_API_KEY"] = "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"

# Try unsloth, fallback to standard
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "unsloth", "-q"], check=False)
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
    print("Unsloth available")
except Exception as e:
    print(f"Unsloth not available: {e}")
    USE_UNSLOTH = False

import wandb
wandb.login(key=os.environ["WANDB_API_KEY"])

from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer

# Config
MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
ADAPTER_PATH = "/workspace/adapter"  # v9 adapter
NUM_SAMPLES = 300
MAX_SEQ_LEN = 2048

# Download competition data
print("Setting up data...")
subprocess.run([sys.executable, "-m", "pip", "install", "kaggle", "-q"], check=False)
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "w") as f:
    json.dump({"username": "bharatmohan", "key": "1f8a45c8e928ba8ef9929855d233a32f"}, f)
os.chmod("/root/.kaggle/kaggle.json", 0o600)
subprocess.run(["kaggle", "competitions", "download",
                 "nvidia-nemotron-model-reasoning-challenge", "-p", "/tmp/data"], check=False)
subprocess.run(["unzip", "-o", "/tmp/data/nvidia-nemotron-model-reasoning-challenge.zip",
                 "-d", "/tmp/data"], check=False)

with open("/tmp/data/train.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)
    all_rows = list(reader)
print(f"Loaded {len(all_rows)} examples")

# Classify and sample
def classify_puzzle(p):
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
    cat = classify_puzzle(r[1])
    by_type.setdefault(cat, []).append(r)

per_type = NUM_SAMPLES // len(by_type)
rows = []
for cat, items in by_type.items():
    sampled = random.sample(items, min(per_type, len(items)))
    rows.extend(sampled)
    print(f"  {cat}: {len(sampled)}")
random.shuffle(rows)

# Build GRPO dataset — prompts + ground truth
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

prompts = []
ground_truths = []
for row in rows:
    user_msg = row[1] + "\nPlease put your final answer inside \\boxed{}."
    messages = [{"role": "user", "content": user_msg}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        text = "<|im_start|>user\n" + user_msg + "<|im_end|>\n<|im_start|>assistant\n"
    prompts.append(text)
    ground_truths.append(row[2])

dataset = HFDataset.from_dict({"prompt": prompts, "ground_truth": ground_truths})
print(f"GRPO dataset: {len(dataset)} prompts")

# Reward function
def normalize_answer(s):
    s = str(s).strip()
    try:
        return str(round(float(s), 2))
    except ValueError:
        return s.lower().strip()

def reward_fn(completions, ground_truth=None, **kwargs):
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        text = completion if isinstance(completion, str) else completion[0]["content"] if isinstance(completion, list) else str(completion)
        matches = re.findall(r'\\boxed\{([^}]*)\}', text)
        if matches and normalize_answer(matches[-1]) == normalize_answer(gt):
            rewards.append(1.0)
        else:
            # Try numeric tolerance
            try:
                pred = float(matches[-1]) if matches else float('nan')
                if abs(pred - float(gt)) < 0.02:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except:
                rewards.append(0.0)
    return rewards

# Load model
print("Loading model...")
if USE_UNSLOTH:
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=MAX_SEQ_LEN,
            load_in_4bit=True,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("Unsloth model loaded")
    except Exception as e:
        print(f"Unsloth load failed: {e}, falling back to HF")
        USE_UNSLOTH = False

if not USE_UNSLOTH:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("HF model loaded")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load v9 adapter
if os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
    print("Loading v9 SFT adapter...")
    if USE_UNSLOTH:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=True)
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=True)
    print("v9 adapter loaded as trainable")
else:
    print("WARNING: No adapter found!")

model.print_trainable_parameters()

# GRPO config
training_args = GRPOConfig(
    output_dir="/workspace/grpo_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    max_steps=200,
    warmup_steps=20,
    lr_scheduler_type="cosine",
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",
    run_name="grpo-v1-on-v9-0.68",
    max_completion_length=1024,
    max_prompt_length=1024,
    num_generations=4,
    remove_unused_columns=False,
    temperature=0.7,
)

wandb.init(project="nemotron-reasoning", name="grpo-v1-on-v9-0.68",
    config={"method": "GRPO", "base": "v9-0.68", "lr": 5e-6,
            "max_steps": 200, "num_generations": 4,
            "samples": NUM_SAMPLES, "unsloth": USE_UNSLOTH},
    tags=["grpo", "on-v9", "unsloth" if USE_UNSLOTH else "hf"])

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_fn,
    processing_class=tokenizer,
)

print(f"Starting GRPO: 200 steps, {NUM_SAMPLES} samples, 4 generations/sample")
trainer.train()
print("GRPO complete!")

# Save
OUTPUT_DIR = "/workspace/grpo_adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)

zip_path = "/workspace/submission.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            zf.write(fpath, fname)
print(f"submission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
wandb.finish()
print("DONE")
