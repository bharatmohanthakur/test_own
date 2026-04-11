"""
GRPO on v9 SFT (0.68) using Unsloth — following official Nemotron notebook.
Exact deps: torch==2.7.1, transformers==4.56.2, trl==0.22.2, mamba_ssm==2.2.5
"""

import os, sys, csv, json, random, re, subprocess, zipfile
import torch

os.environ["WANDB_API_KEY"] = "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"
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"])

from unsloth import FastLanguageModel
from peft import PeftModel

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

# Sample 300 balanced
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

NUM_SAMPLES = 300
per_type = NUM_SAMPLES // len(by_type)
rows = []
for cat, items in by_type.items():
    sampled = random.sample(items, min(per_type, len(items)))
    rows.extend(sampled)
    print(f"  {cat}: {len(sampled)}")
random.shuffle(rows)

# Build dataset with prompts + ground truth
from datasets import Dataset as HFDataset

prompts_list = []
gt_list = []
for row in rows:
    user_msg = row[1] + "\nPlease put your final answer inside \\boxed{}."
    prompts_list.append([{"role": "user", "content": user_msg}])
    gt_list.append(row[2])

dataset = HFDataset.from_dict({"prompt": prompts_list, "ground_truth": gt_list})
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
        if isinstance(completion, list):
            text = completion[0]["content"] if completion else ""
        else:
            text = str(completion)
        matches = re.findall(r'\\boxed\{([^}]*)\}', text)
        correct = False
        if matches:
            pred = normalize_answer(matches[-1])
            exp = normalize_answer(gt)
            if pred == exp:
                correct = True
            else:
                try:
                    if abs(float(pred) - float(exp)) < 0.02:
                        correct = True
                except:
                    pass
        rewards.append(1.0 if correct else 0.0)
    return rewards

# Load model with Unsloth — following official notebook
print("Loading model with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Nemotron-3-Nano-30B-A3B",
    max_seq_length=2048,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    unsloth_force_compile=False,
    attn_implementation="eager",
)
print("Model loaded")

# Load v9 SFT adapter
ADAPTER_PATH = "/workspace/adapter"
if os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
    print("Loading v9 SFT adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=True)
    print("v9 adapter loaded as trainable")
else:
    print("No adapter found — applying fresh LoRA")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj",
                         "in_proj", "out_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

# Patch fast path — causes mat shape errors during generation
for module in model.modules():
    if hasattr(module, 'is_fast_path_available'):
        module.is_fast_path_available = False
print("Patched is_fast_path_available = False")

model.print_trainable_parameters()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# GRPO Training
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir="/workspace/grpo_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    max_steps=150,
    warmup_steps=15,
    lr_scheduler_type="cosine",
    logging_steps=1,
    save_strategy="steps",
    save_steps=75,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",
    run_name="grpo-v2-unsloth-on-v9",
    max_completion_length=1024,
    max_prompt_length=1024,
    num_generations=2,
    use_vllm=False,
    remove_unused_columns=False,
    temperature=0.7,
    seed=3407,
)

wandb.init(project="nemotron-reasoning", name="grpo-v2-unsloth-on-v9",
    config={"method": "GRPO", "base": "v9-0.68", "lr": 5e-6,
            "max_steps": 150, "num_generations": 4,
            "samples": NUM_SAMPLES, "framework": "unsloth"},
    tags=["grpo", "unsloth", "on-v9"])

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_fn,
    processing_class=tokenizer,
)

print(f"Starting GRPO: 150 steps, {NUM_SAMPLES} samples, 4 generations/sample")
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
