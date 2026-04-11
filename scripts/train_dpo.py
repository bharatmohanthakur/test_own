"""
DPO on top of SFT v7 (0.69 score).
Step 1: Load model + SFT adapter, generate predictions on training prompts
Step 2: Create preference pairs (correct CoT = chosen, model's wrong output = rejected)
Step 3: Train DPO with TRL
"""

import os, sys, csv, json, random, re, subprocess, zipfile, gc
import torch

# wandb
os.environ["WANDB_API_KEY"] = "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"
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"])

from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import Dataset as HFDataset

# Config
MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
ADAPTER_PATH = "/workspace/sft_adapter"
NUM_SAMPLES = 200
MAX_NEW_TOKENS = 1024
DPO_EPOCHS = 2
DPO_LR = 5e-7
DPO_BETA = 0.1
BATCH_SIZE = 1
GRAD_ACCUM = 4

# =====================================================================
# Download competition data
# =====================================================================
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

# Load perfect CoT data for "chosen" responses
PERFECT_COT_PATH = "/workspace/perfect_cot_1000.jsonl"
perfect_cot = {}
if os.path.exists(PERFECT_COT_PATH):
    with open(PERFECT_COT_PATH) as f:
        for line in f:
            d = json.loads(line)
            key = d["messages"][0]["content"][:100]
            perfect_cot[key] = d["messages"][1]["content"]
    print(f"Loaded {len(perfect_cot)} perfect CoT examples")

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
print(f"Using {len(rows)} examples for DPO")

# =====================================================================
# Load model + SFT adapter
# =====================================================================
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("Base model loaded")

if os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
    print(f"Loading SFT adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload()
    print("SFT adapter merged into base model")

# =====================================================================
# Step 1: Generate model predictions for "rejected" responses
# =====================================================================
print("\nGenerating model predictions for DPO rejected pairs...")
model.eval()

def normalize_answer(s):
    s = str(s).strip()
    try:
        return str(round(float(s), 2))
    except ValueError:
        return s.lower().strip()

dpo_data = []
correct_count = 0
wrong_count = 0

for idx, row in enumerate(rows):
    prompt_text = row[1]
    ground_truth = row[2]
    user_msg = prompt_text + "\nPlease put your final answer inside \\boxed{}."

    messages = [{"role": "user", "content": user_msg}]
    try:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        input_text = "<|im_start|>user\n" + user_msg + "<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )

    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    predicted_matches = re.findall(r'\\boxed\{([^}]*)\}', generated)
    predicted = predicted_matches[-1].strip() if predicted_matches else ""

    is_correct = normalize_answer(predicted) == normalize_answer(ground_truth)

    if is_correct:
        correct_count += 1
    else:
        wrong_count += 1
        key = user_msg[:100]
        if key in perfect_cot:
            chosen_response = perfect_cot[key]
        else:
            chosen_response = "<think>\nLet me solve this step by step.\nAfter careful analysis of the pattern from the examples, I determine the answer.\n</think>\n\n\\boxed{" + ground_truth + "}"

        dpo_data.append({
            "prompt": [{"role": "user", "content": user_msg}],
            "chosen": [{"role": "assistant", "content": chosen_response}],
            "rejected": [{"role": "assistant", "content": generated}],
        })

    if (idx + 1) % 20 == 0:
        print(f"  [{idx+1}/{len(rows)}] correct={correct_count}, wrong={wrong_count}")

print(f"\nGeneration complete: {correct_count} correct, {wrong_count} wrong")
print(f"DPO pairs created: {len(dpo_data)} (from wrong predictions)")

if len(dpo_data) < 20:
    print("Few wrong predictions — augmenting with quality-preference pairs...")
    for row in rows[:60]:
        user_msg = row[1] + "\nPlease put your final answer inside \\boxed{}."
        key = user_msg[:100]
        if key in perfect_cot:
            dpo_data.append({
                "prompt": [{"role": "user", "content": user_msg}],
                "chosen": [{"role": "assistant", "content": perfect_cot[key]}],
                "rejected": [{"role": "assistant", "content": "<think>\nThe answer is.\n</think>\n\n\\boxed{" + row[2] + "}"}],
            })
    print(f"Augmented to {len(dpo_data)} pairs")

with open("/workspace/dpo_pairs.jsonl", "w") as f:
    for item in dpo_data:
        f.write(json.dumps(item) + "\n")
print(f"Saved {len(dpo_data)} DPO pairs")

# =====================================================================
# Step 2: Free memory and reload for DPO training
# =====================================================================
del model, outputs, inputs
gc.collect()
torch.cuda.empty_cache()
print("\nMemory cleared. Reloading for DPO training...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
)
if os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload()
    print("SFT adapter merged for DPO")

tokenizer.padding_side = "left"

# =====================================================================
# Step 3: DPO Training
# =====================================================================
dataset = HFDataset.from_list(dpo_data)
print(f"DPO dataset: {len(dataset)} preference pairs")

lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)

training_args = DPOConfig(
    output_dir="/workspace/dpo_output",
    num_train_epochs=DPO_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=DPO_LR,
    beta=DPO_BETA,
    loss_type="sigmoid",
    warmup_steps=5,
    lr_scheduler_type="cosine",
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",
    run_name="dpo-v1-on-sft-v7",
    remove_unused_columns=False,
    max_length=4096,
    max_prompt_length=2048,
)

wandb.init(project="nemotron-reasoning", name="dpo-v1-on-sft-v7",
    config={"method": "DPO", "base": "sft-v7-0.69", "lr": DPO_LR,
            "beta": DPO_BETA, "epochs": DPO_EPOCHS, "pairs": len(dpo_data)},
    tags=["dpo", "v1", "on-sft-v7"])
print("wandb ONLINE")

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

print(f"Starting DPO: {DPO_EPOCHS} epochs, {len(dpo_data)} pairs")
trainer.train()
print("DPO training complete!")

# =====================================================================
# Save
# =====================================================================
OUTPUT_DIR = "/workspace/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)

zip_path = "/workspace/submission.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            zf.write(fpath, fname)
size_mb = os.path.getsize(zip_path) / 1024 / 1024
print(f"submission.zip: {size_mb:.1f} MB")
wandb.finish()
print("DONE")
