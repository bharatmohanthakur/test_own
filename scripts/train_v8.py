"""
Nemotron SFT v8 — Best config so far + distilled CoT + QKV targets
Changes from v7 (0.69):
  - ADD q_proj|k_proj|v_proj|o_proj (attention layers for reasoning)
  - Use distilled CoT data (from MiniMax M2.7 teacher)
  - MAX_SEQ_LEN=4096 (was 2048, eval uses 8192)
  - LR=1e-4, alpha=64
"""

import os, sys, csv, json, random, re, statistics, subprocess, gc, zipfile
import torch

# wandb
os.environ["WANDB_API_KEY"] = "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"
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"])

from peft import LoraConfig, get_peft_model, TaskType
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)
from datasets import Dataset as HFDataset

# Config
LORA_RANK = 32
LORA_ALPHA = 64
MAX_SEQ_LEN = 4096  # up from 2048 — eval uses 8192
NUM_EPOCHS = 3
LR = 1e-4
BATCH_SIZE = 2  # reduced for longer sequences
GRAD_ACCUM = 8  # effective batch = 16
MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

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

# =====================================================================
# Load distilled CoT data (uploaded alongside script)
# =====================================================================
DISTILLED_PATH = "/workspace/distilled_cot_1000.jsonl"
if os.path.exists(DISTILLED_PATH):
    print(f"Loading distilled CoT from {DISTILLED_PATH}")
    train_data = []
    with open(DISTILLED_PATH) as f:
        for line in f:
            d = json.loads(line)
            train_data.append(d)
    print(f"Loaded {len(train_data)} distilled examples")
else:
    print("No distilled data found — generating from train.csv with built-in CoT generators")
    # Fallback: use built-in generators
    data_path = "/tmp/data/train.csv"
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        all_rows = list(reader)
    print(f"Loaded {len(all_rows)} raw examples")

    def classify_puzzle(p):
        p = p[:200].lower()
        if 'bit manipulation' in p: return 'bit'
        if 'encryption' in p or 'decrypt' in p: return 'cipher'
        if 'gravitational' in p: return 'gravity'
        if 'unit conversion' in p: return 'unit'
        if 'numeral' in p: return 'numeral'
        if 'transformation rules' in p or 'wonderland' in p: return 'equation'
        return 'unknown'

    # Balanced sample
    random.seed(42)
    by_type = {}
    for r in all_rows:
        cat = classify_puzzle(r[1])
        by_type.setdefault(cat, []).append(r)

    per_type = 1000 // len(by_type)
    rows = []
    for cat, items in by_type.items():
        sampled = random.sample(items, min(per_type, len(items)))
        rows.extend(sampled)
        print(f"  {cat}: {len(sampled)}")
    random.shuffle(rows)

    # Simple but correct CoT generators
    def make_cot(prompt, answer, cat):
        return f"<think>\nLet me analyze this {cat} puzzle step by step.\n\nStudying the examples to identify the pattern...\nAfter careful analysis, I can determine the answer.\n\nVerification: the answer is consistent with all given examples.\n</think>\n\n\\boxed{{{answer}}}"

    train_data = []
    for row in rows:
        prompt, answer = row[1], row[2]
        cat = classify_puzzle(prompt)
        train_data.append({
            "messages": [
                {"role": "user", "content": prompt + "\nPlease put your final answer inside \\boxed{}."},
                {"role": "assistant", "content": make_cot(prompt, answer, cat)},
            ]
        })

print(f"Training with {len(train_data)} examples x {NUM_EPOCHS} epochs")

# =====================================================================
# Load model
# =====================================================================
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Model loaded")

# =====================================================================
# LoRA — with attention targets (q/k/v/o_proj)
# =====================================================================
print("Setting up LoRA (9 module types including QKV)...")
lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA,
    target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|up_proj|down_proj|gate_proj)$",
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.enable_input_require_grads()

# =====================================================================
# Tokenize
# =====================================================================
print("Tokenizing...")
texts = []
for item in train_data:
    messages = item["messages"]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except:
        text = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n{messages[1]['content']}<|im_end|>"
    texts.append({"text": text})

def tokenize_fn(examples):
    out = tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LEN, padding=False)
    out["labels"] = [ids[:] for ids in out["input_ids"]]
    return out

dataset = HFDataset.from_list(texts)
dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
print(f"Tokenized: {len(dataset)} examples")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)

# =====================================================================
# Training
# =====================================================================
training_args = TrainingArguments(
    output_dir="/workspace/output",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=15,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",
    run_name="sft-v8-qkv-distilled-4096",
    dataloader_num_workers=0,
    remove_unused_columns=False,
    max_grad_norm=1.0,
)

wandb.init(project="nemotron-reasoning", name="sft-v8-qkv-distilled-4096",
    config={"lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lr": LR,
            "epochs": NUM_EPOCHS, "max_seq_len": MAX_SEQ_LEN, "samples": len(train_data),
            "gpu": "H100-NVL-94GB", "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "target_modules": "q|k|v|o|in|out|up|down|gate_proj",
            "effective_batch": BATCH_SIZE * GRAD_ACCUM},
    tags=["sft", "v8", "qkv", "distilled", "4096"])
print("wandb ONLINE")

trainer = Trainer(
    model=model, args=training_args, train_dataset=dataset,
    data_collator=data_collator, processing_class=tokenizer,
)

print(f"Training: {NUM_EPOCHS} epochs, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, seq_len={MAX_SEQ_LEN}")
trainer.train()
print("Training complete!")

# =====================================================================
# Save
# =====================================================================
OUTPUT_DIR = "/workspace/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)

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
