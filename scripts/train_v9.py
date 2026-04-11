"""
SFT v9 — Verified data + v7 config (proven best)
- 504 verified examples (all answers confirmed against train.csv)
- 4 LoRA targets (in_proj|out_proj|up_proj|down_proj) — same as v7
- alpha=64, rank=32, LR=2e-4, seq_len=2048
- 3 epochs
"""

import os, sys, json, zipfile, subprocess
import torch

os.environ["WANDB_API_KEY"] = "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"
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"])

from peft import LoraConfig, get_peft_model, TaskType
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)
from datasets import Dataset as HFDataset

# v7 config — proven best (0.69)
LORA_RANK = 32
LORA_ALPHA = 64
MAX_SEQ_LEN = 2048
NUM_EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 4
MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# Load verified data
DATA_PATH = "/workspace/verified_500.jsonl"
train_data = []
with open(DATA_PATH) as f:
    for line in f:
        train_data.append(json.loads(line))
print(f"Loaded {len(train_data)} verified examples")

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Model loaded")

# LoRA — v7 config (4 modules only)
print("Setting up LoRA (v7 config: 4 modules)...")
lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.enable_input_require_grads()

# Tokenize
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

training_args = TrainingArguments(
    output_dir="/workspace/output",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_strategy="no",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",
    run_name="sft-v9-verified-500",
    dataloader_num_workers=0,
    remove_unused_columns=False,
    max_grad_norm=1.0,
)

wandb.init(project="nemotron-reasoning", name="sft-v9-verified-500",
    config={"lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lr": LR,
            "epochs": NUM_EPOCHS, "max_seq_len": MAX_SEQ_LEN, "samples": len(train_data),
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "target_modules": "in_proj|out_proj|up_proj|down_proj",
            "data": "verified_500", "config_base": "v7"},
    tags=["sft", "v9", "verified", "v7-config"])
print("wandb ONLINE")

trainer = Trainer(
    model=model, args=training_args, train_dataset=dataset,
    data_collator=data_collator, processing_class=tokenizer,
)

print(f"Training: {NUM_EPOCHS} epochs, {len(train_data)} verified examples")
trainer.train()
print("Training complete!")

OUTPUT_DIR = "/workspace/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)

zip_path = "/workspace/submission.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            zf.write(fpath, fname)
print(f"submission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
wandb.finish()
print("DONE")
