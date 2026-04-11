"""
Nemotron Reasoning Challenge — SFT Training v2
Uses competition train data + broader reasoning format.
Trains with proper chat template and \boxed{} output format.
"""

import csv
import json
import os
import subprocess
import random

import kagglehub
import mamba_ssm
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

# === Config ===
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
OUTPUT_DIR = "/kaggle/working"
LORA_RANK = 32
MAX_SEQ_LEN = 1536  # keep shorter for memory
TRAIN_EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 1e-4

# === Load competition training data ===
print("Loading training data...")
import glob
possible_paths = (
    glob.glob("/kaggle/input/*/train.csv")
    + glob.glob("/kaggle/input/*/*/train.csv")
    + glob.glob("/kaggle/input/*/*/*/train.csv")
)
print(f"Found data files: {possible_paths}")
data_path = possible_paths[0] if possible_paths else "/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv"
print(f"Using: {data_path}")
train_examples = []
with open(data_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        prompt, answer = row[1], row[2]
        # Detailed chain-of-thought response with boxed answer
        user_msg = prompt + '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

        # Create a more detailed reasoning trace
        assistant_msg = (
            "<think>\n"
            "Let me carefully analyze the examples provided to identify the underlying pattern or rule.\n\n"
            "Looking at the input-output pairs, I need to find the transformation being applied.\n\n"
            "After examining all the examples systematically, I can identify the pattern and apply it to the given input.\n"
            "</think>\n\n"
            f"\\boxed{{{answer}}}"
        )

        train_examples.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        })

# Repeat real data 3x for emphasis
train_examples = train_examples * 3
random.seed(42)
random.shuffle(train_examples)
print(f"Total training examples: {len(train_examples)}")

# === Load model ===
print("Loading Nemotron-3-Nano-30B...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    offload_folder="/kaggle/tmp/offload",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Model loaded.")

# === Apply LoRA ===
print(f"Applying LoRA adapter (rank={LORA_RANK})...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=32,  # alpha = rank for stable training
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.enable_input_require_grads()

# === Tokenize data ===
print("Tokenizing data...")

def tokenize_chat(example):
    try:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    except Exception:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = Dataset.from_list(train_examples)
tokenized = dataset.map(tokenize_chat, remove_columns=["messages"], num_proc=2)
print(f"Tokenized {len(tokenized)} examples")

# === Train ===
print("Starting training...")
training_args = TrainingArguments(
    output_dir="/kaggle/tmp/checkpoints",
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    bf16=True,
    logging_steps=25,
    save_strategy="no",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

trainer.train()
print("Training complete.")

# === Save adapter ===
print(f"Saving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)

# === Package submission ===
os.chdir(OUTPUT_DIR)
subprocess.run("zip -m submission.zip *", shell=True, check=True)
print("Done — submission.zip created.")
