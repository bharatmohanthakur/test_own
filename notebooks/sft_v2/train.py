"""
Nemotron Reasoning Challenge — SFT v2
- No internet on RTX Pro 6000 — zero pip installs
- Force PyTorch native conv1d (GPU) instead of custom causal_conv1d CUDA kernel
- wandb offline mode
"""

import sys
import os

# CRITICAL: Block causal_conv1d CUDA kernel BEFORE any imports
# This forces mamba to use PyTorch native ops (still GPU, just not custom kernel)
# The custom kernel isn't compiled for Blackwell sm120
import types
fake_module = types.ModuleType('causal_conv1d')
fake_module.causal_conv1d_fn = None
fake_module.causal_conv1d_update = None
sys.modules['causal_conv1d'] = fake_module
sys.modules['causal_conv1d.causal_conv1d_cuda'] = None
print("Blocked causal_conv1d CUDA kernel — using PyTorch native ops (GPU)")

import csv
import glob
import json
import random
import re
import statistics
import subprocess

# wandb offline (no internet on RTX Pro 6000)
WANDB_ACTIVE = False
try:
    import wandb
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(
        project="nemotron-reasoning",
        name="sft-v2-cot-expanded-lora",
        config={
            "lora_rank": 32, "lora_alpha": 64, "lr": 2e-4, "epochs": 2,
            "max_seq_len": 4096, "batch_size_effective": 16,
            "target_modules": "q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|up_proj|down_proj|gate_proj",
            "data": "9500 real x2 reasoning + ~4750 non-reasoning",
            "model": "nemotron-3-nano-30b", "baseline_score": 0.48,
        },
        tags=["sft", "cot", "expanded-lora"],
    )
    WANDB_ACTIVE = True
    print("wandb initialized (offline — sync after download)")
except Exception as e:
    print(f"wandb: {e}")

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
LORA_ALPHA = 64
MAX_SEQ_LEN = 4096
TRAIN_EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 2e-4

# === Find training data ===
print("Loading training data...")
possible_paths = (
    glob.glob("/kaggle/input/*/train.csv")
    + glob.glob("/kaggle/input/*/*/train.csv")
    + glob.glob("/kaggle/input/*/*/*/train.csv")
)
print(f"Found: {possible_paths}")
data_path = possible_paths[0]
print(f"Using: {data_path}")


# === CoT generators ===
def classify_puzzle(prompt):
    p = prompt[:120].lower()
    if 'bit manipulation' in p: return 'bit_manipulation'
    if 'encryption' in p or 'decrypt' in p: return 'cipher'
    if 'gravitational' in p: return 'gravity'
    if 'unit conversion' in p: return 'unit_conversion'
    if 'numeral' in p: return 'numeral_system'
    if 'transformation rules' in p: return 'equation_transform'
    return 'unknown'

def gravity_cot(prompt, answer):
    pairs = re.findall(r't\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)', prompt)
    target = re.search(r'for\s+t\s*=\s*([\d.]+)s?\s+given', prompt)
    cot = "<think>\nI need to find the secret gravitational constant g using d = 0.5*g*t^2.\n\n"
    gs = []
    for t_str, d_str in pairs:
        t, d = float(t_str), float(d_str)
        g = 2 * d / (t * t)
        gs.append(g)
        cot += f"From t={t_str}, d={d_str}: g = 2*{d_str}/{t_str}^2 = {g:.4f}\n"
    g_avg = statistics.mean(gs) if gs else 9.8
    cot += f"\nAverage g = {g_avg:.4f}\n"
    if target:
        t_val = float(target.group(1))
        cot += f"\nFor t={t_val}: d = 0.5 * {g_avg:.4f} * {t_val}^2 = {0.5*g_avg*t_val*t_val:.2f}\n"
    cot += f"</think>\n\n\\boxed{{{answer}}}"
    return cot

def unit_cot(prompt, answer):
    pairs = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    target = re.search(r'convert the following measurement:\s*([\d.]+)', prompt)
    cot = "<think>\nI need to find the secret conversion factor.\n\n"
    factors = []
    for i_str, o_str in pairs:
        f = float(o_str) / float(i_str)
        factors.append(f)
        cot += f"{i_str}m → {o_str}: factor = {f:.6f}\n"
    f_avg = statistics.mean(factors) if factors else 1.0
    cot += f"\nAverage factor = {f_avg:.6f}\n"
    if target:
        t = float(target.group(1))
        cot += f"\n{t}m * {f_avg:.6f} = {f_avg*t:.2f}\n"
    cot += f"</think>\n\n\\boxed{{{answer}}}"
    return cot

def numeral_cot(prompt, answer):
    examples = re.findall(r'(\d+)\s*->\s*([A-Z]+)', prompt)
    target = re.search(r'write the number\s+(\d+)', prompt)
    cot = "<think>\nChecking the numeral system from examples:\n"
    for n, r in examples:
        cot += f"  {n} → {r}\n"
    cot += "\nThis is Roman numeral conversion.\nI=1, V=5, X=10, L=50, C=100, D=500, M=1000\n"
    cot += "Subtractive: IV=4, IX=9, XL=40, XC=90\n"
    if target:
        num = int(target.group(1))
        vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
                (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
        parts = []
        rem = num
        for val, sym in vals:
            while rem >= val:
                parts.append(sym)
                rem -= val
        cot += f"\n{num} = {''.join(parts)}\n"
    cot += f"</think>\n\n\\boxed{{{answer}}}"
    return cot

def cipher_cot(prompt, answer):
    cot = "<think>\nThis is a substitution cipher. Building letter mapping from examples:\n\n"
    mapping = {}
    for line in prompt.split('\n'):
        if '->' in line:
            parts = line.split('->')
            if len(parts) == 2:
                enc_words = parts[0].strip().split()
                dec_words = parts[1].strip().split()
                if len(enc_words) == len(dec_words):
                    for ew, dw in zip(enc_words, dec_words):
                        if len(ew) == len(dw):
                            for ec, dc in zip(ew, dw):
                                if ec not in mapping:
                                    mapping[ec] = dc
    for k in sorted(mapping.keys()):
        cot += f"  '{k}' → '{mapping[k]}'\n"
    target_match = re.search(r'decrypt the following text:\s*(.*)', prompt, re.IGNORECASE)
    if target_match:
        cipher_text = target_match.group(1).strip()
        cot += f"\nApplying to: {cipher_text}\n"
        result = ''.join(mapping.get(c, c) if c != ' ' else ' ' for c in cipher_text)
        cot += f"Result: {result}\n"
    cot += f"</think>\n\n\\boxed{{{answer}}}"
    return cot

def bit_cot(prompt, answer):
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    target_match = re.search(r'determine the output for:\s*([01]{8})', prompt)
    cot = "<think>\nAnalyzing bit manipulation pattern:\n\n"
    for inp, out in pairs[:4]:
        cot += f"  {inp} → {out}\n"
    examples = [(int(i, 2), int(o, 2)) for i, o in pairs]
    found = None
    if len(examples) >= 2:
        xor_val = examples[0][0] ^ examples[0][1]
        if all(o == (i ^ xor_val) for i, o in examples):
            found = f"XOR with {format(xor_val, '08b')}"
    if not found and all(o == (~i & 0xFF) for i, o in examples):
        found = "bitwise NOT"
    def bit_rev(n): return int(format(n, '08b')[::-1], 2)
    if not found and all(o == bit_rev(i) for i, o in examples):
        found = "bit reverse"
    if not found:
        for k in range(1, 8):
            if all(o == ((i << k | i >> (8-k)) & 0xFF) for i, o in examples):
                found = f"rotate left by {k}"
                break
    if found:
        cot += f"\nIdentified operation: {found}\n"
    else:
        cot += "\nTesting compound operations...\n"
    if target_match:
        cot += f"\nApplying to {target_match.group(1)} → {answer}\n"
    cot += f"</think>\n\n\\boxed{{{answer}}}"
    return cot

def equation_cot(prompt, answer):
    cot = "<think>\nThis is a symbol transformation puzzle. Finding character mapping:\n\n"
    mapping = {}
    for line in prompt.split('\n'):
        if '=' in line and 'determine' not in line.lower() and 'below' not in line.lower():
            parts = line.split('=')
            if len(parts) == 2:
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                for i in range(min(len(lhs), len(rhs))):
                    if lhs[i] not in mapping:
                        mapping[lhs[i]] = rhs[i]
    for k, v in list(mapping.items())[:12]:
        cot += f"  '{k}' → '{v}'\n"
    cot += f"\nApplying mapping to get: {answer}\n"
    cot += f"</think>\n\n\\boxed{{{answer}}}"
    return cot

COT_GENERATORS = {
    'gravity': gravity_cot, 'unit_conversion': unit_cot,
    'numeral_system': numeral_cot, 'cipher': cipher_cot,
    'bit_manipulation': bit_cot, 'equation_transform': equation_cot,
}

# === Build training data ===
train_examples = []
with open(data_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        prompt, answer = row[1], row[2]
        cat = classify_puzzle(prompt)
        user_msg = prompt + '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
        try:
            gen = COT_GENERATORS.get(cat)
            assistant_msg = gen(prompt, answer) if gen else f"<think>\nAnalyzing the pattern.\n</think>\n\n\\boxed{{{answer}}}"
        except Exception:
            assistant_msg = f"<think>\nAnalyzing the pattern.\n</think>\n\n\\boxed{{{answer}}}"
        train_examples.append({"messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]})

# 25% non-reasoning
non_reasoning = []
with open(data_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        prompt, answer = row[1], row[2]
        cat = classify_puzzle(prompt)
        if cat in ('numeral_system', 'unit_conversion', 'gravity'):
            user_msg = prompt + '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
            non_reasoning.append({"messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
            ]})

all_data = train_examples * 2 + non_reasoning
random.seed(42)
random.shuffle(all_data)
print(f"Total training examples: {len(all_data)} ({len(train_examples)*2} reasoning + {len(non_reasoning)} non-reasoning)")

# === Load model ===
print("Loading Nemotron-3-Nano-30B...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True,
    dtype=torch.bfloat16, offload_folder="/kaggle/tmp/offload",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Model loaded.")

# === Apply LoRA ===
print(f"Applying LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})...")
lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA,
    target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|up_proj|down_proj|gate_proj)$",
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.enable_input_require_grads()

# === Tokenize ===
print("Tokenizing...")
def tokenize_chat(example):
    try:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False,
            add_generation_prompt=False, enable_thinking=True,
        )
    except Exception:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )
    tokens = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = Dataset.from_list(all_data)
tokenized = dataset.map(tokenize_chat, remove_columns=["messages"], num_proc=2)
print(f"Tokenized {len(tokenized)} examples")

# === Train ===
print("Training...")
training_args = TrainingArguments(
    output_dir="/kaggle/tmp/checkpoints",
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR, bf16=True, logging_steps=10,
    save_strategy="no", warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="wandb" if WANDB_ACTIVE else "none",
    run_name="sft-v2-cot-expanded-lora",
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    max_grad_norm=1.0, weight_decay=0.01,
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
train_result = trainer.train()
print(f"Training complete. Loss: {train_result.training_loss:.4f}")

if WANDB_ACTIVE:
    wandb.log({"final_train_loss": train_result.training_loss, "total_steps": train_result.global_step})

# === Save & zip ===
print("Saving adapter...")
model.save_pretrained(OUTPUT_DIR)
os.chdir(OUTPUT_DIR)
subprocess.run("zip -m submission.zip *", shell=True, check=True)
print("Done — submission.zip created.")

if WANDB_ACTIVE:
    wandb.finish()
