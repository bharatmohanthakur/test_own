"""
Nemotron Reasoning Challenge — Baseline (Zero-shot LoRA)
Creates an untrained LoRA adapter to establish a baseline score.
"""

import os
import subprocess

import kagglehub
import mamba_ssm  # provided by nvidia-utility-script kernel source
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# === Config ===
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
OUTPUT_DIR = "/kaggle/working"
LORA_RANK = 32

# === Load model ===
print("Loading Nemotron-3-Nano-30B...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    offload_folder="/kaggle/tmp/offload",
)
print("Model loaded.")

# === Apply LoRA (untrained — zero-shot baseline) ===
print(f"Applying LoRA adapter (rank={LORA_RANK})...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=16,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Save adapter ===
print(f"Saving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)

# === Package submission ===
os.chdir(OUTPUT_DIR)
subprocess.run("zip -m submission.zip *", shell=True, check=True)
print("Done — submission.zip created.")
