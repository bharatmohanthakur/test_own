"""
DPO on v9 adapter — continues training the SAME LoRA (single adapter output).
1. Load base model + v9 LoRA (as trainable PEFT, not merged)
2. Train DPO on preference pairs
3. Save — output is the same single LoRA, now with DPO refinement baked in
"""

import os, sys, json, zipfile, gc
import torch

os.environ["WANDB_API_KEY"] = "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"
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"])

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import Dataset as HFDataset

MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
ADAPTER_PATH = "/workspace/adapter"  # v9 adapter

# Load DPO pairs (120 pairs from vLLM generation)
with open("/workspace/dpo_pairs.jsonl") as f:
    dpo_data = [json.loads(line) for line in f]
print(f"Loaded {len(dpo_data)} DPO pairs")

# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("Base model loaded")

# Load v9 LoRA as TRAINABLE (not merged)
print(f"Loading v9 adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=True)
model.print_trainable_parameters()
print("v9 adapter loaded as trainable")

# DPO dataset
dataset = HFDataset.from_list(dpo_data)
print(f"DPO dataset: {len(dataset)} pairs")

training_args = DPOConfig(
    output_dir="/workspace/dpo_output",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    beta=0.1,
    loss_type="sigmoid",
    warmup_steps=5,
    lr_scheduler_type="cosine",
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",
    run_name="dpo-on-v9-single-lora",
    remove_unused_columns=False,
    max_length=4096,
    max_prompt_length=2048,
)

wandb.init(project="nemotron-reasoning", name="dpo-on-v9-single-lora",
    config={"method": "DPO", "base_adapter": "v9-0.68", "lr": 5e-7,
            "beta": 0.1, "epochs": 2, "pairs": len(dpo_data)},
    tags=["dpo", "on-v9", "single-lora"])

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print(f"Starting DPO: 2 epochs, {len(dpo_data)} pairs, continuing v9 LoRA")
trainer.train()
print("DPO complete!")

# Save — this is the v9 LoRA with DPO refinement
OUTPUT_DIR = "/workspace/dpo_adapter"
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
