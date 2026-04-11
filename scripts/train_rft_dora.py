"""
Phase 1: RFT (Rejection Fine-Tuning) with DoRA
- Uses rejection-sampled data from MiniMax M2.7
- DoRA instead of LoRA (+4% on RLVR benchmarks)
- Skips numeral_system (already 100%)
- v7 proven config: 4 modules, alpha=64, 2048 seq
"""

import os, sys, json, zipfile, subprocess
import torch

os.environ["WANDB_API_KEY"] = "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"

from unsloth import FastLanguageModel
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"])

from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset as HFDataset

# Config
MAX_SEQ_LEN = 2048
NUM_EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 4

# Load RFT data
DATA_PATH = "/workspace/rft_1000.jsonl"
train_data = []
with open(DATA_PATH) as f:
    for line in f:
        train_data.append(json.loads(line))
print(f"Loaded {len(train_data)} RFT examples")

# Load model with Unsloth
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Nemotron-3-Nano-30B-A3B",
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    unsloth_force_compile=False,
    attn_implementation="eager",
)
print("Model loaded")

# DoRA instead of LoRA (+4% on RLVR benchmarks)
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
    use_rslora=False,
    use_dora=True,  # DoRA: decomposed LoRA, +4% on RLVR
)
model.print_trainable_parameters()

# Patch fast path for Mamba generation
for module in model.modules():
    if hasattr(module, 'is_fast_path_available'):
        module.is_fast_path_available = False
print("Patched is_fast_path_available = False")

# Format data as conversations
def format_data(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

# Convert messages to conversations format
conversations = []
for item in train_data:
    conversations.append(item["messages"])

dataset = HFDataset.from_dict({"conversations": conversations})
dataset = dataset.map(format_data, batched=True)
print(f"Dataset: {len(dataset)} examples")

# SFT Trainer with response-only training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=10,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="wandb",
        run_name="rft-dora-phase1",
        bf16=True,
        max_seq_length=MAX_SEQ_LEN,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    ),
)

# Train only on assistant responses (not user prompts)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

wandb.init(project="nemotron-reasoning", name="rft-dora-phase1",
    config={"method": "RFT+DoRA", "data": "rft_1000_rejection_sampled",
            "lr": LR, "epochs": NUM_EPOCHS, "dora": True,
            "targets": "all_9_modules", "alpha": 64, "rank": 32},
    tags=["rft", "dora", "phase1"])

print(f"Training: {NUM_EPOCHS} epochs, {len(train_data)} RFT examples, DoRA")
trainer.train()
print("Training complete!")

# Save
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
