"""
SFT v34 — RunPod adaptation (BINARY REBUILD)

Tracking-informed data iteration:
- v22 cipher/gravity/unit/roman are saturated (kept verbatim, 419 examples)
- v22 binary teaches "I identify the pattern" hallucination (0/10 on v31)
- v34 replaces v22 binary with 150 verifier-derived per-bit derivation traces
- 0 Donald labels, 0 vocab_fill (the v30 failure pattern)
- Total: 569 examples

Same proven v22 config: r=32, alpha=64, LR=2e-4, 3 epochs, 4 LoRA targets.
Unsloth SFTTrainer (much faster than GRPO since no rollouts).

Layout on pod:
  /workspace/
    train_v34.py            (this file)
    data/training_v34.jsonl (kaggle CLI download from bharatmohan/nemotron-training-v34)
    model/                  (Nemotron-3-Nano-30B via HF snapshot_download)
    output_v34/
      nemotron-lora-adapter/  (final saved adapter)
      submission.zip
"""

import os, sys, subprocess, glob, json, random, time, zipfile, gc, re, math

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# wandb
os.environ["WANDB_API_KEY"] = "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"
os.environ["WANDB_PROJECT"] = "nemotron-reasoning"
os.environ["WANDB_NAME"] = "sft-v34-runpod"

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset as HFDataset

print(f"Torch {torch.__version__}, CUDA {torch.version.cuda}")

# Patch Mamba fast path for the slow fallback (needed for Blackwell)
# (For SFT the model load path uses the same mamba_ssm imports as GRPO)
import sys as _sys
for name, mod in list(_sys.modules.items()):
    if "modeling_nemotron_h" in name and hasattr(mod, "is_fast_path_available"):
        mod.is_fast_path_available = False

# ============================================================
# CONFIG
# ============================================================
WORKSPACE = "/workspace"
MODEL_PATH = os.path.join(WORKSPACE, "model")
DATA_PATH = os.path.join(WORKSPACE, "data", "training_v34.jsonl")
OUTPUT_DIR = os.path.join(WORKSPACE, "output_v34")
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "nemotron-lora-adapter")
os.makedirs(ADAPTER_DIR, exist_ok=True)

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
MAX_SEQ_LEN = 4096
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 2e-4
RANDOM_SEED = 42

# ============================================================
# LOAD DATA
# ============================================================
print(f"Loading training data from {DATA_PATH}...")
all_examples = []
with open(DATA_PATH) as f:
    for line in f:
        all_examples.append(json.loads(line))
print(f"Loaded {len(all_examples)} examples")

random.seed(RANDOM_SEED)
random.shuffle(all_examples)

# ============================================================
# LOAD MODEL WITH UNSLOTH
# ============================================================
print(f"Loading base model from {MODEL_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    dtype=None,
    unsloth_force_compile=False,
    attn_implementation="eager",
)

# Patch Mamba fast path on model modules
for module in model.modules():
    if hasattr(module, "is_fast_path_available"):
        module.is_fast_path_available = False

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Model loaded.")

# ============================================================
# APPLY LoRA
# ============================================================
target_modules = ["in_proj", "out_proj", "up_proj", "down_proj"]
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=target_modules,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=RANDOM_SEED,
)
model.print_trainable_parameters()

# ============================================================
# PREPARE DATASET
# ============================================================
records = [{"messages": item["messages"]} for item in all_examples]
dataset = HFDataset.from_list(records)

dataset = dataset.map(lambda ex: {
    "text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False,
        add_generation_prompt=False, enable_thinking=True,
    )
})
print(f"Dataset: {len(dataset)} examples")

# ============================================================
# TRAIN
# ============================================================
start_time = time.time()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        optim="adamw_8bit",
        seed=RANDOM_SEED,
        report_to="wandb",
        dataloader_num_workers=2,
        weight_decay=0.01,
        max_grad_norm=1.0,
    ),
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    dataset_kwargs={"skip_prepare_dataset": False},
)

# Mask prompt tokens — only compute loss on assistant response
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

print(f"\n{'='*60}")
print(f"SFT v34 (RunPod) — Mixed v22 verbose + v29 templated, {NUM_EPOCHS} epochs")
print(f"{'='*60}")
print(f"  Examples: {len(dataset)}")
print(f"  Batch: {BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
print(f"  LR: {LR}, LoRA: r={LORA_RANK}, a={LORA_ALPHA}")
print(f"  Targets: {target_modules}")
print(f"{'='*60}\n")

print("Training...")
trainer.train()
elapsed = (time.time() - start_time) / 60
print(f"\nTraining complete in {elapsed:.1f} min")

# ============================================================
# SAVE & ZIP
# ============================================================
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

# Patch adapter_config for inference
config_path = os.path.join(ADAPTER_DIR, "adapter_config.json")
if os.path.exists(config_path):
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["inference_mode"] = True
        cfg["lora_dropout"] = 0.0
        # Write to a temp file then rename (avoid truncation on crash)
        tmp_path = config_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(cfg, f, indent=2)
        os.rename(tmp_path, config_path)
        print(f"  Patched adapter_config.json (inference_mode=True)")
    except Exception as e:
        print(f"  WARN: failed to patch adapter_config.json: {e}")

# Build submission.zip
zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in ["adapter_config.json", "adapter_model.safetensors"]:
        fpath = os.path.join(ADAPTER_DIR, fname)
        if os.path.exists(fpath):
            zf.write(fpath, fname)
            print(f"  Added {fname}")

if os.path.exists(zip_path):
    print(f"\nsubmission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
print(f"Total: {elapsed:.1f} min")
print("DONE")
