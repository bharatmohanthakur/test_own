"""
SFT v21 — Unsloth 16-bit LoRA + weakness-weighted data
1550 examples weighted by base model failure rate:
  bit_manipulation 400, equation 350, cipher 300, gravity 150,
  unit_conversion 100, numeral 50, other 200
v7 hyperparams: alpha=64, LR=2e-4, 2 epochs
"""

import sys, os, subprocess, glob, json, random, time, zipfile, gc

# ============================================================
# 1. INSTALL FROM OFFLINE WHEELS (exact pattern from reference notebook)
# ============================================================
print("Installing packages from offline wheels...")

# Path to mayukh18/nemotron-packages
PKG_DIR = None
for candidate in [
    "/kaggle/input/datasets/mayukh18/nemotron-packages/packages",
    "/kaggle/input/nemotron-packages/packages",
    "/kaggle/input/mayukh18/nemotron-packages/packages",
]:
    if os.path.exists(candidate):
        PKG_DIR = candidate
        break

if PKG_DIR:
    print(f"Found packages at: {PKG_DIR}")
    # Install unsloth + all dependencies from offline wheels
    subprocess.run(
        f"pip install -q --no-index --find-links {PKG_DIR} "
        f"unsloth trl peft transformers datasets accelerate bitsandbytes",
        shell=True
    )
    # Install causal_conv1d and mamba_ssm specific wheels
    for pattern in ["causal_conv1d*.whl", "mamba_ssm*.whl"]:
        wheels = sorted(glob.glob(os.path.join(PKG_DIR, "..", pattern)) +
                       glob.glob(os.path.join(PKG_DIR, pattern)))
        if wheels:
            subprocess.run(f"pip install -q {wheels[-1]}", shell=True)
            print(f"Installed: {os.path.basename(wheels[-1])}")
else:
    print("WARNING: nemotron-packages not found! Listing available files:")
    for root, dirs, files in os.walk("/kaggle/input/"):
        depth = root.replace("/kaggle/input/", "").count(os.sep)
        if depth > 4:
            dirs.clear()
            continue
        for f in files:
            if f.endswith(('.whl', '.jsonl', '.csv')):
                print(f"  {os.path.join(root, f)}")
    # Fallback to dennisfong packages
    alt = "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages"
    if os.path.exists(alt):
        subprocess.run(f"pip install -q --no-index --find-links {alt} trl", shell=True)
    sys.exit(1)

# ============================================================
# 2. IMPORTS
# ============================================================
import torch
import kagglehub
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset as HFDataset

print(f"Unsloth loaded. torch={torch.__version__}, CUDA={torch.cuda.is_available()}")

# ============================================================
# 3. WANDB
# ============================================================
WANDB_ACTIVE = False
try:
    import wandb
    try:
        wandb.login(key="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", relogin=True)
        wandb.init(project="nemotron-reasoning", name="sft-v21-weighted",
            config={"method": "SFT", "framework": "unsloth", "lora_rank": 32,
                    "lora_alpha": 64, "lr": 2e-4, "epochs": 2,
                    "data": "v21-weighted-1550"},
            settings=wandb.Settings(init_timeout=120))
        WANDB_ACTIVE = True
        print("wandb online")
    except Exception:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="nemotron-reasoning", name="sft-v21-weighted",
            settings=wandb.Settings(init_timeout=30))
        WANDB_ACTIVE = True
        print("wandb offline")
except Exception as e:
    print(f"wandb unavailable: {e}")

# ============================================================
# 4. CONFIG
# ============================================================
OUTPUT_DIR = "/kaggle/working"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "nemotron-lora-adapter")
os.makedirs(ADAPTER_DIR, exist_ok=True)

LORA_RANK = 32
LORA_ALPHA = 64       # v7 proven (0.69 score) — NOT 32
LORA_DROPOUT = 0.0    # 0 enables Unsloth fast kernels
MAX_SEQ_LEN = 4096
NUM_EPOCHS = 2        # v7 used 3, but 2 saves quota
BATCH_SIZE = 2        # Unsloth uses less VRAM, can go higher
GRAD_ACCUM = 4        # effective batch = 8
LR = 2e-4             # v7 proven — NOT 1e-4
RANDOM_SEED = 42

# ============================================================
# 5. LOAD DATA
# ============================================================
print("Loading training data...")
all_examples = []

jsonl_paths = []
for pat in [
    "/kaggle/input/nemotron-training-v21/training_v21.jsonl",
    "/kaggle/input/datasets/bharatmohan/nemotron-training-v21/training_v21.jsonl",
    "/kaggle/input/*/training_v21.jsonl",
    "/kaggle/input/*/*/training_v21.jsonl",
    "/kaggle/input/*/*/*/training_v21.jsonl",
]:
    jsonl_paths.extend(glob.glob(pat))
jsonl_paths = list(dict.fromkeys(jsonl_paths))

if jsonl_paths:
    fpath = jsonl_paths[0]
    print(f"Loading: {fpath}")
    with open(fpath) as f:
        for line in f:
            all_examples.append(json.loads(line))
    print(f"Loaded {len(all_examples)} curated examples")
else:
    print("ERROR: training_v21.jsonl NOT FOUND")
    sys.exit(1)

random.seed(RANDOM_SEED)
random.shuffle(all_examples)

# ============================================================
# 6. LOAD MODEL WITH UNSLOTH (16-bit, no quantization)
# ============================================================
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
print(f"Model at: {MODEL_PATH}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    dtype=None,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Model loaded.")

# ============================================================
# 7. APPLY LoRA VIA UNSLOTH
# ============================================================
# Same 4 targets as our best scoring config (v7=0.69)
# NOT the 11-target config from mayukh18 (which scored 0.65)
target_modules = ["in_proj", "out_proj", "up_proj", "down_proj"]

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=target_modules,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth-optimized
    random_state=RANDOM_SEED,
)
model.print_trainable_parameters()

# ============================================================
# 8. PREPARE DATASET
# ============================================================
records = [{"messages": item["messages"]} for item in all_examples]
dataset = HFDataset.from_list(records)

# Apply chat template
dataset = dataset.map(lambda ex: {
    "text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False,
        add_generation_prompt=False, enable_thinking=True,
    )
})
print(f"Dataset: {len(dataset)} examples")

# ============================================================
# 9. TRAIN
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
        report_to="wandb" if WANDB_ACTIVE else "none",
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
print(f"SFT v21 — Unsloth 16-bit LoRA, weighted data")
print(f"{'='*60}")
print(f"  Examples: {len(dataset)}, Epochs: {NUM_EPOCHS}")
print(f"  Batch: {BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
print(f"  LR: {LR}, LoRA: r={LORA_RANK}, a={LORA_ALPHA}")
print(f"  Targets: {target_modules}")
print(f"  Optimizer: adamw_8bit, Grad ckpt: unsloth")
print(f"{'='*60}\n")

print("Training...")
trainer.train()
elapsed = (time.time() - start_time) / 60
print(f"\nTraining complete in {elapsed:.1f} min")

# ============================================================
# 10. SAVE & ZIP
# ============================================================
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

# Fix adapter_config for inference
config_path = os.path.join(ADAPTER_DIR, "adapter_config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["inference_mode"] = True
    cfg["lora_dropout"] = 0.0
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in ["adapter_config.json", "adapter_model.safetensors"]:
        fpath = os.path.join(ADAPTER_DIR, fname)
        if os.path.exists(fpath):
            zf.write(fpath, fname)
            print(f"  Added {fname}")

print(f"\nsubmission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
print(f"Total: {elapsed:.1f} min")

if WANDB_ACTIVE:
    wandb.log({"total_time_min": elapsed})
    wandb.finish()

print("DONE")
