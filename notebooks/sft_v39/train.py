"""
SFT v34 — BINARY REBUILD (tracking-informed data iteration)

Based on tracking system analysis of v30 (0.54) vs v31 (0.66):
- v22 SFT (0.65) is strong on cipher/gravity/unit/roman (4/6 types saturate at 10/10)
- v22 is terrible on binary (1/10) and equation (0/10)
- v22's binary examples teach "I identify the pattern" hallucination without showing work
- v30's mixed data BROKE cipher by teaching Donald labels + VOCAB fill → hallucinated vocab

v34 strategy:
- KEEP v22's cipher/gravity/unit/roman/equation verbatim (419 examples)
- DROP v22's weak binary examples (the hallucination ones)
- ADD 150 new binary examples from tracking/verifiers/binary.py solver, showing
  explicit per-bit derivation (IDENTITY, CONSTANT, NOT, 2-input gates, MAJORITY)
- NO Donald labels (LEN:/TABLE:/VOCAB fill) — they leaked into cipher/equation in v30
- Total: 569 examples

Config: r=32 alpha=64 LR=2e-4 3 epochs (same as v22 proven baseline).
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
        wandb.init(project="nemotron-reasoning", name="sft-v39-full-kh0a-boxed-weight",
            config={"method": "SFT", "framework": "unsloth", "lora_rank": 32,
                    "lora_alpha": 64, "lr": 2e-4, "epochs": 2},
            settings=wandb.Settings(init_timeout=120))
        WANDB_ACTIVE = True
        print("wandb online")
    except Exception:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="nemotron-reasoning", name="sft-v39-full-kh0a-boxed-weight",
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
LORA_ALPHA = 16       # Kh0a 0.73 exact (we were at 64)
LORA_DROPOUT = 0.1    # Kh0a 0.73 exact (we were at 0.0)
MAX_SEQ_LEN = 3500    # Kh0a 0.73 exact (we were at 4096)
NUM_EPOCHS = 2        # Kh0a 0.73 exact
BATCH_SIZE = 2
GRAD_ACCUM = 1        # Kh0a 0.73 exact (we were at 4); effective batch = 2
LR = 1e-4             # Kh0a 0.73 exact
WARMUP_RATIO = 0.03   # Kh0a 0.73 exact
RANDOM_SEED = 42

# ============================================================
# 5. LOAD DATA
# ============================================================
print("Loading training data...")
all_examples = []

jsonl_paths = []
for pat in [
    "/kaggle/input/nemotron-training-v39/training_v39.jsonl",
    "/kaggle/input/datasets/bharatmohan/nemotron-training-v39/training_v39.jsonl",
    "/kaggle/input/*/training_v39.jsonl",
    "/kaggle/input/*/*/training_v39.jsonl",
    "/kaggle/input/*/*/*/training_v39.jsonl",
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
    print("ERROR: training_v39.jsonl NOT FOUND")
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
# Kh0a 0.73 config: 11 targets including embed_tokens + lm_head.
# Unsloth auto-adds MoE expert LoRA when gate_proj is present.
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj", "out_proj",
    "up_proj", "down_proj", "gate_proj",
    "embed_tokens", "lm_head",
]

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

# ── Weighted loss: upweight final \boxed{answer} tokens 5x (Kh0a 0.73) ─────
BOXED_LOSS_WEIGHT = 5.0
_boxed_marker_ids = tokenizer.encode("\\boxed{", add_special_tokens=False)
print(f"[loss] \\boxed{{ marker = {len(_boxed_marker_ids)} token IDs: {_boxed_marker_ids}")

def _weighted_boxed_loss(outputs, labels, num_items_in_batch=None):
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    batch_size, seq_len = shift_labels.shape
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(batch_size, seq_len)
    weights = torch.ones(batch_size, seq_len, device=per_token_loss.device)
    marker = torch.tensor(_boxed_marker_ids, device=shift_labels.device)
    marker_len = len(_boxed_marker_ids)
    for bi in range(batch_size):
        last_pos = -1
        for i in range(seq_len - marker_len + 1):
            if torch.equal(shift_labels[bi, i:i+marker_len], marker):
                last_pos = i
        if last_pos >= 0:
            weights[bi, last_pos:] = BOXED_LOSS_WEIGHT
    mask = (shift_labels != -100).float()
    return (per_token_loss * weights * mask).sum() / (weights * mask).sum()

import inspect
_extra_kwargs = {}
if 'compute_loss_func' in inspect.signature(SFTTrainer.__init__).parameters:
    _extra_kwargs['compute_loss_func'] = _weighted_boxed_loss
    print(f"[loss] Weighted loss active: {BOXED_LOSS_WEIGHT}x on \\boxed{{}} tokens")
else:
    print("[loss] compute_loss_func not supported in this trl version — using default")

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
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        optim="adamw_8bit",
        seed=RANDOM_SEED,
        report_to="wandb" if WANDB_ACTIVE else "none",
        dataloader_num_workers=2,
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
    ),
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    dataset_kwargs={"skip_prepare_dataset": False},
    packing=False,
    **_extra_kwargs,
)

print(f"\n{'='*60}")
print(f"SFT v39 — full Kh0a 3910 + BOXED_LOSS_WEIGHT=5.0 + Kh0a 11-target config")
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
