"""
SFT v19 — RunPod A100 version (Cascade-2 hyperparams)
1699 examples (1200 puzzles + 499 math), 2 epochs (budget-constrained)
Lower LR=5e-5, weight_decay=0.1, beta2=0.98 (Nemotron-Cascade-2 recipe)
"""

import sys, os, gc, math, zipfile, time, json, random, re, csv
import subprocess

# Install dependencies
print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "peft", "transformers", "accelerate", "datasets", "wandb",
    "mamba_ssm", "causal_conv1d"], check=False)

import torch, torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# wandb
WANDB_ACTIVE = False
try:
    import wandb
    wandb_config = {"method": "SFT", "lora_rank": 32, "lora_alpha": 64,
                    "lr": 5e-5, "epochs": 2, "max_seq_len": 4096,
                    "data": "v18-1699-gf2-math", "targets": "4-modules", "gpu": "A100-80GB",
                    "weight_decay": 0.1, "beta2": 0.98}
    wandb_tags = ["sft", "v19", "runpod", "a100", "cascade-lr"]
    try:
        wandb.login(key="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", relogin=True)
        wandb.init(project="nemotron-reasoning", name="sft-v19-cascade-runpod",
            config=wandb_config, tags=wandb_tags,
            settings=wandb.Settings(init_timeout=120))
        WANDB_ACTIVE = True
        print(f"wandb online")
    except Exception as e:
        print(f"wandb failed: {e}")
except Exception as e:
    print(f"wandb unavailable: {e}")

# ============================================================
# CONFIG
# ============================================================
MODEL_ID = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
OUTPUT_DIR = "/workspace/adapter"
WORK_DIR = "/workspace"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LORA_RANK = 32
LORA_ALPHA = 64
MAX_SEQ_LEN = 4096
NUM_EPOCHS = 2
LR = 5e-5
BATCH_SIZE = 1
GRAD_ACCUM = 16
RANDOM_SEED = 42

# ============================================================
# LOAD DATA
# ============================================================
print("Loading training data...")
all_examples = []

data_path = os.path.join(WORK_DIR, "training_v18.jsonl")
if os.path.exists(data_path):
    with open(data_path) as f:
        for line in f:
            all_examples.append(json.loads(line))
    print(f"Loaded {len(all_examples)} examples from {data_path}")
else:
    print(f"ERROR: {data_path} not found! Upload training_v18.jsonl to /workspace/")
    sys.exit(1)

random.seed(RANDOM_SEED)
random.shuffle(all_examples)
print(f"Total training examples: {len(all_examples)}")

# ============================================================
# LOAD MODEL from HuggingFace
# ============================================================
import kagglehub
print("Downloading Nemotron-3-Nano-30B via kagglehub...")
MODEL_PATH = kagglehub.model_download(MODEL_ID)
print(f"Model at: {MODEL_PATH}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Model loaded.")

# ============================================================
# APPLY LoRA — 4 targets (same as v7, our best score)
# ============================================================
print("Setting up LoRA (4 targets, v7-style)...")
lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.enable_input_require_grads()

# ============================================================
# TOKENIZE
# ============================================================
print("Tokenizing...")

class SFTDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.items = []
        skipped = 0
        for item in examples:
            messages = item["messages"]
            user_only = tokenizer.apply_chat_template(
                [messages[0]], tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )
            user_ids = tokenizer(user_only, add_special_tokens=False)["input_ids"]
            user_len = len(user_ids)

            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=False, enable_thinking=True,
            )
            enc = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
            ids = enc["input_ids"]

            decoded = tokenizer.decode(ids[user_len:], skip_special_tokens=True)
            if 'boxed' not in decoded:
                skipped += 1
                continue

            labels = list(ids)
            for j in range(min(user_len, len(labels))):
                labels[j] = -100

            self.items.append({
                "input_ids": ids, "labels": labels, "length": len(ids)
            })

        print(f"Tokenized {len(self.items)} examples (skipped {skipped} truncated)")
        self.items.sort(key=lambda x: x["length"])

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask, labels = [], [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad_len)
        attention_mask.append([1] * len(b["input_ids"]) + [0] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

dataset = SFTDataset(all_examples, tokenizer, MAX_SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

lengths = [item["length"] for item in dataset.items]
print(f"Seq lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}, median={sorted(lengths)[len(lengths)//2]}")

# ============================================================
# TRAINING
# ============================================================
model.train()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=0.1, betas=(0.9, 0.98)
)
total_steps = (len(dataloader) * NUM_EPOCHS) // GRAD_ACCUM
warmup_steps = max(10, total_steps // 20)

def cosine_lr(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)

print(f"\n{'='*60}")
print(f"SFT v19 — RunPod A100 80GB (Cascade-2 LR)")
print(f"{'='*60}")
print(f"  Examples: {len(dataset)}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch: {BATCH_SIZE} x grad_accum={GRAD_ACCUM} = effective {BATCH_SIZE*GRAD_ACCUM}")
print(f"  Steps: ~{total_steps}")
print(f"  LR: {LR}, warmup: {warmup_steps}")
print(f"  LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, targets=4 modules")
print(f"  Max seq len: {MAX_SEQ_LEN}")
print(f"{'='*60}\n")

global_step = 0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        running_loss += outputs.loss.item()
        n_batches += 1

        if (i + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 50 == 0:
                avg = running_loss / n_batches
                lr_now = scheduler.get_last_lr()[0]
                elapsed = (time.time() - start_time) / 60
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f"  ep {epoch+1} | step {global_step}/{total_steps} | loss {avg:.4f} | lr {lr_now:.2e} | {elapsed:.1f}min | mem {mem_gb:.1f}GB")
                if WANDB_ACTIVE:
                    wandb.log({"loss": avg, "lr": lr_now, "step": global_step,
                               "epoch": epoch + 1, "mem_gb": mem_gb})

            if global_step % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    if n_batches % GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    avg_loss = running_loss / max(n_batches, 1)
    elapsed = (time.time() - start_time) / 60
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} — loss: {avg_loss:.4f} — {elapsed:.1f}min")
    if WANDB_ACTIVE:
        wandb.log({"epoch_loss": avg_loss, "epoch": epoch+1})
    gc.collect()
    torch.cuda.empty_cache()

total_time = (time.time() - start_time) / 60
print(f"\nTraining complete in {total_time:.1f} min")

# ============================================================
# SAVE & ZIP
# ============================================================
print("Saving adapter...")
model.save_pretrained(OUTPUT_DIR)

zip_path = os.path.join(WORK_DIR, "submission.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            zf.write(fpath, fname)

size_mb = os.path.getsize(zip_path) / 1024 / 1024
print(f"submission.zip: {size_mb:.1f} MB")
print(f"Total time: {total_time:.1f} min")

if WANDB_ACTIVE:
    wandb.log({"total_time_min": total_time, "submission_size_mb": size_mb})
    wandb.finish()

print("DONE — download /workspace/submission.zip")
