"""
SFT+GRPO v26 — Quick test with 20 SFT + 20 GRPO samples
Disables torch.compile BEFORE any imports to fix Unsloth GRPO FX tracing
"""

# CRITICAL: Disable torch.compile at the earliest point
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_TORCH_COMPILE"] = "1"

import sys, shutil, stat, gc, math, zipfile, time, json, types
import importlib, importlib.util
import subprocess, glob, random, re, csv

# Monkey-patch torch.compile to no-op BEFORE Unsloth imports
import torch
import torch._dynamo
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

_original_compile = torch.compile
def _no_compile(model=None, *args, **kwargs):
    if model is None:
        return lambda f: f
    return model
torch.compile = _no_compile
print("torch.compile disabled globally")

# ============================================================
# INSTALL PACKAGES
# ============================================================
print("Installing packages...")

PKG_DIR = None
for candidate in [
    "/kaggle/input/datasets/mayukh18/nemotron-packages/packages",
    "/kaggle/input/nemotron-packages/packages",
]:
    if os.path.exists(candidate):
        PKG_DIR = candidate
        break

if PKG_DIR:
    subprocess.run(
        f"pip install -q --no-index --find-links {PKG_DIR} "
        f"unsloth trl peft transformers datasets accelerate bitsandbytes",
        shell=True
    )
    for pattern in ["causal_conv1d*.whl", "mamba_ssm*.whl"]:
        wheels = sorted(glob.glob(os.path.join(PKG_DIR, "..", pattern)) +
                       glob.glob(os.path.join(PKG_DIR, pattern)))
        if wheels:
            subprocess.run(f"pip install -q {wheels[-1]}", shell=True)
else:
    print("ERROR: nemotron-packages not found!")
    sys.exit(1)

import torch
import torch.nn.functional as F
import kagglehub
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset as HFDataset

# ============================================================
# CONFIG
# ============================================================
OUTPUT_DIR = "/kaggle/working"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "nemotron-lora-adapter")
os.makedirs(ADAPTER_DIR, exist_ok=True)

LORA_RANK = 32
LORA_ALPHA = 64
MAX_SEQ_LEN = 4096
RANDOM_SEED = 42

# SFT config (v7 proven)
SFT_EPOCHS = 3
SFT_LR = 2e-4
SFT_BATCH = 1
SFT_GRAD_ACCUM = 16

# GRPO config
GRPO_LR = 5e-6
GRPO_STEPS = 100
GRPO_GENERATIONS = 4
GRPO_BATCH = 1
GRPO_GRAD_ACCUM = 4
GRPO_TEMP = 0.9

# ============================================================
# LOAD DATA
# ============================================================
print("Loading training data...")

# Find SFT data (504 verified) and GRPO data (distilled)
sft_examples = []
grpo_examples = []

# Search for both files
for pat in [
    "/kaggle/input/nemotron-training-v24/sft_504.jsonl",
    "/kaggle/input/datasets/bharatmohan/nemotron-training-v24/sft_504.jsonl",
    "/kaggle/input/*/sft_504.jsonl",
    "/kaggle/input/*/*/sft_504.jsonl",
    "/kaggle/input/*/*/*/sft_504.jsonl",
]:
    for f in glob.glob(pat):
        with open(f) as fh:
            for line in fh:
                sft_examples.append(json.loads(line))
        print(f"SFT data: {f} ({len(sft_examples)} examples)")
        break
    if sft_examples:
        break

for pat in [
    "/kaggle/input/nemotron-training-v24/grpo_100.jsonl",
    "/kaggle/input/datasets/bharatmohan/nemotron-training-v24/grpo_100.jsonl",
    "/kaggle/input/*/grpo_100.jsonl",
    "/kaggle/input/*/*/grpo_100.jsonl",
    "/kaggle/input/*/*/*/grpo_100.jsonl",
]:
    for f in glob.glob(pat):
        with open(f) as fh:
            for line in fh:
                grpo_examples.append(json.loads(line))
        print(f"GRPO data: {f} ({len(grpo_examples)} examples)")
        break
    if grpo_examples:
        break

if not sft_examples:
    print("ERROR: sft_504.jsonl not found!")
    sys.exit(1)
if not grpo_examples:
    print("ERROR: grpo_100.jsonl not found!")
    sys.exit(1)

random.seed(RANDOM_SEED)
random.shuffle(sft_examples)

# v26: Limit to 20 samples each for quick test
sft_examples = sft_examples[:20]
grpo_examples = grpo_examples[:20]
print(f"LIMITED: {len(sft_examples)} SFT + {len(grpo_examples)} GRPO samples")

# ============================================================
# LOAD MODEL (Unsloth 16-bit)
# ============================================================
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
print(f"Model at: {MODEL_PATH}")

# Disable torch.compile to fix Mamba FX tracing issues in GRPO
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "0"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    dtype=None,
    unsloth_force_compile=False,  # CRITICAL: prevents torch.compile FX tracing failure
    attn_implementation="eager",
)

# Patch Mamba fast_path for generation compatibility
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        if hasattr(mod, 'is_fast_path_available'):
            mod.is_fast_path_available = False
            print(f"Patched {name}: is_fast_path_available = False")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # SFT uses right padding
print("Model loaded.")

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.0,
    target_modules=["in_proj", "out_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=RANDOM_SEED,
)
model.print_trainable_parameters()

# ============================================================
# PHASE 1: SFT (custom loop — proven to score 0.69)
# ============================================================
print(f"\n{'='*60}")
print(f"PHASE 1: SFT — 504 verified examples, 3 epochs")
print(f"{'='*60}\n")

# Tokenize SFT data
from torch.utils.data import Dataset, DataLoader

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

            self.items.append({"input_ids": ids, "labels": labels, "length": len(ids)})

        print(f"Tokenized {len(self.items)} examples (skipped {skipped})")
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

dataset = SFTDataset(sft_examples, tokenizer, MAX_SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=SFT_BATCH, shuffle=True, collate_fn=collate_fn)

# Custom training loop
model.train()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=SFT_LR, weight_decay=0.01
)
total_steps = (len(dataloader) * SFT_EPOCHS) // SFT_GRAD_ACCUM
warmup_steps = max(10, total_steps // 20)

def cosine_lr(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)

global_step = 0
start_time = time.time()

for epoch in range(SFT_EPOCHS):
    running_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / SFT_GRAD_ACCUM
        loss.backward()
        running_loss += outputs.loss.item()
        n_batches += 1

        if (i + 1) % SFT_GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 10 == 0:
                avg = running_loss / n_batches
                elapsed = (time.time() - start_time) / 60
                print(f"  SFT ep{epoch+1} step {global_step}/{total_steps} loss={avg:.4f} {elapsed:.1f}min", flush=True)

    avg_loss = running_loss / max(n_batches, 1)
    elapsed = (time.time() - start_time) / 60
    print(f"SFT Epoch {epoch+1}/{SFT_EPOCHS} loss={avg_loss:.4f} {elapsed:.1f}min", flush=True)
    gc.collect()
    torch.cuda.empty_cache()

sft_time = (time.time() - start_time) / 60
print(f"\nSFT complete in {sft_time:.1f} min")

# ============================================================
# PHASE 2: GRPO (100 distilled examples)
# ============================================================
print(f"\n{'='*60}")
print(f"PHASE 2: GRPO — 100 distilled examples")
print(f"{'='*60}\n")

# Switch padding for generation
tokenizer.padding_side = "left"

# Re-apply Mamba patches (some may have been reset during SFT)
for name, mod in list(sys.modules.items()):
    if "modeling_nemotron_h" in name:
        if hasattr(mod, 'is_fast_path_available'):
            mod.is_fast_path_available = False

# Clean up GPU memory before GRPO
gc.collect()
torch.cuda.empty_cache()

# Competition verify function
def verify(stored_answer, predicted):
    stored_answer = str(stored_answer).strip()
    predicted = str(predicted).strip()
    if not predicted: return False
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        return math.isclose(float(stored_answer), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except:
        return predicted.lower() == stored_answer.lower()

def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else ""

SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# Reward functions
def correctness_reward(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    return [2.0 if verify(a, extract_boxed(r)) else 0.0 for r, a in zip(responses, answer)]

def format_reward(completions, **kwargs):
    responses = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    return [0.5 if '\\boxed{' in r else 0.0 for r in responses]

# Build GRPO dataset
grpo_data = []
for ex in grpo_examples:
    prompt_text = ex["messages"][0]["content"] if "messages" in ex else ex.get("prompt", "")
    if not prompt_text.endswith(SUFFIX):
        prompt_text += SUFFIX
    grpo_data.append({
        "prompt": [{"role": "user", "content": prompt_text}],
        "answer": ex.get("answer", ""),
    })

grpo_dataset = HFDataset.from_list(grpo_data)
print(f"GRPO dataset: {len(grpo_dataset)} prompts")

grpo_start = time.time()

training_args = GRPOConfig(
    output_dir=os.path.join(OUTPUT_DIR, "grpo_output"),
    learning_rate=GRPO_LR,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=5,
    per_device_train_batch_size=GRPO_BATCH,
    gradient_accumulation_steps=GRPO_GRAD_ACCUM,
    num_generations=GRPO_GENERATIONS,
    max_prompt_length=512,
    max_completion_length=2048,
    max_grad_norm=0.1,
    temperature=GRPO_TEMP,
    max_steps=GRPO_STEPS,
    save_steps=GRPO_STEPS,
    report_to="none",
    seed=RANDOM_SEED,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[correctness_reward, format_reward],
    args=training_args,
    train_dataset=grpo_dataset,
)

print("Starting GRPO...", flush=True)
trainer.train()
grpo_time = (time.time() - grpo_start) / 60
print(f"\nGRPO complete in {grpo_time:.1f} min")

# ============================================================
# SAVE & ZIP
# ============================================================
print("Saving adapter...")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

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

total_time = sft_time + grpo_time
print(f"\nsubmission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
print(f"Total: SFT={sft_time:.1f}min + GRPO={grpo_time:.1f}min = {total_time:.1f}min")
print("DONE")
