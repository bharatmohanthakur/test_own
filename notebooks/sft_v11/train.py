"""
SFT v11 — TIR (Tool-Integrated Reasoning) + AIMO3 broad reasoning
- 9,500 puzzle examples with computational CoT (Python code traces)
- 1,800+ AIMO3 math reasoning examples
- Expanded LoRA targets (all 9 module types)
- alpha=64, rank=32, max_seq=2048
- Blackwell RTX Pro 6000 compatible
"""

import sys, os, shutil, stat, gc, math, zipfile, time, json
import importlib, importlib.util

# === Fix 1: Hide causal_conv1d before imports ===
_orig_find_spec = importlib.util.find_spec
def _patched_find_spec(name, *args, **kwargs):
    if 'causal_conv1d' in str(name):
        return None
    return _orig_find_spec(name, *args, **kwargs)
importlib.util.find_spec = _patched_find_spec

# === Fix 2: Triton ptxas-blackwell permission fix ===
try:
    src = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/triton/backends/nvidia/bin/ptxas-blackwell"
    dst = "/tmp/ptxas-blackwell"
    shutil.copy2(src, dst)
    os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    import triton.backends.nvidia as nv_backend
    src_bin = os.path.join(os.path.dirname(nv_backend.__file__), "bin")
    dst_bin = "/tmp/triton_nvidia_bin"
    shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
    for f in os.listdir(dst_bin):
        fp = os.path.join(dst_bin, f)
        if os.path.isfile(fp):
            os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    nv_backend.__file__ = os.path.join(dst_bin, "..", "__init__.py")
    os.environ["TRITON_PTXAS_PATH"] = dst
    print("Triton ptxas-blackwell fixed")
except Exception as e:
    print(f"Triton fix: {e}")

import subprocess, csv, glob, random, re, statistics
import torch, torch.nn.functional as F

# === Fix: Patch mamba_ssm __init__ to skip mamba3 import ===
import glob as _glob
_util_dirs = _glob.glob("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script")
if _util_dirs:
    _util_dir = _util_dirs[0]
    _mamba_init = os.path.join(_util_dir, "mamba_ssm", "__init__.py")
    if os.path.exists(_mamba_init):
        with open(_mamba_init) as _f:
            _src = _f.read()
        if 'mamba3' in _src:
            _writable_mamba = "/tmp/mamba_ssm_patched"
            if os.path.exists(os.path.join(_writable_mamba, "mamba_ssm")):
                shutil.rmtree(os.path.join(_writable_mamba, "mamba_ssm"))
            os.makedirs(_writable_mamba, exist_ok=True)
            shutil.copytree(os.path.join(_util_dir, "mamba_ssm"),
                          os.path.join(_writable_mamba, "mamba_ssm"))
            _patched_init = os.path.join(_writable_mamba, "mamba_ssm", "__init__.py")
            with open(_patched_init) as _f:
                _src2 = _f.read()
            _src2 = _src2.replace('from mamba_ssm.modules.mamba3 import Mamba3', '# mamba3 skipped')
            _src2 = _src2.replace('from mamba_ssm.modules.mamba3 import *', '# mamba3 skipped')
            with open(_patched_init, 'w') as _f:
                _f.write(_src2)
            sys.path.insert(0, _writable_mamba)
            print(f"Patched mamba_ssm: skipped mamba3")
        else:
            if _util_dir not in sys.path:
                sys.path.insert(0, _util_dir)

import kagglehub, mamba_ssm
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# === wandb ===
WANDB_ACTIVE = False
try:
    import wandb
    os.environ["WANDB_API_KEY"] = "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"
    try:
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
        wandb.init(project="nemotron-reasoning", name="sft-v11-tir",
            config={"method": "SFT-TIR", "lora_rank": 32, "lora_alpha": 64,
                    "lr": 1.5e-4, "epochs": 3, "max_seq_len": 2048,
                    "data": "9500-puzzle-tir+1800-aimo3", "targets": "9-modules"},
            tags=["sft", "v11", "tir"],
            settings=wandb.Settings(init_timeout=120))
        WANDB_ACTIVE = True
        print("wandb online")
    except Exception as e2:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="nemotron-reasoning", name="sft-v11-tir",
            config={"method": "SFT-TIR"}, tags=["sft", "v11", "tir"])
        WANDB_ACTIVE = True
        print(f"wandb offline: {e2}")
except Exception as e:
    print(f"wandb unavailable: {e}")

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
OUTPUT_DIR = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LORA_RANK = 32
LORA_ALPHA = 64
MAX_SEQ_LEN = 2048
NUM_EPOCHS = 3
LR = 1.5e-4
BATCH_SIZE = 1
GRAD_ACCUM = 16

# ============================================================
# LOAD DATA
# ============================================================
print("Loading training data...")

# Try v11 JSONL dataset first
jsonl_paths = (
    glob.glob("/kaggle/input/nemotron-training-v11/*.jsonl") +
    glob.glob("/kaggle/input/nemotron-training-v11/**/*.jsonl", recursive=True)
)
print(f"Found JSONL paths: {jsonl_paths}")

all_examples = []

if jsonl_paths:
    fpath = jsonl_paths[0]
    print(f"Loading TIR dataset: {fpath}")
    with open(fpath) as f:
        for line in f:
            all_examples.append(json.loads(line))
    print(f"Loaded {len(all_examples)} TIR examples")

if len(all_examples) < 100:
    print("TIR dataset not found, generating from train.csv...")
    all_examples = []
    possible_paths = (
        glob.glob("/kaggle/input/*/train.csv") +
        glob.glob("/kaggle/input/*/*/train.csv") +
        glob.glob("/kaggle/input/*/*/*/train.csv")
    )
    data_path = possible_paths[0]
    print(f"Using: {data_path}")

    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)

    def classify_puzzle(p):
        p = p[:300].lower()
        if 'bit manipulation' in p: return 'bit'
        if 'encryption' in p or 'decrypt' in p: return 'cipher'
        if 'gravitational' in p: return 'gravity'
        if 'unit conversion' in p: return 'unit'
        if 'numeral' in p: return 'numeral'
        if 'transformation rules' in p or 'wonderland' in p: return 'equation'
        return 'unknown'

    def make_cot(prompt, answer, cat):
        """Fallback CoT if TIR dataset not available."""
        if cat == 'gravity':
            # Parse and compute
            import re as _re
            pairs = _re.findall(r't\s*=\s*([\d.]+)\s*s?,\s*distance\s*=\s*([\d.]+)', prompt)
            if pairs:
                g_vals = [2*float(d)/float(t)**2 for t, d in pairs]
                avg_g = sum(g_vals)/len(g_vals)
                after = prompt.split('Now')[1] if 'Now' in prompt else prompt[-200:]
                tm = _re.search(r't\s*=\s*([\d.]+)', after)
                if tm:
                    tt = float(tm.group(1))
                    comp = "\n".join([f"t={t}, d={d}: g = 2*{d}/{t}^2 = {2*float(d)/float(t)**2:.4f}" for t,d in pairs])
                    return f"<think>\nGravity problem: d = 0.5*g*t^2, so g = 2d/t^2.\n\n{comp}\n\nAvg g = {avg_g:.4f}\nd = 0.5 * {avg_g:.4f} * {tt}^2 = {0.5*avg_g*tt*tt:.2f}\n\nAnswer: {answer}\n</think>\n\n\\boxed{{{answer}}}"
            return f"<think>\nGravity problem. Using d=0.5*g*t^2, computing g from examples then applying.\n\nAnswer: {answer}\n</think>\n\n\\boxed{{{answer}}}"
        elif cat == 'cipher':
            return f"<think>\nSubstitution cipher. Building letter mapping from example pairs by aligning characters.\nApplying mapping to decrypt target.\n\nResult: {answer}\n</think>\n\n\\boxed{{{answer}}}"
        elif cat == 'numeral':
            return f"<think>\nRoman numeral conversion.\nI=1, V=5, X=10, L=50, C=100, D=500, M=1000\nSubtractive: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900\n\nConverting: {answer}\n</think>\n\n\\boxed{{{answer}}}"
        elif cat == 'unit':
            return f"<think>\nUnit conversion. Computing factor = output/input for each example pair.\nAveraging factors and applying to target.\n\nResult: {answer}\n</think>\n\n\\boxed{{{answer}}}"
        elif cat == 'bit':
            return f"<think>\nBit manipulation. Testing XOR, rotation, NOT, reverse on examples to find the rule.\nApplying identified operation to target.\n\nResult: {answer}\n</think>\n\n\\boxed{{{answer}}}"
        else:
            return f"<think>\nSymbol transformation. Building character mapping from example pairs.\nApplying to target.\n\nResult: {answer}\n</think>\n\n\\boxed{{{answer}}}"

    for row in rows:
        prompt_text, answer = row[1], row[2]
        cat = classify_puzzle(prompt_text)
        user_msg = prompt_text + "\nPlease put your final answer inside \\boxed{}."
        asst_msg = make_cot(prompt_text, answer, cat)
        all_examples.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": asst_msg}
            ]
        })
    print(f"Generated {len(all_examples)} examples from train.csv")

random.seed(42)
random.shuffle(all_examples)
print(f"Total training examples: {len(all_examples)}")

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading Nemotron-3-Nano-30B...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Model loaded.")

# === Fix 3: Force slow path AFTER model load ===
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        mod.is_fast_path_available = False
        print(f"Patched {name}: is_fast_path_available = False")

# === Fix 4: Replace Triton rmsnorm_fn with pure PyTorch ===
def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                     group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    if upcast:
        x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()
    if bias is not None:
        out = out + bias.float()
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype)

for name, mod in list(sys.modules.items()):
    if hasattr(mod, 'rmsnorm_fn'):
        mod.rmsnorm_fn = _pure_rmsnorm_fn
        print(f"Patched rmsnorm_fn in {name}")

# ============================================================
# APPLY LoRA — EXPANDED TARGETS (all 9 module types)
# ============================================================
print("Setting up LoRA with expanded targets...")
lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA,
    target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|up_proj|down_proj|gate_proj)$",
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.enable_input_require_grads()

# ============================================================
# TOKENIZE — dynamic padding
# ============================================================
print("Tokenizing...")

class SFTDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.items = []
        skipped = 0
        for item in examples:
            messages = item["messages"]
            user_only = tokenizer.apply_chat_template(
                [messages[0]], tokenize=False, add_generation_prompt=True
            )
            user_ids = tokenizer(user_only, add_special_tokens=False)["input_ids"]
            user_len = len(user_ids)

            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            enc = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
            ids = enc["input_ids"]

            # Check boxed not truncated
            decoded = tokenizer.decode(ids[user_len:], skip_special_tokens=True)
            if 'boxed' not in decoded:
                skipped += 1
                continue

            labels = list(ids)
            for j in range(min(user_len, len(labels))):
                labels[j] = -100

            self.items.append({
                "input_ids": ids,
                "labels": labels,
                "length": len(ids)
            })

        print(f"Tokenized {len(self.items)} examples (skipped {skipped} truncated)")
        self.items.sort(key=lambda x: x["length"])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

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
    lr=LR, weight_decay=0.01
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
print(f"SFT v11 — TIR + AIMO3 Training")
print(f"{'='*60}")
print(f"  Examples: {len(dataset)}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch: {BATCH_SIZE} x grad_accum={GRAD_ACCUM} = effective {BATCH_SIZE*GRAD_ACCUM}")
print(f"  Steps: ~{total_steps}")
print(f"  LR: {LR}, warmup: {warmup_steps}")
print(f"  LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, targets=9 modules")
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

zip_path = "/kaggle/working/submission.zip"
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

print("DONE")
