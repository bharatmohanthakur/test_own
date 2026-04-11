"""
SFT v16-tiny — 1000 curated examples, 3 epochs, fast iteration
- 200 per category (5 types, no equation_transform)
- Research shows 600 curated > 9500 raw
- 4 LoRA targets, alpha=64, LR=2e-4, 3 epochs
- ~2.5 hour run on RTX Pro 6000
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

# === wandb — try online first, fallback to offline ===
WANDB_ACTIVE = False
try:
    import wandb
    wandb_config = {"method": "SFT", "lora_rank": 32, "lora_alpha": 64,
                    "lr": 2e-4, "epochs": 3, "max_seq_len": 4096,
                    "data": "v16-tiny-1000-curated", "targets": "4-modules"}
    wandb_tags = ["sft", "v16-tiny", "curated", "3epoch"]
    try:
        wandb.login(key="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", relogin=True)
        wandb.init(project="nemotron-reasoning", name="sft-v16-tiny-3ep",
            config=wandb_config, tags=wandb_tags,
            settings=wandb.Settings(init_timeout=120))
        WANDB_ACTIVE = True
        print(f"wandb online (mode={wandb.run.settings.mode})")
    except Exception:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="nemotron-reasoning", name="sft-v16-tiny-3ep",
            config=wandb_config, tags=wandb_tags,
            settings=wandb.Settings(init_timeout=30))
        WANDB_ACTIVE = True
        print("wandb offline (fallback)")
except Exception as e:
    print(f"wandb unavailable: {e}")

# ============================================================
# CONFIG — v7-style (our best: 0.69) with better data
# ============================================================
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
OUTPUT_DIR = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LORA_RANK = 32
LORA_ALPHA = 64
MAX_SEQ_LEN = 4096
NUM_EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 16
RANDOM_SEED = 42

# ============================================================
# LOAD DATA — prefer pre-computed v15 dataset, fallback to inline solver
# ============================================================
print("Loading training data...")

all_examples = []

# Try pre-uploaded v15 dataset first
jsonl_paths = (
    glob.glob("/kaggle/input/nemotron-training-v16-tiny/training_v16_tiny.jsonl") +
    glob.glob("/kaggle/input/nemotron-training-v16-tiny/**/training_v16_tiny.jsonl", recursive=True) +
    glob.glob("/kaggle/input/*/training_v16_tiny.jsonl")
)

if jsonl_paths:
    fpath = jsonl_paths[0]
    print(f"Loading v16-tiny curated dataset: {fpath}")
    with open(fpath) as f:
        for line in f:
            all_examples.append(json.loads(line))
    print(f"Loaded {len(all_examples)} v16-tiny examples")

# Fallback: generate from train.csv with inline solvers
if len(all_examples) < 100:
    print("v16-tiny dataset not found, generating from train.csv with solvers...")
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
        p = p[:400].lower()
        if 'bit manipulation' in p: return 'bit_manipulation'
        if 'encryption' in p or 'decrypt' in p: return 'cipher'
        if 'gravitational' in p or 'falling distance' in p: return 'gravity'
        if 'unit conversion' in p: return 'unit_conversion'
        if 'numeral' in p: return 'numeral_system'
        if 'transformation rules' in p: return 'equation_transform'
        return 'unknown'

    def solve_gravity(prompt, answer):
        pairs = re.findall(r't\s*=\s*([\d.]+)\s*s?,\s*distance\s*=\s*([\d.]+)', prompt)
        target_m = re.search(r'for\s+t\s*=\s*([\d.]+)', prompt.split('Now')[1] if 'Now' in prompt else prompt[-300:], re.IGNORECASE)
        if not pairs or not target_m: return None
        t_t = float(target_m.group(1))
        g_vals = [2*float(d)/float(t)**2 for t, d in pairs]
        avg_g = sum(g_vals)/len(g_vals)
        lines = ["Given d = 0.5*g*t^2, computing g from examples:"]
        for t, d in pairs:
            lines.append(f"  t={t}, d={d}: g = 2*{d}/{t}^2 = {2*float(d)/float(t)**2:.4f}")
        lines.append(f"\nAvg g = {avg_g:.4f}")
        result = 0.5*avg_g*t_t**2
        lines.append(f"d = 0.5 * {avg_g:.4f} * {t_t}^2 = {result:.2f}")
        return "\n".join(lines)

    def solve_numeral(prompt, answer):
        roman_map = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
        lines = ["Roman numeral conversion:"]
        lines.append("M=1000, CM=900, D=500, CD=400, C=100, XC=90, L=50, XL=40, X=10, IX=9, V=5, IV=4, I=1")
        lines.append(f"\nResult: {answer}")
        return "\n".join(lines)

    def solve_unit(prompt, answer):
        pairs = re.findall(r'([\d.]+)\s*\w*\s+becomes\s+([\d.]+)', prompt)
        if not pairs: return None
        factors = [float(o)/float(i) for i, o in pairs]
        avg_f = sum(factors)/len(factors)
        lines = ["Computing conversion factor from examples:"]
        for i, o in pairs:
            lines.append(f"  {i} -> {o}: factor = {float(o)/float(i):.6f}")
        lines.append(f"\nAvg factor = {avg_f:.6f}")
        lines.append(f"Result: {answer}")
        return "\n".join(lines)

    def solve_cipher(prompt, answer):
        parts = re.split(r'Now,?\s+decrypt\s+the\s+following\s+text:\s*', prompt, flags=re.IGNORECASE)
        if len(parts) < 2: return None
        target = parts[1].strip()
        mapping = {}
        for line in parts[0].split('\n'):
            m = re.match(r'^(.+?)\s*->\s*(.+?)$', line.strip())
            if m:
                for ec, dc in zip(m.group(1).strip(), m.group(2).strip()):
                    if ec != ' ' and dc != ' ':
                        mapping[ec.lower()] = dc.lower()
        lines = [f"Building letter mapping from {len(mapping)} cipher pairs."]
        sorted_m = sorted(mapping.items())
        lines.append(f"Mapping: {', '.join(f'{k}->{v}' for k,v in sorted_m)}")
        lines.append(f"\nDecrypting '{target}' -> '{answer}'")
        return "\n".join(lines)

    def solve_bit(prompt, answer):
        examples = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
        if not examples: return None
        inputs = [int(e[0], 2) for e in examples]
        outputs = [int(e[1], 2) for e in examples]

        # Test NOT
        if all(o == (~i & 0xFF) for i, o in zip(inputs, outputs)):
            return f"NOT (bitwise complement). Result: {answer}"

        # Test XOR
        xor_masks = set(i ^ o for i, o in zip(inputs, outputs))
        if len(xor_masks) == 1:
            mask = xor_masks.pop()
            return f"XOR with mask {mask:08b}. Result: {answer}"

        # Test rotations/shifts
        for shift in range(1, 8):
            if all(o == (((i << shift) & 0xFF) | (i >> (8 - shift))) for i, o in zip(inputs, outputs)):
                return f"Left rotation by {shift}. Result: {answer}"
            if all(o == ((i >> shift) | ((i << (8 - shift)) & 0xFF)) for i, o in zip(inputs, outputs)):
                return f"Right rotation by {shift}. Result: {answer}"

        # Test Majority/Choice with masks
        for m1 in range(256):
            for m2 in range(256):
                if all(o == ((i & m1) | (m1 & m2) | (i & m2)) for i, o in zip(inputs, outputs)):
                    return f"Majority function with masks {m1:08b} and {m2:08b}. Result: {answer}"
                if all(o == ((i & m1) | ((~i & 0xFF) & m2)) for i, o in zip(inputs, outputs)):
                    return f"Choice function with masks {m1:08b} and {m2:08b}. Result: {answer}"

        return f"Bit manipulation pattern analysis. Result: {answer}"

    def solve_equation(prompt, answer):
        parts = re.split(r'Now,?\s+determine\s+the\s+result\s+for:\s*', prompt, flags=re.IGNORECASE)
        if len(parts) < 2: return None
        return f"Analyzing transformation rules from examples.\nResult: {answer}"

    solvers = {
        'gravity': solve_gravity, 'numeral_system': solve_numeral,
        'unit_conversion': solve_unit, 'cipher': solve_cipher,
        'bit_manipulation': solve_bit, 'equation_transform': solve_equation,
    }

    for row in rows:
        prompt_text, answer = row[1], row[2].strip()
        cat = classify_puzzle(prompt_text)
        solver = solvers.get(cat)
        if not solver:
            continue
        try:
            cot = solver(prompt_text, answer)
        except:
            cot = f"Solving {cat} puzzle.\nResult: {answer}"
        if cot is None:
            cot = f"Solving {cat} puzzle.\nResult: {answer}"
        user_msg = prompt_text + "\nPlease put your final answer inside \\boxed{}."
        asst_msg = f"<think>\n{cot}\n</think>\n\n\\boxed{{{answer}}}"
        all_examples.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": asst_msg}
            ]
        })
    print(f"Generated {len(all_examples)} examples from train.csv")

random.seed(RANDOM_SEED)
random.shuffle(all_examples)
print(f"Total training examples: {len(all_examples)}")

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading Nemotron-3-Nano-30B...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    offload_folder="/kaggle/tmp/offload",
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
print(f"SFT v16-tiny — 1000 curated, 3 epochs")
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
