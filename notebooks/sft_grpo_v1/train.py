"""
Nemotron Reasoning Challenge — SFT + GRPO (Phase 1+2)
RTX Pro 6000 Blackwell with all fixes.
SFT on 9500 competition examples with quality CoT, then GRPO.
"""

import sys, os, shutil, stat, gc, math, zipfile, time
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

import subprocess, csv, glob, json, random, re, statistics
import torch, torch.nn.functional as F

# === Fix: Patch mamba_ssm __init__ to skip mamba3 import (needs cutlass) ===
# The updated utility script imports mamba3 which requires cutlass/quack.
# We don't need Mamba3 — our model uses Mamba2. Patch the source file.
import glob as _glob
_init_paths = _glob.glob("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/mamba_ssm/__init__.py")
if _init_paths:
    _p = _init_paths[0]
    with open(_p) as _f:
        _src = _f.read()
    if 'mamba3' in _src:
        _src = _src.replace('from mamba_ssm.modules.mamba3 import Mamba3', '# mamba3 skipped (no cutlass)')
        _src = _src.replace('from mamba_ssm.modules.mamba3 import *', '# mamba3 skipped')
        with open(_p, 'w') as _f:
            _f.write(_src)
        print("Patched mamba_ssm __init__.py: skipped mamba3 import")

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
        wandb.init(project="nemotron-reasoning", name="sft-grpo-v1",
            config={"method": "SFT+GRPO", "lora_rank": 32, "lora_alpha": 64,
                    "sft_lr": 2e-4, "sft_epochs": 2, "grpo_lr": 5e-6,
                    "grpo_steps": 50, "max_seq_len": 2048},
            tags=["sft", "grpo", "kaggle"],
            settings=wandb.Settings(init_timeout=120))
        WANDB_ACTIVE = True
        print("wandb online")
    except Exception as e2:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="nemotron-reasoning", name="sft-grpo-v1",
            config={"method": "SFT+GRPO"}, tags=["sft", "grpo"])
        WANDB_ACTIVE = True
        print(f"wandb offline: {e2}")
except Exception as e:
    print(f"wandb unavailable: {e}")

# === Config ===
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
OUTPUT_DIR = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LORA_RANK = 32
LORA_ALPHA = 64
MAX_SEQ_LEN = 2048
SFT_EPOCHS = 2
SFT_LR = 2e-4
GRAD_ACCUM = 8
GRPO_STEPS = 50
GRPO_LR = 5e-6
GRPO_NUM_GENERATIONS = 2
GRPO_MAX_COMPLETION = 512
GRPO_NUM_PROMPTS = 300

# === Puzzle classifier ===
def classify_puzzle(p):
    p = p[:200].lower()
    if 'bit manipulation' in p: return 'bit'
    if 'encryption' in p or 'decrypt' in p: return 'cipher'
    if 'gravitational' in p: return 'gravity'
    if 'unit conversion' in p: return 'unit'
    if 'numeral' in p: return 'numeral'
    if 'transformation rules' in p or 'wonderland' in p: return 'equation'
    return 'unknown'

# === Load data ===
print("Loading training data...")
possible_paths = glob.glob("/kaggle/input/*/train.csv") + glob.glob("/kaggle/input/*/*/train.csv") + glob.glob("/kaggle/input/*/*/*/train.csv")
data_path = possible_paths[0]
print(f"Using: {data_path}")

all_rows = []
with open(data_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    all_rows = list(reader)
print(f"Loaded {len(all_rows)} total examples")

# === Load model ===
print("Loading Nemotron-3-Nano-30B...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
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
    if upcast: x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()
    if bias is not None: out = out + bias.float()
    if z is not None: out = out * F.silu(z.float())
    return out.to(dtype)

for name, mod in list(sys.modules.items()):
    if hasattr(mod, 'rmsnorm_fn'):
        mod.rmsnorm_fn = _pure_rmsnorm_fn
        print(f"Patched rmsnorm_fn in {name}")

# === Apply LoRA ===
lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj|gate_proj)$",
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Fix 5: Gradient checkpointing
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.enable_input_require_grads()

# ============================================================
# PHASE 1: SFT
# ============================================================
print("\n" + "="*60)
print("PHASE 1: SFT TRAINING")
print("="*60)

# === Quality CoT generation functions ===
def gravity_cot(prompt, answer):
    pairs = re.findall(r't\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)', prompt)
    target = re.search(r'for\s+t\s*=\s*([\d.]+)s?\s+given', prompt)
    cot = "<think>\nFalling distance problem: d = 0.5*g*t^2, so g = 2d/t^2.\n\n"
    gs = []
    for t_str, d_str in pairs:
        t, d = float(t_str), float(d_str)
        g = 2*d/(t*t); gs.append(g)
        cot += f"t={t_str}, d={d_str}: g = 2*{d_str}/{t_str}^2 = {g:.4f}\n"
    if gs:
        g_avg = statistics.mean(gs)
        cot += f"\nAverage g = {g_avg:.4f}\n"
        if target:
            tv = float(target.group(1))
            result = 0.5*g_avg*tv*tv
            cot += f"\nd = 0.5 * {g_avg:.4f} * {tv}^2 = {result:.2f}\n"
    return cot + f"</think>\n\n\\boxed{{{answer}}}"

def unit_cot(prompt, answer):
    pairs = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    target = re.search(r'convert the following measurement:\s*([\d.]+)', prompt)
    cot = "<think>\nUnit conversion: output = factor * input.\n\n"
    fs = []
    for i_s, o_s in pairs:
        f = float(o_s)/float(i_s); fs.append(f)
        cot += f"{i_s}m -> {o_s}: factor = {f:.6f}\n"
    if fs:
        fa = statistics.mean(fs)
        cot += f"\nConversion factor = {fa:.6f}\n"
        if target:
            t = float(target.group(1))
            cot += f"\n{t} * {fa:.6f} = {fa*t:.2f}\n"
    return cot + f"</think>\n\n\\boxed{{{answer}}}"

def numeral_cot(prompt, answer):
    target = re.search(r'write the number\s+(\d+)', prompt)
    cot = "<think>\nRoman numeral conversion.\nI=1, V=5, X=10, L=50, C=100, D=500, M=1000\nSubtractive: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900\n\n"
    if target:
        num = int(target.group(1))
        vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
                (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
        rem, parts = num, []
        for v,s in vals:
            while rem >= v: parts.append(s); rem -= v
        cot += f"{num} = {''.join(parts)}\n"
    return cot + f"</think>\n\n\\boxed{{{answer}}}"

def cipher_cot(prompt, answer):
    cot = "<think>\nSubstitution cipher decryption.\n\nBuilding letter mapping from examples:\n"
    mapping = {}
    for line in prompt.split('\n'):
        if '->' in line:
            parts = line.split('->')
            if len(parts) == 2:
                ew, dw = parts[0].strip().split(), parts[1].strip().split()
                if len(ew) == len(dw):
                    for e,d in zip(ew,dw):
                        if len(e)==len(d):
                            for ec,dc in zip(e,d):
                                if ec not in mapping:
                                    mapping[ec] = dc
                                    cot += f"  '{ec}' -> '{dc}'\n"
    tm = re.search(r'decrypt the following text:\s*(.*)', prompt, re.IGNORECASE)
    if tm:
        ct = tm.group(1).strip()
        cot += f"\nApplying mapping to '{ct}':\n"
        result = ''.join(mapping.get(c,c) for c in ct)
        cot += f"Result: {result}\n"
    return cot + f"</think>\n\n\\boxed{{{answer}}}"

def bit_cot(prompt, answer):
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    cot = "<think>\nBit manipulation puzzle. Analyzing the transformation pattern:\n\n"
    exs = [(int(i,2),int(o,2)) for i,o in pairs]

    found = None
    # Test XOR
    if len(exs) >= 2:
        xv = exs[0][0] ^ exs[0][1]
        if all(o == (i ^ xv) for i,o in exs):
            found = f"XOR with {format(xv,'08b')}"
    # Test NOT
    if not found and all(o == (~i & 0xFF) for i,o in exs):
        found = "bitwise NOT"
    # Test bit reverse
    br = lambda n: int(format(n,'08b')[::-1], 2)
    if not found and all(o == br(i) for i,o in exs):
        found = "bit reversal"
    # Test rotations
    if not found:
        for k in range(1, 8):
            rl = lambda n, k=k: ((n << k) | (n >> (8-k))) & 0xFF
            if all(o == rl(i) for i,o in exs):
                found = f"rotate left by {k}"
                break
            rr = lambda n, k=k: ((n >> k) | (n << (8-k))) & 0xFF
            if all(o == rr(i) for i,o in exs):
                found = f"rotate right by {k}"
                break
    # Test NOT + XOR
    if not found:
        for xv in range(256):
            if all(o == ((~i & 0xFF) ^ xv) for i,o in exs):
                found = f"NOT then XOR with {format(xv,'08b')}"
                break
    # Test reverse + XOR
    if not found:
        for xv in range(256):
            if all(o == (br(i) ^ xv) for i,o in exs):
                found = f"bit reverse then XOR with {format(xv,'08b')}"
                break
    # Test shifts
    if not found:
        for k in range(1, 8):
            if all(o == ((i << k) & 0xFF) for i,o in exs):
                found = f"left shift by {k}"
                break
            if all(o == (i >> k) for i,o in exs):
                found = f"right shift by {k}"
                break

    for i_s, o_s in pairs[:3]:
        cot += f"  {i_s} -> {o_s}\n"
    cot += f"\nIdentified operation: {found or 'compound transformation'}\n"
    cot += f"Applied to input: result = {answer}\n"
    return cot + f"</think>\n\n\\boxed{{{answer}}}"

def equation_cot(prompt, answer):
    cot = "<think>\nSymbol/equation transformation puzzle.\n\nAnalyzing the mapping rules from examples:\n"
    mapping = {}
    for line in prompt.split('\n'):
        if '=' in line and 'determine' not in line.lower() and 'below' not in line.lower() and 'wonderland' not in line.lower():
            parts = line.split('=')
            if len(parts) == 2:
                l, r = parts[0].strip(), parts[1].strip()
                for i in range(min(len(l), len(r))):
                    if l[i] != r[i] and l[i] not in mapping:
                        mapping[l[i]] = r[i]
                        cot += f"  '{l[i]}' -> '{r[i]}'\n"
    cot += f"\nApplying transformation rules to get the result.\n"
    cot += f"Result: {answer}\n"
    return cot + f"</think>\n\n\\boxed{{{answer}}}"

COT_FNS = {
    'gravity': gravity_cot, 'unit': unit_cot, 'numeral': numeral_cot,
    'cipher': cipher_cot, 'bit': bit_cot, 'equation': equation_cot
}

def build_messages(prompt, answer):
    cat = classify_puzzle(prompt)
    user_msg = prompt + "\nPlease put your final answer inside \\boxed{}."
    try:
        gen = COT_FNS.get(cat)
        assistant_msg = gen(prompt, answer) if gen else f"<think>\nAnalyzing the pattern step by step.\n</think>\n\n\\boxed{{{answer}}}"
    except Exception:
        assistant_msg = f"<think>\nAnalyzing step by step.\n</think>\n\n\\boxed{{{answer}}}"
    return [{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_msg}]

# Build training texts with PROPER label masking
print("Building training data with proper label masking...")

class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.items = []
        skipped = 0
        for row in rows:
            prompt_text, answer = row[1], row[2]
            messages = build_messages(prompt_text, answer)

            # Tokenize user message only (to find where assistant starts)
            user_only = tokenizer.apply_chat_template(
                [messages[0]], tokenize=False, add_generation_prompt=True
            )
            user_ids = tokenizer(user_only, add_special_tokens=False)["input_ids"]
            user_len = len(user_ids)

            # Tokenize full conversation
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            enc = tokenizer(full_text, truncation=True, max_length=max_length,
                           padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)

            # Labels: mask user prompt tokens + padding with -100
            labels = ids.clone()
            labels[:user_len] = -100  # mask user prompt
            labels[mask == 0] = -100  # mask padding

            # Skip if answer is truncated (no boxed in trainable part)
            trainable_text = tokenizer.decode(ids[user_len:], skip_special_tokens=True)
            if 'boxed{' not in trainable_text:
                skipped += 1
                continue

            self.items.append({"input_ids": ids, "attention_mask": mask, "labels": labels})

        print(f"Tokenized {len(self.items)} examples (skipped {skipped} truncated)")

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

random.seed(42)
random.shuffle(all_rows)
dataset = SFTDataset(all_rows, tokenizer, MAX_SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# === SFT Training with cosine LR ===
model.train()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=SFT_LR, weight_decay=0.01
)
total_steps = (len(dataloader) * SFT_EPOCHS) // GRAD_ACCUM
warmup_steps = max(10, total_steps // 20)

def cosine_lr(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr)

print(f"\nSFT Training: {SFT_EPOCHS} epochs, {len(dataset)} examples, ~{total_steps} steps")
print(f"Warmup: {warmup_steps} steps, LR: {SFT_LR}")

global_step = 0
sft_start = time.time()

for epoch in range(SFT_EPOCHS):
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

            if global_step % 20 == 0:
                avg = running_loss / n_batches
                lr_now = scheduler.get_last_lr()[0]
                elapsed = (time.time() - sft_start) / 60
                print(f"  epoch {epoch+1} | step {global_step}/{total_steps} | loss {avg:.4f} | lr {lr_now:.2e} | {elapsed:.1f}min")
                if WANDB_ACTIVE:
                    wandb.log({"sft/loss": avg, "sft/lr": lr_now, "sft/step": global_step})

    # Handle remaining gradients
    if n_batches % GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    avg_loss = running_loss / max(n_batches, 1)
    print(f"Epoch {epoch+1}/{SFT_EPOCHS} done - avg loss: {avg_loss:.4f}")
    if WANDB_ACTIVE:
        wandb.log({"sft/epoch_loss": avg_loss, "epoch": epoch+1})

sft_time = (time.time() - sft_start) / 60
print(f"\nSFT complete in {sft_time:.1f} min")

# Save SFT checkpoint (backup before GRPO)
SFT_DIR = "/kaggle/working/sft_adapter"
os.makedirs(SFT_DIR, exist_ok=True)
model.save_pretrained(SFT_DIR)
print(f"SFT adapter saved to {SFT_DIR}")

# Also save as submission zip in case GRPO fails
zip_path_sft = "/kaggle/working/submission_sft.zip"
with zipfile.ZipFile(zip_path_sft, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(SFT_DIR):
        fpath = os.path.join(SFT_DIR, fname)
        if os.path.isfile(fpath):
            zf.write(fpath, fname)
print(f"SFT backup: submission_sft.zip ({os.path.getsize(zip_path_sft)/1024/1024:.1f} MB)")

# ============================================================
# PHASE 2: GRPO (Reinforcement Learning)
# ============================================================
print("\n" + "="*60)
print("PHASE 2: GRPO TRAINING")
print("="*60)

# Free memory
del dataset, dataloader, optimizer, scheduler
gc.collect()
torch.cuda.empty_cache()

# === Answer matching utilities ===
def normalize_answer(s):
    s = str(s).strip()
    try:
        return str(round(float(s), 2))
    except ValueError:
        return s.lower().strip()

def extract_boxed(text):
    idx = text.rfind('\\boxed{')
    if idx < 0:
        return None
    start = idx + len('\\boxed{')
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == '{': depth += 1
        elif text[pos] == '}': depth -= 1
        pos += 1
    if depth == 0:
        return text[start:pos-1].strip()
    return None

def check_answer(predicted, expected):
    if predicted is None:
        return False
    pred_n = normalize_answer(predicted)
    exp_n = normalize_answer(expected)
    if pred_n == exp_n:
        return True
    try:
        if abs(float(pred_n) - float(exp_n)) < 0.02:
            return True
    except (ValueError, TypeError):
        pass
    return False

# === Select GRPO prompts (skip numeral - already 100% zero-shot) ===
grpo_rows = []
by_type = {}
for r in all_rows:
    cat = classify_puzzle(r[1])
    if cat == 'numeral':
        continue
    by_type.setdefault(cat, []).append(r)

per_type = GRPO_NUM_PROMPTS // len(by_type)
for cat, items in by_type.items():
    sampled = random.sample(items, min(per_type, len(items)))
    grpo_rows.extend(sampled)
    print(f"  GRPO {cat}: {len(sampled)}")
random.shuffle(grpo_rows)
print(f"GRPO using {len(grpo_rows)} prompts, {GRPO_NUM_GENERATIONS} generations each")

# === Custom GRPO training loop ===
model.train()
grpo_optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=GRPO_LR, weight_decay=0.01
)

grpo_start = time.time()
total_correct = 0
total_generated = 0
step = 0

try:
    for prompt_idx in range(min(GRPO_STEPS, len(grpo_rows))):
        row = grpo_rows[prompt_idx]
        prompt_text, gt_answer = row[1], row[2]
        user_msg = prompt_text + "\nPlease put your final answer inside \\boxed{}."

        # Tokenize prompt
        chat_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        prompt_len = chat_input.shape[1]

        # Generate completions
        model.eval()
        completions = []
        rewards = []

        with torch.no_grad():
            for g in range(GRPO_NUM_GENERATIONS):
                try:
                    output_ids = model.generate(
                        chat_input,
                        max_new_tokens=GRPO_MAX_COMPLETION,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    completion_ids = output_ids[0, prompt_len:]
                    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                    completions.append((completion_ids, completion_text))

                    predicted = extract_boxed(completion_text)
                    correct = check_answer(predicted, gt_answer)
                    rewards.append(1.0 if correct else 0.0)
                    total_correct += int(correct)
                    total_generated += 1
                except Exception as e:
                    print(f"  Generation error: {e}")
                    completions.append((torch.tensor([]), ""))
                    rewards.append(0.0)
                    total_generated += 1

        # Skip if no completions or all same reward (no signal)
        if len(rewards) < 2 or len(set(rewards)) == 1:
            if (prompt_idx + 1) % 10 == 0:
                elapsed = (time.time() - grpo_start) / 60
                acc = total_correct / max(total_generated, 1) * 100
                print(f"  GRPO step {prompt_idx+1}/{GRPO_STEPS} | acc {acc:.1f}% | {elapsed:.1f}min (no signal)")
            continue

        # Compute advantages (group relative)
        mean_r = sum(rewards) / len(rewards)
        std_r = max((sum((r - mean_r)**2 for r in rewards) / len(rewards))**0.5, 1e-6)
        advantages = [(r - mean_r) / std_r for r in rewards]

        # REINFORCE loss on completions with advantage
        model.train()
        grpo_optimizer.zero_grad()
        total_loss = 0.0
        n_loss_terms = 0

        for (comp_ids, comp_text), adv in zip(completions, advantages):
            if len(comp_ids) == 0 or abs(adv) < 0.01:
                continue

            # Build full sequence for log-prob computation
            full_ids = torch.cat([chat_input.squeeze(0), comp_ids.to(model.device)])
            full_ids = full_ids.unsqueeze(0)

            # Create labels: mask prompt, only compute loss on completion
            labels = full_ids.clone()
            labels[0, :prompt_len] = -100

            outputs = model(input_ids=full_ids, labels=labels)

            # Scale loss by advantage
            loss_term = -adv * outputs.loss
            loss_term.backward()
            n_loss_terms += 1

        if n_loss_terms > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grpo_optimizer.step()
            step += 1

        if (prompt_idx + 1) % 10 == 0:
            elapsed = (time.time() - grpo_start) / 60
            acc = total_correct / max(total_generated, 1) * 100
            print(f"  GRPO step {prompt_idx+1}/{GRPO_STEPS} | acc {acc:.1f}% | optim_step {step} | {elapsed:.1f}min")
            if WANDB_ACTIVE:
                wandb.log({"grpo/accuracy": acc, "grpo/step": step, "grpo/prompt_idx": prompt_idx+1})

except Exception as grpo_err:
    print(f"\nGRPO ERROR: {grpo_err}")
    print("Falling back to SFT-only adapter")
    # Copy SFT adapter as final output
    for fname in os.listdir(SFT_DIR):
        shutil.copy2(os.path.join(SFT_DIR, fname), os.path.join(OUTPUT_DIR, fname))

grpo_time = (time.time() - grpo_start) / 60
final_acc = total_correct / max(total_generated, 1) * 100
print(f"\nGRPO complete in {grpo_time:.1f} min | accuracy: {final_acc:.1f}%")

# ============================================================
# SAVE & ZIP
# ============================================================
print("\n" + "="*60)
print("SAVING")
print("="*60)

model.save_pretrained(OUTPUT_DIR)

zip_path = "/kaggle/working/submission.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            zf.write(fpath, fname)

print(f"submission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
total_time = (time.time() - sft_start) / 60
print(f"Total time: {total_time:.1f} min (SFT: {sft_time:.1f} + GRPO: {grpo_time:.1f})")

if WANDB_ACTIVE:
    wandb.log({"total_time_min": total_time, "sft_time_min": sft_time, "grpo_time_min": grpo_time})
    wandb.finish()

print("DONE")
