"""
SFT v13 — TIR data + alpha=128 (Tina paper) + 4 LoRA modules (speed)
- 9,500 puzzle examples with computational CoT (Python code traces)
- 4 LoRA targets (in_proj, out_proj, up_proj, down_proj) — proven in v7
- alpha=128 (4x rank per Tina paper for stronger learning signal)
- 2 epochs, LR=2e-4, max_seq=2048
- Expected runtime: ~3.5 hours on RTX Pro 6000
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

# === Fix 3: Patch mamba_ssm __init__ to skip mamba3 import ===
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
        wandb.init(project="nemotron-reasoning", name="sft-v13-alpha128-4mod",
            config={"method": "SFT-TIR", "lora_rank": 32, "lora_alpha": 128,
                    "lr": 2e-4, "epochs": 2, "max_seq_len": 2048,
                    "data": "9500-puzzle-tir", "targets": "4-modules"},
            tags=["sft", "v13", "alpha128"],
            settings=wandb.Settings(init_timeout=120))
        WANDB_ACTIVE = True
        print("wandb online")
    except Exception as e2:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="nemotron-reasoning", name="sft-v13-alpha128-4mod",
            config={"method": "SFT-TIR"}, tags=["sft", "v13"])
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
LORA_ALPHA = 128       # 4x rank (Tina paper — stronger learning signal)
MAX_SEQ_LEN = 2048
NUM_EPOCHS = 2
LR = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 16

# ============================================================
# LOAD DATA — robust path finding
# ============================================================
print("Loading training data...")

# Try multiple possible dataset locations
jsonl_search_patterns = [
    "/kaggle/input/nemotron-training-v11/*.jsonl",
    "/kaggle/input/nemotron-training-v11/**/*.jsonl",
    "/kaggle/input/nemotron-training*/*.jsonl",
    "/kaggle/input/nemotron-training*/**/*.jsonl",
    "/kaggle/input/*training*/*.jsonl",
    "/kaggle/input/*/*.jsonl",
]

jsonl_paths = []
for pattern in jsonl_search_patterns:
    found = glob.glob(pattern, recursive=True)
    if found:
        jsonl_paths = found
        break

print(f"Found JSONL paths: {jsonl_paths}")

# Also list /kaggle/input/ for debugging
try:
    input_contents = os.listdir("/kaggle/input/")
    print(f"/kaggle/input/ contents: {input_contents}")
    for d in input_contents:
        dp = os.path.join("/kaggle/input/", d)
        if os.path.isdir(dp):
            print(f"  {d}/: {os.listdir(dp)[:10]}")
except Exception as e:
    print(f"Cannot list /kaggle/input/: {e}")

all_examples = []

# Prefer puzzle-only data (9500 examples)
if jsonl_paths:
    # Pick puzzle_data if available, else training_data
    puzzle_paths = [p for p in jsonl_paths if 'puzzle' in os.path.basename(p)]
    fpath = puzzle_paths[0] if puzzle_paths else jsonl_paths[0]
    print(f"Loading TIR dataset: {fpath}")
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                all_examples.append(json.loads(line))
    print(f"Loaded {len(all_examples)} TIR examples from JSONL")

if len(all_examples) < 100:
    print("TIR dataset not found or too small, generating from train.csv...")
    all_examples = []
    possible_paths = (
        glob.glob("/kaggle/input/*/train.csv") +
        glob.glob("/kaggle/input/*/*/train.csv") +
        glob.glob("/kaggle/input/*/*/*/train.csv")
    )
    if not possible_paths:
        raise FileNotFoundError("No train.csv found!")
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

    def parse_gravity(prompt):
        """Parse gravity puzzle and generate computational CoT."""
        pairs = re.findall(r't\s*=\s*([\d.]+)\s*(?:s|seconds?)?,?\s*(?:the\s+)?(?:fallen\s+)?distance\s*(?:=|is)\s*([\d.]+)', prompt, re.IGNORECASE)
        if not pairs:
            pairs = re.findall(r'distance\s*(?:=|is)\s*([\d.]+).*?t\s*=\s*([\d.]+)', prompt, re.IGNORECASE)
            pairs = [(t, d) for d, t in pairs]

        after = prompt.split('Now')[1] if 'Now' in prompt else prompt.split('?')[0] if '?' in prompt else prompt[-300:]
        tm = re.search(r't\s*=\s*([\d.]+)', after)
        target_t = float(tm.group(1)) if tm else None

        return pairs, target_t

    def make_gravity_cot(prompt, answer):
        pairs, target_t = parse_gravity(prompt)
        if pairs and target_t:
            lines = ["<think>", "Gravity problem: d = 0.5 * g * t^2, so g = 2d / t^2", "",
                     "```python", "# Calculate g from each example"]
            g_vals = []
            for t, d in pairs:
                g = 2 * float(d) / float(t)**2
                g_vals.append(g)
                lines.append(f"g = 2 * {d} / {t}**2 = {g:.6f}")
            avg_g = sum(g_vals) / len(g_vals)
            lines.append(f"\n# Average g = {avg_g:.6f}")
            result = 0.5 * avg_g * target_t**2
            lines.append(f"d = 0.5 * {avg_g:.6f} * {target_t}**2 = {result:.2f}")
            lines.append("```")
            lines.append(f"\nThe answer is {answer}.")
            lines.append("</think>")
            lines.append(f"\n\\boxed{{{answer}}}")
            return "\n".join(lines)
        return f"<think>\nGravity: d = 0.5*g*t^2. Computing g from examples, then applying.\nAnswer: {answer}\n</think>\n\n\\boxed{{{answer}}}"

    def make_cipher_cot(prompt, answer):
        return (f"<think>\nSubstitution cipher puzzle.\n\n```python\n"
                f"# Build mapping from encrypted->decrypted pairs\nmapping = {{}}\n"
                f"for enc, dec in example_pairs:\n    for e, d in zip(enc, dec):\n"
                f"        if e != ' ' and d != ' ':\n            mapping[e] = d\n\n"
                f"# Apply mapping to decrypt target\nresult = ''.join(mapping.get(c, c) for c in target)\n"
                f"# Result: {answer}\n```\n\nDecrypted text: {answer}\n</think>\n\n\\boxed{{{answer}}}")

    def make_numeral_cot(prompt, answer):
        return (f"<think>\nRoman numeral conversion.\n\n```python\n"
                f"roman_vals = {{'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}}\n"
                f"# Subtractive: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900\n\n"
                f"def to_decimal(s):\n    total = 0\n    for i, c in enumerate(s):\n"
                f"        val = roman_vals[c]\n        if i+1 < len(s) and val < roman_vals[s[i+1]]:\n"
                f"            total -= val\n        else:\n            total += val\n    return total\n\n"
                f"def to_roman(n):\n    result = ''\n    for val, sym in [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),\n"
                f"                       (100,'C'),(90,'XC'),(50,'L'),(40,'XL'),\n"
                f"                       (10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]:\n"
                f"        while n >= val:\n            result += sym\n            n -= val\n    return result\n\n"
                f"# Result: {answer}\n```\n</think>\n\n\\boxed{{{answer}}}")

    def make_unit_cot(prompt, answer):
        pairs = re.findall(r'(\d+(?:\.\d+)?)\s*\w+\s*(?:=|->|→|converts?\s+to|is)\s*(\d+(?:\.\d+)?)', prompt)
        if pairs:
            factors = [float(o)/float(i) for i, o in pairs if float(i) > 0]
            if factors:
                avg_f = sum(factors) / len(factors)
                lines = ["<think>", "Unit conversion. Computing conversion factor from examples.", "",
                         "```python", "# Calculate factor from each pair"]
                for i, o in pairs:
                    if float(i) > 0:
                        lines.append(f"factor = {o} / {i} = {float(o)/float(i):.6f}")
                lines.append(f"\n# Average factor = {avg_f:.6f}")
                lines.append(f"# Result: {answer}")
                lines.append("```")
                lines.append("</think>")
                lines.append(f"\n\\boxed{{{answer}}}")
                return "\n".join(lines)
        return f"<think>\nUnit conversion. Computing factor from example pairs.\nResult: {answer}\n</think>\n\n\\boxed{{{answer}}}"

    def make_bit_cot(prompt, answer):
        return (f"<think>\nBit manipulation puzzle.\n\n```python\n"
                f"# Parse input-output pairs and test operations\n"
                f"# Operations to test: XOR(mask), NOT, rotation, reverse, bit shift\n"
                f"pairs = [...]  # from problem\n\n"
                f"for xor_val in range(256):\n"
                f"    if all(int(inp, 2) ^ xor_val == int(out, 2) for inp, out in pairs):\n"
                f"        print(f'XOR with {{xor_val:08b}}')\n\n"
                f"# Also test NOT, left/right rotation, reverse\n"
                f"# Apply identified operation to target\n"
                f"# Result: {answer}\n```\n</think>\n\n\\boxed{{{answer}}}")

    def make_equation_cot(prompt, answer):
        return (f"<think>\nEquation/symbol transformation puzzle.\n\n```python\n"
                f"# Build character mapping from example pairs\nmapping = {{}}\n"
                f"for src, dst in example_pairs:\n"
                f"    for s, d in zip(src, dst):\n"
                f"        mapping[s] = d\n\n"
                f"# Apply mapping to transform target expression\n"
                f"result = ''.join(mapping.get(c, c) for c in target)\n"
                f"# Result: {answer}\n```\n</think>\n\n\\boxed{{{answer}}}")

    cot_makers = {
        'gravity': make_gravity_cot,
        'cipher': make_cipher_cot,
        'numeral': make_numeral_cot,
        'unit': make_unit_cot,
        'bit': make_bit_cot,
        'equation': make_equation_cot,
    }

    for row in rows:
        prompt_text, answer = row[1], row[2]
        cat = classify_puzzle(prompt_text)
        user_msg = prompt_text + "\nPlease put your final answer inside \\boxed{}."
        maker = cot_makers.get(cat, make_equation_cot)
        asst_msg = maker(prompt_text, answer)
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

# === Fix 4: Force slow path AFTER model load ===
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        mod.is_fast_path_available = False
        print(f"Patched {name}: is_fast_path_available = False")

# === Fix 5: Replace Triton rmsnorm_fn with pure PyTorch ===
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
# APPLY LoRA — 4 MODULES (proven fast, v7 scored 0.69)
# ============================================================
print("Setting up LoRA with 4 targets + alpha=128...")
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
print(f"SFT v13 — TIR + alpha=128 + 4 modules")
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
                eta_min = elapsed / max(global_step, 1) * (total_steps - global_step)
                print(f"  ep {epoch+1} | step {global_step}/{total_steps} | loss {avg:.4f} | lr {lr_now:.2e} | {elapsed:.1f}min | ETA {eta_min:.0f}min | mem {mem_gb:.1f}GB")
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
