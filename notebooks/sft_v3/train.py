"""
Nemotron Reasoning Challenge — SFT v3 (All Blackwell fixes applied)
Based on johnnyhyland's working reference notebook.
"""

import sys, os, shutil, stat, gc
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

import csv, glob, json, random, re, statistics, subprocess
import torch, torch.nn.functional as F
import kagglehub, mamba_ssm
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# === wandb offline (RTX Pro 6000 has no internet — confirmed) ===
WANDB_ACTIVE = False
try:
    import wandb
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project="nemotron-reasoning", name="sft-v3-quality-cot",
        config={"lora_rank": 32, "lora_alpha": 8, "lr": 2e-4, "epochs": 1,
                "max_seq_len": 2048, "data": "9500 quality CoT", "baseline_score": 0.48},
        tags=["sft", "quality-cot", "blackwell"])
    WANDB_ACTIVE = True
    print("wandb offline — sync after: kaggle kernels output → wandb sync wandb/offline-run-*")
except Exception as e:
    print(f"wandb: {e}")

# === Config ===
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
OUTPUT_DIR = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LORA_RANK = 32
MAX_SEQ_LEN = 2048
NUM_EPOCHS = 3  # more epochs on fewer examples = better
GRAD_ACCUM = 8
LR = 2e-4
SAMPLE_SIZE = 2000  # quality > quantity

# === Puzzle classifier (needed for sampling + CoT) ===
def classify_puzzle(p):
    p = p[:120].lower()
    if 'bit manipulation' in p: return 'bit'
    if 'encryption' in p or 'decrypt' in p: return 'cipher'
    if 'gravitational' in p: return 'gravity'
    if 'unit conversion' in p: return 'unit'
    if 'numeral' in p: return 'numeral'
    if 'transformation rules' in p: return 'equation'
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

# Sample balanced across puzzle types for quality training
random.seed(42)
by_type = {}
for r in all_rows:
    cat = classify_puzzle(r[1])
    if cat not in by_type: by_type[cat] = []
    by_type[cat].append(r)

per_type = SAMPLE_SIZE // len(by_type)
rows = []
for cat, items in by_type.items():
    sampled = random.sample(items, min(per_type, len(items)))
    rows.extend(sampled)
    print(f"  {cat}: {len(sampled)} examples")
random.shuffle(rows)
print(f"Using {len(rows)} curated examples × {NUM_EPOCHS} epochs")

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

# === Apply LoRA ===
lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=8,
    target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|up_proj|down_proj|gate_proj)$",
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Fix 5: Gradient checkpointing with use_reentrant=False ===
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# === Build QUALITY CoT training data ===

def gravity_cot(prompt, answer):
    pairs = re.findall(r't\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)', prompt)
    target = re.search(r'for\s+t\s*=\s*([\d.]+)s?\s+given', prompt)
    cot = "<think>\nFalling distance problem: d = 0.5*g*t², so g = 2d/t².\n\n"
    gs = []
    for t_str, d_str in pairs:
        t, d = float(t_str), float(d_str)
        g = 2*d/(t*t); gs.append(g)
        cot += f"t={t_str}, d={d_str}: g = 2×{d_str}/{t_str}² = {g:.4f}\n"
    if gs:
        g_avg = statistics.mean(gs)
        cot += f"\ng ≈ {g_avg:.4f}\n"
        if target:
            tv = float(target.group(1))
            cot += f"\nd = 0.5 × {g_avg:.4f} × {tv}² = {0.5*g_avg*tv*tv:.2f}\n"
    return cot + f"</think>\n\n\\boxed{{{answer}}}"

def unit_cot(prompt, answer):
    pairs = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    target = re.search(r'convert the following measurement:\s*([\d.]+)', prompt)
    cot = "<think>\nUnit conversion: output = factor × input.\n\n"
    fs = []
    for i_s, o_s in pairs:
        f = float(o_s)/float(i_s); fs.append(f)
        cot += f"{i_s}m → {o_s}: factor = {f:.6f}\n"
    if fs:
        fa = statistics.mean(fs)
        cot += f"\nFactor = {fa:.6f}\n"
        if target:
            t = float(target.group(1))
            cot += f"{t} × {fa:.6f} = {fa*t:.2f}\n"
    return cot + f"</think>\n\n\\boxed{{{answer}}}"

def numeral_cot(prompt, answer):
    target = re.search(r'write the number\s+(\d+)', prompt)
    cot = "<think>\nRoman numeral conversion.\nI=1,V=5,X=10,L=50,C=100,D=500,M=1000\nSubtractive: IV=4,IX=9,XL=40,XC=90\n\n"
    if target:
        num = int(target.group(1))
        vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
        rem, parts = num, []
        for v,s in vals:
            while rem >= v: parts.append(s); rem -= v
        cot += f"{num} = {''.join(parts)}\n"
    return cot + f"</think>\n\n\\boxed{{{answer}}}"

def cipher_cot(prompt, answer):
    cot = "<think>\nSubstitution cipher. Building letter mapping:\n\n"
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
                                    cot += f"'{ec}'→'{dc}' "
                    cot += "\n"
    tm = re.search(r'decrypt the following text:\s*(.*)', prompt, re.IGNORECASE)
    if tm:
        ct = tm.group(1).strip()
        cot += f"\nDecrypt '{ct}':\n"
        cot += ''.join(mapping.get(c,c) if c!=' ' else ' ' for c in ct) + "\n"
    return cot + f"</think>\n\n\\boxed{{{answer}}}"

def bit_cot(prompt, answer):
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    cot = "<think>\nBit manipulation. Testing operations:\n\n"
    exs = [(int(i,2),int(o,2)) for i,o in pairs]
    found = None
    if len(exs)>=2:
        xv = exs[0][0]^exs[0][1]
        if all(o==(i^xv) for i,o in exs):
            found = f"XOR with {format(xv,'08b')}"
    if not found and all(o==(~i&0xFF) for i,o in exs): found = "NOT"
    br = lambda n: int(format(n,'08b')[::-1],2)
    if not found and all(o==br(i) for i,o in exs): found = "bit reverse"
    if not found:
        for k in range(1,8):
            rl = lambda n,k=k: ((n<<k)|(n>>(8-k)))&0xFF
            if all(o==rl(i) for i,o in exs): found=f"rotate left {k}"; break
    if not found:
        for xv in range(256):
            if all(o==((~i&0xFF)^xv) for i,o in exs): found=f"NOT then XOR {format(xv,'08b')}"; break
    if not found:
        for xv in range(256):
            if all(o==(br(i)^xv) for i,o in exs): found=f"reverse then XOR {format(xv,'08b')}"; break
    cot += f"Operation: {found or 'compound pattern'}\n"
    cot += f"Result: {answer}\n"
    return cot + f"</think>\n\n\\boxed{{{answer}}}"

def equation_cot(prompt, answer):
    cot = "<think>\nSymbol substitution. Mapping:\n"
    mapping = {}
    for line in prompt.split('\n'):
        if '=' in line and 'determine' not in line.lower() and 'below' not in line.lower() and 'wonderland' not in line.lower():
            parts = line.split('=')
            if len(parts)==2:
                l,r = parts[0].strip(), parts[1].strip()
                for i in range(min(len(l),len(r))):
                    if l[i] not in mapping: mapping[l[i]]=r[i]
    for k,v in list(mapping.items())[:10]: cot += f"'{k}'→'{v}' "
    cot += f"\nResult: {answer}\n"
    return cot + f"</think>\n\n\\boxed{{{answer}}}"

COT_FNS = {'gravity':gravity_cot,'unit':unit_cot,'numeral':numeral_cot,
           'cipher':cipher_cot,'bit':bit_cot,'equation':equation_cot}

def build_text(prompt, answer):
    cat = classify_puzzle(prompt)
    user_msg = prompt + "\nPlease put your final answer inside \\boxed{}."
    try:
        gen = COT_FNS.get(cat)
        assistant_msg = gen(prompt, answer) if gen else f"<think>\nAnalyzing pattern.\n</think>\n\n\\boxed{{{answer}}}"
    except:
        assistant_msg = f"<think>\nAnalyzing pattern.\n</think>\n\n\\boxed{{{answer}}}"
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_msg}],
            tokenize=False, add_generation_prompt=False
        )
    except:
        return f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"

texts = [build_text(row[1], row[2]) for row in rows]
print(f"Built {len(texts)} quality CoT training texts")

class SFTDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = []
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_length,
                           padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
            labels = ids.clone()
            labels[mask == 0] = -100
            self.encodings.append({"input_ids": ids, "attention_mask": mask, "labels": labels})
        print(f"Tokenized {len(self.encodings)} examples")

    def __len__(self): return len(self.encodings)
    def __getitem__(self, idx): return self.encodings[idx]

dataset = SFTDataset(texts, tokenizer, MAX_SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# === Train ===
model.train()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=0.01
)
total_steps = (len(dataloader) * NUM_EPOCHS) // GRAD_ACCUM
print(f"Training: {NUM_EPOCHS} epochs, ~{total_steps} optimizer steps, {len(dataset)} examples")

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    optimizer.zero_grad()
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        running_loss += outputs.loss.item()

        if (i + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step = (i + 1) // GRAD_ACCUM
            avg = running_loss / (i + 1)
            if step % 10 == 0:
                print(f"  epoch {epoch+1} | step {step}/{total_steps} | avg_loss {avg:.4f}")
                if WANDB_ACTIVE:
                    wandb.log({"train/loss": avg, "train/step": step})

    if (i + 1) % GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} — avg loss: {avg_loss:.4f}")
    if WANDB_ACTIVE:
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch+1})

# === Save & zip ===
import zipfile
model.save_pretrained(OUTPUT_DIR)
zip_path = "/kaggle/working/submission.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        zf.write(os.path.join(OUTPUT_DIR, fname), fname)
print(f"submission.zip created ({os.path.getsize(zip_path)/1024/1024:.1f} MB)")

if WANDB_ACTIVE:
    wandb.finish()
print("Done.")
