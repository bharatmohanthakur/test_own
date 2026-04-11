"""
SFT v19 — SFTTrainer with offline trl wheel
1699 curated examples, 1 epoch, alpha=32, LR=1e-4
"""

import sys, os, shutil, stat, gc, math, zipfile, time, json, types
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

import subprocess, glob, random, re
import torch, torch.nn.functional as F

# === Fix 3: Patch mamba_ssm __init__ to skip mamba3 import ===
_util_dirs = glob.glob("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script")
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

# Stub mamba3 submodules
for _mod_name in [
    'mamba_ssm.modules.mamba3',
    'mamba_ssm.ops.cute',
    'mamba_ssm.ops.cute.mamba3',
    'mamba_ssm.ops.cute.mamba3.mamba3_step_fn',
]:
    sys.modules[_mod_name] = types.ModuleType(_mod_name)
sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None

import kagglehub, mamba_ssm
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Install trl from OFFLINE wheel (no internet on RTX Pro 6000) ===
try:
    import trl
    print(f"trl already installed: {trl.__version__}")
except ImportError:
    offline_path = "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages/"
    if os.path.exists(offline_path):
        subprocess.run(f"pip install --no-index --find-links={offline_path} trl", shell=True)
    else:
        # Broader search for trl wheel
        wheels = glob.glob("/kaggle/input/**/trl-*.whl", recursive=True)
        if wheels:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", wheels[0]], check=False)
        else:
            # Last resort: try pip (works if internet available)
            subprocess.run([sys.executable, "-m", "pip", "install", "trl", "-q"], check=False)
    import trl
    print(f"trl installed: {trl.__version__}")

from trl import SFTTrainer, SFTConfig
from datasets import Dataset as HFDataset

# === wandb ===
WANDB_ACTIVE = False
try:
    import wandb
    wandb_config = {"method": "SFT", "lora_rank": 32, "lora_alpha": 32,
                    "lr": 1e-4, "epochs": 1, "max_seq_len": 4096,
                    "data": "v18-1699-curated", "targets": "4-modules",
                    "trainer": "SFTTrainer"}
    wandb_tags = ["sft", "v19", "sfttrainer", "alpha32"]
    try:
        wandb.login(key="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", relogin=True)
        wandb.init(project="nemotron-reasoning", name="sft-v19",
            config=wandb_config, tags=wandb_tags,
            settings=wandb.Settings(init_timeout=120))
        WANDB_ACTIVE = True
        print(f"wandb online")
    except Exception:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="nemotron-reasoning", name="sft-v19",
            config=wandb_config, tags=wandb_tags,
            settings=wandb.Settings(init_timeout=30))
        WANDB_ACTIVE = True
        print("wandb offline (fallback)")
except Exception as e:
    print(f"wandb unavailable: {e}")

# ============================================================
# CONFIG
# ============================================================
OUTPUT_DIR = "/kaggle/working"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "sft_adapter")
os.makedirs(ADAPTER_DIR, exist_ok=True)

LORA_RANK = 32
LORA_ALPHA = 32
MAX_SEQ_LEN = 4096
NUM_EPOCHS = 1
LR = 1e-4
BATCH_SIZE = 1
GRAD_ACCUM = 8
RANDOM_SEED = 42

# ============================================================
# LOAD DATA
# ============================================================
print("Loading training data...")
all_examples = []

# Debug: list input files
print("Scanning /kaggle/input/...")
for root, dirs, files in os.walk("/kaggle/input/"):
    depth = root.replace("/kaggle/input/", "").count(os.sep)
    if depth > 3:
        dirs.clear()
        continue
    for f in files:
        if f.endswith(('.jsonl', '.csv', '.json', '.whl')):
            fpath = os.path.join(root, f)
            try:
                sz = os.path.getsize(fpath)
            except:
                sz = -1
            print(f"  {fpath} ({sz} bytes)")

# Comprehensive glob
jsonl_paths = []
for pat in [
    "/kaggle/input/nemotron-training-v18/training_v18.jsonl",
    "/kaggle/input/datasets/bharatmohan/nemotron-training-v18/training_v18.jsonl",
    "/kaggle/input/*/training_v18.jsonl",
    "/kaggle/input/*/*/training_v18.jsonl",
    "/kaggle/input/*/*/*/training_v18.jsonl",
    "/kaggle/input/*/*/*/*/training_v18.jsonl",
]:
    jsonl_paths.extend(glob.glob(pat, recursive=True))
jsonl_paths = list(dict.fromkeys(jsonl_paths))

if jsonl_paths:
    fpath = jsonl_paths[0]
    print(f"\nLoading v18 dataset: {fpath}")
    with open(fpath) as f:
        for line in f:
            all_examples.append(json.loads(line))
    print(f"Loaded {len(all_examples)} curated examples")
else:
    print("\n*** ERROR: training_v18.jsonl NOT FOUND ***")
    sys.exit(1)

if len(all_examples) < 100:
    print(f"ERROR: Only {len(all_examples)} examples. Expected ~1699.")
    sys.exit(1)

random.seed(RANDOM_SEED)
random.shuffle(all_examples)
print(f"Total training examples: {len(all_examples)}")

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
print(f"Model at: {MODEL_PATH}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Model loaded.")

# === Fix 4: Force slow path ===
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        mod.is_fast_path_available = False
        print(f"Patched {name}: is_fast_path_available = False")

# === Fix 5: Replace rmsnorm_fn ===
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
# APPLY LoRA
# ============================================================
lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.enable_input_require_grads()

# ============================================================
# PREPARE DATASET & TRAIN
# ============================================================
records = [{"messages": item["messages"]} for item in all_examples]
dataset = HFDataset.from_list(records)
print(f"SFT dataset: {len(dataset)} examples")

start_time = time.time()

training_args = SFTConfig(
    output_dir=os.path.join(OUTPUT_DIR, "sft_output"),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_length=MAX_SEQ_LEN,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=2,
    remove_unused_columns=False,
    seed=RANDOM_SEED,
    report_to="wandb" if WANDB_ACTIVE else "none",
    packing=False,
    weight_decay=0.01,
    max_grad_norm=1.0,
)

print(f"\n{'='*60}")
print(f"SFT v19 — SFTTrainer, alpha=32, LR=1e-4")
print(f"{'='*60}")
print(f"  Examples: {len(dataset)}")
print(f"  Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
print(f"  LR: {LR}, LoRA: r={LORA_RANK}, a={LORA_ALPHA}")
print(f"{'='*60}\n")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("Starting training...")
trainer.train()
elapsed = (time.time() - start_time) / 60
print(f"\nTraining complete in {elapsed:.1f} min")

# ============================================================
# SAVE & ZIP
# ============================================================
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

print(f"\nsubmission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
print(f"Total time: {elapsed:.1f} min")

if WANDB_ACTIVE:
    wandb.log({"total_time_min": elapsed})
    wandb.finish()

print("DONE")
