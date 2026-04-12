"""GRPO v43 on Kaggle — v42 SFT adapter (THK binary + boxed weight) as base.

Uses:
- transformers 5.5.3 wheel from grpo-wheels-tf53-trl11 dataset
- Native NemotronH (Komil's 20x speed fix)
- v42 adapter from kernel output
- Full bf16 + LoRA + no gradient checkpointing
"""
import subprocess, sys, os, glob

# ============================================================
# 1. INSTALL GRPO WHEELS (override mayukh18's older transformers)
# ============================================================
print("Installing GRPO wheels...")
WHEEL_DIR = None
for candidate in [
    "/kaggle/input/grpo-wheels-tf53-trl11",
    "/kaggle/input/datasets/bharatmohan/grpo-wheels-tf53-trl11",
]:
    if os.path.exists(candidate):
        WHEEL_DIR = candidate
        break

if WHEEL_DIR:
    for whl in glob.glob(os.path.join(WHEEL_DIR, "*.whl")):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", whl], check=False)
        print(f"  Installed: {os.path.basename(whl)}")
else:
    print("WARNING: GRPO wheels not found")

# Also install from mayukh18 for base deps
PKG_DIR = None
for candidate in [
    "/kaggle/input/nemotron-packages/packages",
    "/kaggle/input/datasets/mayukh18/nemotron-packages/packages",
]:
    if os.path.exists(candidate):
        PKG_DIR = candidate
        break
if PKG_DIR:
    subprocess.run(f"pip install -q --no-index --find-links {PKG_DIR} peft datasets accelerate bitsandbytes", shell=True, check=False)

# ============================================================
# 2. IMPORTS (after wheel install)
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc, math, time, json, csv, re, random, shutil, zipfile
import torch

# Fix TRL 1.1.0 imports
import transformers.utils.hub as _hub
if not hasattr(_hub, 'TRANSFORMERS_CACHE'):
    try:
        from huggingface_hub import constants as _hf_c
        _hub.TRANSFORMERS_CACHE = getattr(_hf_c, 'HF_HUB_CACHE', '/tmp/hf_cache')
    except Exception:
        _hub.TRANSFORMERS_CACHE = '/tmp/hf_cache'
for _mod in ['llm_blender', 'mergekit', 'mergekit.config']:
    if _mod not in sys.modules:
        _fake = type(sys)(_mod)
        _fake.__spec__ = type(sys)(_mod)
        _fake.__path__ = []
        if '.' not in _mod:
            _fake.MergeConfiguration = None
        sys.modules[_mod] = _fake

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset as HFDataset
import kagglehub

print(f"transformers={__import__('transformers').__version__}, trl={__import__('trl').__version__}")

# ============================================================
# 3. CONFIG
# ============================================================
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
OUTPUT_DIR = "/kaggle/working"

# Find v42 adapter
ADAPTER_PATH = None
for pat in [
    "/kaggle/input/notebooks/bharatmohan/nemotron-sft-v42/nemotron-lora-adapter",
    "/kaggle/input/nemotron-sft-v42/nemotron-lora-adapter",
]:
    if os.path.exists(os.path.join(pat, "adapter_config.json")):
        ADAPTER_PATH = pat
        break
if not ADAPTER_PATH:
    # Walk to find it
    for root, dirs, files in os.walk("/kaggle/input"):
        if "adapter_config.json" in files and "adapter_model.safetensors" in files:
            ADAPTER_PATH = root
            break
print(f"Adapter: {ADAPTER_PATH}")

# Find train.csv
TRAIN_CSV = glob.glob("/kaggle/input/*/train.csv") + glob.glob("/kaggle/input/*/*/train.csv")
TRAIN_CSV = TRAIN_CSV[0] if TRAIN_CSV else None
print(f"Train CSV: {TRAIN_CSV}")

GRPO_STEPS = 200
GRPO_GENERATIONS = 4
GRPO_MAX_COMPLETION = 1024
GRPO_BATCH = 1
GRPO_GRAD_ACCUM = 4
GRPO_LR = 5e-6
GRPO_SAVE_STEPS = 25

# ============================================================
# 4. LOAD MODEL + ADAPTER
# ============================================================
# Strip custom code from Kaggle model input so transformers uses native NemotronH
import shutil as _sh
CLEAN_MODEL = "/kaggle/tmp/model_clean"
if not os.path.exists(CLEAN_MODEL):
    print("Copying model + stripping custom code for native NemotronH...")
    _sh.copytree(MODEL_PATH, CLEAN_MODEL, symlinks=True)
    # Remove ONLY modeling/config files (keep tokenizer parser!)
    for f in glob.glob(os.path.join(CLEAN_MODEL, "modeling_*.py")) + glob.glob(os.path.join(CLEAN_MODEL, "configuration_*.py")):
        os.remove(f)
        print(f"  Removed: {os.path.basename(f)}")
MODEL_PATH = CLEAN_MODEL

print("Loading model (native NemotronH, NO trust_remote_code)...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

if ADAPTER_PATH:
    print(f"Loading v42 adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=True)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
else:
    print("ERROR: No adapter found!")
    sys.exit(1)

print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ============================================================
# 5. PROMPTS + REWARDS
# ============================================================
SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
prompts = []
with open(TRAIN_CSV) as f:
    for row in csv.DictReader(f):
        prompts.append({"prompt": row["prompt"] + SUFFIX, "answer": row["answer"]})
random.seed(43)
random.shuffle(prompts)
prompts = prompts[:500]

grpo_data = [{"prompt": [{"role": "user", "content": p["prompt"]}], "answer": p["answer"]} for p in prompts]
dataset = HFDataset.from_list(grpo_data)
print(f"GRPO dataset: {len(dataset)} prompts")

def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else ""

def verify(stored, predicted):
    stored, predicted = str(stored).strip(), str(predicted).strip()
    if not predicted: return False
    if re.fullmatch(r'[01]+', stored): return predicted.lower() == stored.lower()
    try: return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except: return predicted.lower() == stored.lower()

def correctness_reward(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    return [2.0 if verify(a, extract_boxed(r)) else 0.0 for r, a in zip(responses, answer)]

def format_reward(completions, **kwargs):
    responses = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    return [0.5 if '\\boxed{' in r else 0.0 for r in responses]

# ============================================================
# 6. GRPO TRAINING
# ============================================================
gc.collect(); torch.cuda.empty_cache()

config = GRPOConfig(
    output_dir=os.path.join(OUTPUT_DIR, "grpo_output"),
    max_steps=GRPO_STEPS,
    per_device_train_batch_size=GRPO_BATCH,
    gradient_accumulation_steps=GRPO_GRAD_ACCUM,
    learning_rate=GRPO_LR,
    num_generations=GRPO_GENERATIONS,
    max_completion_length=GRPO_MAX_COMPLETION,
    temperature=0.9,
    beta=0.0,
    use_vllm=False,
    logging_steps=1,
    save_steps=GRPO_SAVE_STEPS,
    bf16=True,
    gradient_checkpointing=False,
    remove_unused_columns=False,
    report_to="none",
    seed=43,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[correctness_reward, format_reward],
    args=config,
    train_dataset=dataset,
)

print(f"\nGRPO v43 — v42 adapter + 200 steps + 1024 completion + native NemotronH")
start = time.time()
try:
    trainer.train()
    print(f"\nGRPO complete in {(time.time()-start)/60:.1f} min")
except (RuntimeError, torch.OutOfMemoryError) as e:
    print(f"\n!!! GRPO failed after {(time.time()-start)/60:.1f} min: {e}")
    gc.collect(); torch.cuda.empty_cache()

# ============================================================
# 7. SAVE & ZIP
# ============================================================
ADAPTER_OUT = os.path.join(OUTPUT_DIR, "nemotron-lora-adapter")
os.makedirs(ADAPTER_OUT, exist_ok=True)
try:
    model.save_pretrained(ADAPTER_OUT)
    tokenizer.save_pretrained(ADAPTER_OUT)
except Exception as e:
    print(f"Save failed: {e}, using checkpoint")
    ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "grpo_output/checkpoint-*")),
                   key=lambda p: int(p.rsplit("-",1)[-1]))
    if ckpts:
        for f in ["adapter_config.json","adapter_model.safetensors"]:
            src = os.path.join(ckpts[-1], f)
            if os.path.exists(src): shutil.copy(src, os.path.join(ADAPTER_OUT, f))

# Zip
zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ["adapter_config.json", "adapter_model.safetensors"]:
        fp = os.path.join(ADAPTER_OUT, f)
        if os.path.exists(fp):
            zf.write(fp, f)
print(f"submission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
print("DONE")
