"""
GRPO v27 — Load existing v22 SFT adapter, do GRPO on top
No SFT phase — saves 30+ min per run
Test with 20 GRPO samples first (per testing_policy)
"""

# CRITICAL: Disable torch.compile BEFORE any imports
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys, shutil, stat, gc, math, zipfile, time, json, types
import importlib, importlib.util
import subprocess, glob, random, re

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
    "/kaggle/input/mayukh18/nemotron-packages/packages",
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

import kagglehub
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset as HFDataset
from peft import PeftModel

# ============================================================
# CONFIG
# ============================================================
OUTPUT_DIR = "/kaggle/working"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "nemotron-lora-adapter")
os.makedirs(ADAPTER_DIR, exist_ok=True)

MAX_SEQ_LEN = 4096
RANDOM_SEED = 42

# GRPO config — reduced VRAM (v26 hit OOM with 4 gens × 2048 tokens)
GRPO_LR = 5e-6
GRPO_STEPS = 50
GRPO_GENERATIONS = 2   # reduced from 4
GRPO_BATCH = 1
GRPO_GRAD_ACCUM = 4
GRPO_TEMP = 0.9
GRPO_MAX_COMPLETION = 1024  # reduced from 2048
GRPO_MAX_PROMPT = 512

# ============================================================
# FIND V22 SFT ADAPTER
# ============================================================
print("Finding v22 SFT adapter...")
sft_adapter_path = None
for pattern in [
    "/kaggle/input/notebooks/bharatmohan/nemotron-sft-v22/nemotron-lora-adapter",
    "/kaggle/input/notebooks/*/nemotron-sft-v22/nemotron-lora-adapter",
    "/kaggle/input/nemotron-sft-v22/nemotron-lora-adapter",
    "/kaggle/input/nemotron-sft-v22/sft_adapter",
    "/kaggle/input/*/nemotron-lora-adapter",
    "/kaggle/input/*/*/nemotron-lora-adapter",
    "/kaggle/input/*/*/*/nemotron-lora-adapter",
    "/kaggle/input/*/sft_adapter",
    "/kaggle/input/nemotron-sft-v22",
]:
    for p in glob.glob(pattern):
        # Check for adapter files
        if os.path.exists(os.path.join(p, "adapter_model.safetensors")) or \
           os.path.exists(os.path.join(p, "adapter_config.json")):
            sft_adapter_path = p
            break
    if sft_adapter_path:
        break

# Debug: list input files
if not sft_adapter_path:
    print("Listing /kaggle/input/ contents:")
    for root, dirs, files in os.walk("/kaggle/input/"):
        depth = root.replace("/kaggle/input/", "").count(os.sep)
        if depth > 4:
            dirs.clear()
            continue
        for f in files:
            if 'adapter' in f or '.safetensors' in f:
                print(f"  {os.path.join(root, f)}")
    print("ERROR: v22 SFT adapter not found!")
    sys.exit(1)

print(f"Found v22 adapter at: {sft_adapter_path}")

# ============================================================
# LOAD GRPO DATA
# ============================================================
print("Loading GRPO data...")
grpo_examples = []
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
        print(f"GRPO data: {f} ({len(grpo_examples)})")
        break
    if grpo_examples:
        break

if not grpo_examples:
    print("ERROR: grpo_100.jsonl not found")
    sys.exit(1)

# v27 test: only 20 samples (per testing_policy)
random.seed(RANDOM_SEED)
random.shuffle(grpo_examples)
grpo_examples = grpo_examples[:20]
print(f"LIMITED: {len(grpo_examples)} GRPO samples for test")

# ============================================================
# LOAD BASE MODEL + V22 ADAPTER
# ============================================================
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
print(f"Base model: {MODEL_PATH}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    dtype=None,
    unsloth_force_compile=False,
    attn_implementation="eager",
)

# Patch Mamba fast_path
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        if hasattr(mod, 'is_fast_path_available'):
            mod.is_fast_path_available = False
            print(f"Patched {name}: is_fast_path_available = False")

# Load v22 SFT adapter ON TOP of base model
print(f"Loading v22 SFT adapter from {sft_adapter_path}...")
model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)
for module in model.modules():
    if hasattr(module, "is_fast_path_available"):
        module.is_fast_path_available = False
print("Patched is_fast_path_available = False on model modules")
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
model.print_trainable_parameters()
print("SFT adapter loaded. Now doing GRPO on top.")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # GRPO needs left padding

# ============================================================
# GRPO SETUP
# ============================================================

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

# Clean GPU before GRPO
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# GRPO TRAINING
# ============================================================
start_time = time.time()

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
    save_strategy="steps",
    per_device_train_batch_size=GRPO_BATCH,
    gradient_accumulation_steps=GRPO_GRAD_ACCUM,
    num_generations=GRPO_GENERATIONS,
    max_prompt_length=GRPO_MAX_PROMPT,
    max_completion_length=GRPO_MAX_COMPLETION,
    max_grad_norm=0.1,
    temperature=GRPO_TEMP,
    max_steps=GRPO_STEPS,
    save_steps=GRPO_STEPS,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    use_vllm=False,
    remove_unused_columns=False,
    seed=RANDOM_SEED,
    loss_type="grpo",
)

print(f"\n{'='*60}")
print(f"GRPO v27 — SFT adapter loaded, GRPO only")
print(f"{'='*60}")
print(f"  Prompts: {len(grpo_dataset)}, Steps: {GRPO_STEPS}")
print(f"  Generations: {GRPO_GENERATIONS}, Max completion: {GRPO_MAX_COMPLETION}")
print(f"{'='*60}\n")

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[correctness_reward, format_reward],
    args=training_args,
    train_dataset=grpo_dataset,
)

print("Starting GRPO...", flush=True)
trainer.train()
elapsed = (time.time() - start_time) / 60
print(f"\nGRPO complete in {elapsed:.1f} min")

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
print("DONE")
