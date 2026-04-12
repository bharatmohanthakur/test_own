"""GRPO v40b — Komil's native NemotronH (20x speed) + full bf16 + gradient checkpointing.

Combines:
- v31's proven GRPO structure (rewards, OOM recovery, checkpoint fallback)
- Komil's transformers 5.3+ fix (native NemotronH, 38 tok/s vs 2 tok/s)
- Full bf16 (NOT quantized — per user requirement)
- gradient_checkpointing=True (saves ~15GB VRAM vs False)
- LoRA on base model (no SFT adapter needed)
"""
import os, sys, gc, math, time, json, csv, re, random, glob, shutil, zipfile
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

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
        _fake.__spec__ = type(sys)(_mod)  # prevent find_spec ValueError
        _fake.__path__ = []
        if '.' not in _mod:
            _fake.MergeConfiguration = None
        sys.modules[_mod] = _fake

from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset as HFDataset

# ============================================================
MODEL_PATH = "/workspace/model"
ADAPTER_PATH = "/workspace/v38a_adapter"
TRAIN_CSV = "/workspace/data/train.csv"
OUTPUT_DIR = "/workspace/output_v41"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRPO_STEPS = 200  # Long run with 2048 completion length for real learning
GRPO_GENERATIONS = 4  # More generations = better reward signal estimation
GRPO_MAX_COMPLETION = 1024  # 2048 OOMs, 512 too short — 1024 is the sweet spot
GRPO_BATCH = 1
GRPO_GRAD_ACCUM = 4
GRPO_LR = 5e-6
GRPO_SAVE_STEPS = 10

# ============================================================
# LOAD MODEL — full bf16, native NemotronH (Komil's speed fix)
# ============================================================
print("Loading model — full bf16, native NemotronH (NO trust_remote_code)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print(f"Model loaded: {type(model).__name__}, VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ============================================================
# LOAD v38a SFT ADAPTER (0.72 Kaggle) or ADD fresh LoRA
# ============================================================
if os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
    print(f"Loading v38a adapter from {ADAPTER_PATH}...")
    from peft import PeftModel as PM
    model = PM.from_pretrained(model, ADAPTER_PATH, is_trainable=True)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print("v38a adapter loaded — GRPO on top of 0.72 SFT base")
else:
    print("No adapter found, adding fresh LoRA")
    lora_config = LoraConfig(
        r=32, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj","in_proj","out_proj",
                         "up_proj","down_proj","gate_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
print(f"After LoRA: VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ============================================================
# PROMPTS + REWARDS
# ============================================================
SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
prompts = []
with open(TRAIN_CSV) as f:
    for row in csv.DictReader(f):
        prompts.append({"prompt": row["prompt"] + SUFFIX, "answer": row["answer"]})
random.seed(40)
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
# GRPO TRAINING
# ============================================================
gc.collect(); torch.cuda.empty_cache()
print(f"Pre-training VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    max_steps=GRPO_STEPS,
    per_device_train_batch_size=GRPO_BATCH,
    gradient_accumulation_steps=GRPO_GRAD_ACCUM,
    learning_rate=GRPO_LR,
    num_generations=GRPO_GENERATIONS,
    max_completion_length=GRPO_MAX_COMPLETION,
    # max_prompt_length removed in TRL 1.1
    temperature=0.9,
    beta=0.0,  # No KL — v38a has MoE expert LoRA that conflicts with ref adapter
    use_vllm=False,
    logging_steps=1,
    save_steps=GRPO_SAVE_STEPS,
    bf16=True,
    gradient_checkpointing=False,  # Native NemotronH doesn't support it; LoRA keeps optimizer small
    remove_unused_columns=False,
    report_to="none",
    seed=42,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[correctness_reward, format_reward],
    args=config,
    train_dataset=dataset,
)

print(f"\n{'='*60}")
print(f"GRPO v40b — native NemotronH (Komil fix) + full bf16 + grad ckpt")
print(f"  Steps: {GRPO_STEPS}, Generations: {GRPO_GENERATIONS}")
print(f"  Max completion: {GRPO_MAX_COMPLETION}, Batch: {GRPO_BATCH}x{GRPO_GRAD_ACCUM}")
print(f"{'='*60}\n")

start = time.time()
try:
    trainer.train()
    print(f"\nGRPO complete in {(time.time()-start)/60:.1f} min")
except (RuntimeError, torch.OutOfMemoryError) as e:
    print(f"\n!!! GRPO failed after {(time.time()-start)/60:.1f} min: {e}")
    gc.collect(); torch.cuda.empty_cache()

# ============================================================
# SAVE
# ============================================================
try:
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}")
except Exception as e:
    print(f"Save failed: {e}")
    ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-",1)[-1]))
    if ckpts:
        print(f"Using checkpoint: {ckpts[-1]}")
        for f in ["adapter_config.json","adapter_model.safetensors"]:
            src = os.path.join(ckpts[-1], f)
            if os.path.exists(src): shutil.copy(src, os.path.join(OUTPUT_DIR, f))

print("DONE")
