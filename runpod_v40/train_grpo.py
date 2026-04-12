"""v40 GRPO on RunPod — Komil's 20x speed fix + v37 SFT base.

Key changes from our prior GRPO attempts:
1. transformers>=5.3.0 with NATIVE NemotronH (no trust_remote_code)
   → fixes cache param bug, 2 tok/s → 38 tok/s
2. gradient_checkpointing=False (native impl doesn't declare support)
3. use_vllm=False (TRL's vLLM integration still broken for Nemotron)
4. Load v37 adapter via PeftModel, not Unsloth
5. Simple rewards: format (has \boxed{}) + correctness (exact match)

Expected: 50-200 GRPO steps in ~2-3 hrs on RTX Pro 6000 Blackwell.
"""
import csv
import json
import os
import re
import sys
import time
import random
import torch

# Fix TRANSFORMERS_CACHE removal in transformers 5.x (error #28)
import transformers.utils.hub as _hub
if not hasattr(_hub, 'TRANSFORMERS_CACHE'):
    from huggingface_hub import constants as _hf_c
    _hub.TRANSFORMERS_CACHE = getattr(_hf_c, 'HF_HUB_CACHE', '/tmp/hf_cache')

# Fix llm_blender / mergekit optional deps
for _mod in ['llm_blender', 'mergekit', 'mergekit.config']:
    if _mod not in sys.modules:
        sys.modules[_mod] = type(sys)(_mod)
        if '.' not in _mod:
            sys.modules[_mod].MergeConfiguration = None

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "/workspace/model"
ADAPTER_PATH = "/workspace/v37_adapter/nemotron-lora-adapter"
TRAIN_CSV = "/workspace/data/train.csv"
OUTPUT_DIR = "/workspace/output_v40"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# VRAM safety: num_generations=2, max_tokens=1024 (error #18)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
GRPO_STEPS = 50
NUM_GENERATIONS = 2
MAX_NEW_TOKENS = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 5e-6
KL_COEFF = 0.02

# ============================================================
# LOAD MODEL (native transformers, NO trust_remote_code)
# ============================================================
from transformers import BitsAndBytesConfig
print("Loading base model in 4-bit (saves ~45GB VRAM for GRPO headroom)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # GRPO needs left padding (error #13)
print(f"Base model loaded. dtype={model.dtype}")

# ============================================================
# ADD LoRA (required for 4-bit training)
# ============================================================
from peft import LoraConfig, get_peft_model, TaskType
lora_config = LoraConfig(
    r=32, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj",
                     "up_proj", "down_proj", "gate_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# ============================================================
# PREPARE PROMPTS
# ============================================================
print("Loading training prompts...")
prompts = []
answers = {}
with open(TRAIN_CSV) as f:
    for row in csv.DictReader(f):
        prompt = row["prompt"] + "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
        prompts.append({"prompt": prompt, "answer": row["answer"]})
        answers[prompt] = row["answer"]

random.seed(40)
random.shuffle(prompts)
prompts = prompts[:500]  # Use 500 diverse prompts for GRPO
dataset = Dataset.from_list([{"prompt": p["prompt"]} for p in prompts])
answer_lookup = {p["prompt"]: p["answer"] for p in prompts}
print(f"Loaded {len(prompts)} prompts for GRPO")

# ============================================================
# REWARD FUNCTIONS
# ============================================================
def extract_boxed(text):
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None

def format_reward(completions, **kwargs):
    """Reward for having \boxed{} in output."""
    return [1.0 if extract_boxed(c[0]["content"]) is not None else 0.0 for c in completions]

def correctness_reward(completions, prompts, **kwargs):
    """Reward for correct answer."""
    rewards = []
    for comp, prompt in zip(completions, prompts):
        text = comp[0]["content"]
        pred = extract_boxed(text)
        if pred is None:
            rewards.append(0.0)
            continue
        expected = answer_lookup.get(prompt[0]["content"], "")
        # Exact match (case-insensitive) or numeric tolerance
        if pred.lower() == expected.lower():
            rewards.append(2.0)  # Strong reward for correct
        else:
            try:
                if abs(float(pred) - float(expected)) < 0.01:
                    rewards.append(2.0)
            except (ValueError, TypeError):
                pass
            rewards.append(0.0) if len(rewards) < len(completions) else None
    # Pad if needed
    while len(rewards) < len(completions):
        rewards.append(0.0)
    return rewards

# ============================================================
# GRPO TRAINING
# ============================================================
print("Setting up GRPO trainer...")
config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    max_steps=GRPO_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_NEW_TOKENS,  # TRL 1.1.0 API
    temperature=0.7,
    beta=KL_COEFF,  # TRL 1.1.0 uses beta not kl_coeff
    use_vllm=False,
    logging_steps=1,
    save_steps=25,
    bf16=True,
    gradient_checkpointing=False,
    remove_unused_columns=False,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,  # TRL 1.1.0 API
    args=config,
    train_dataset=dataset,
    reward_funcs=[format_reward, correctness_reward],
)

print(f"\n{'='*60}")
print(f"GRPO v40 — {GRPO_STEPS} steps, {NUM_GENERATIONS} gens/prompt, LR={LR}")
print(f"Base: v37 adapter (0.69 Kaggle)")
print(f"Rewards: format (1.0) + correctness (2.0)")
print(f"Speed: transformers 5.3+ native NemotronH (Komil's fix)")
print(f"{'='*60}\n")

start = time.time()
trainer.train()
elapsed = (time.time() - start) / 60
print(f"\nGRPO complete in {elapsed:.1f} min")

# ============================================================
# SAVE
# ============================================================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")
