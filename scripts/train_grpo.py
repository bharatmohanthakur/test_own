"""
GRPO on top of SFT v7 (0.69) adapter.
Uses TRL's GRPOTrainer with 200 distilled samples.
Binary reward: correct \boxed{} answer = 1.0, wrong = 0.0
"""

import os, sys, csv, json, random, re, subprocess, zipfile
import torch

# wandb
os.environ["WANDB_API_KEY"] = "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"
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"])

from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset as HFDataset

# Config
MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
ADAPTER_PATH = "/workspace/sft_adapter"  # v7 adapter uploaded here
NUM_SAMPLES = 200
NUM_GENERATIONS = 4  # group size for GRPO (must divide batch)
NUM_EPOCHS = 3
LR = 3e-6  # much lower for RL
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_COMPLETION_LENGTH = 4096
MAX_PROMPT_LENGTH = 2048

# =====================================================================
# Download competition data
# =====================================================================
print("Setting up data...")
subprocess.run([sys.executable, "-m", "pip", "install", "kaggle", "-q"], check=False)
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "w") as f:
    json.dump({"username": "bharatmohan", "key": "1f8a45c8e928ba8ef9929855d233a32f"}, f)
os.chmod("/root/.kaggle/kaggle.json", 0o600)
subprocess.run(["kaggle", "competitions", "download",
                 "nvidia-nemotron-model-reasoning-challenge", "-p", "/tmp/data"], check=False)
subprocess.run(["unzip", "-o", "/tmp/data/nvidia-nemotron-model-reasoning-challenge.zip",
                 "-d", "/tmp/data"], check=False)

# Load distilled data for prompts + ground truth
DISTILLED_PATH = "/workspace/distilled_cot_1000.jsonl"
all_examples = []
if os.path.exists(DISTILLED_PATH):
    with open(DISTILLED_PATH) as f:
        for line in f:
            d = json.loads(line)
            all_examples.append(d)
    print(f"Loaded {len(all_examples)} distilled examples")
else:
    # Fallback: load from train.csv
    with open("/tmp/data/train.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
    random.seed(42)
    random.shuffle(rows)
    for row in rows[:NUM_SAMPLES]:
        all_examples.append({
            "messages": [
                {"role": "user", "content": row[1] + "\nPlease put your final answer inside \\boxed{}."},
                {"role": "assistant", "content": f"\\boxed{{{row[2]}}}"}
            ],
            "ground_truth": row[2]
        })
    print(f"Loaded {len(all_examples)} examples from train.csv")

# Sample 200 for GRPO
random.seed(42)
samples = random.sample(all_examples, min(NUM_SAMPLES, len(all_examples)))
print(f"Using {len(samples)} samples for GRPO")

# Extract prompts and ground truth answers
prompts = []
ground_truths = []
for s in samples:
    user_msg = s["messages"][0]["content"]
    # Extract ground truth from assistant message
    asst_msg = s["messages"][1]["content"]
    gt_match = re.findall(r'\\boxed\{([^}]*)\}', asst_msg)
    gt = gt_match[-1] if gt_match else ""
    prompts.append(user_msg)
    ground_truths.append(gt)

# Create dataset with just prompts
dataset = HFDataset.from_dict({
    "prompt": prompts,
    "ground_truth": ground_truths,
})
print(f"Dataset: {len(dataset)} prompts")

# =====================================================================
# Reward function
# =====================================================================
def normalize_answer(s):
    s = str(s).strip()
    try:
        return round(float(s), 2)
    except:
        return s.lower().strip()

def reward_fn(completions, ground_truth=None, **kwargs):
    """Binary reward: 1.0 if \boxed{} answer matches ground truth, 0.0 otherwise."""
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        # Extract answer from completion
        text = completion if isinstance(completion, str) else completion[0]["content"]
        matches = re.findall(r'\\boxed\{([^}]*)\}', text)
        if matches:
            predicted = normalize_answer(matches[-1])
            expected = normalize_answer(gt)
            if predicted == expected:
                rewards.append(1.0)
            else:
                # Try numeric comparison
                try:
                    if abs(float(predicted) - float(expected)) < 0.01:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                except:
                    rewards.append(0.0)
        else:
            rewards.append(0.0)  # No \boxed{} found
    return rewards

# =====================================================================
# Load model + SFT adapter
# =====================================================================
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # left padding for generation
print("Base model loaded")

# Load SFT adapter if available
if os.path.exists(ADAPTER_PATH) and os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
    print(f"Loading SFT adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload()  # Merge SFT weights into base
    print("SFT adapter merged")
else:
    print("WARNING: No SFT adapter found, training from base model")

# Apply new LoRA for GRPO training
print("Setting up LoRA for GRPO...")
lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|up_proj|down_proj|gate_proj)$",
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.enable_input_require_grads()

# =====================================================================
# GRPO Training
# =====================================================================
training_args = GRPOConfig(
    output_dir="/workspace/grpo_output",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",
    run_name="grpo-v1-on-sft-v7",
    max_completion_length=MAX_COMPLETION_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
    num_generations=NUM_GENERATIONS,
    remove_unused_columns=False,
)

wandb.init(project="nemotron-reasoning", name="grpo-v1-on-sft-v7",
    config={"method": "GRPO", "base": "sft-v7-0.69", "lr": LR,
            "epochs": NUM_EPOCHS, "samples": NUM_SAMPLES,
            "num_generations": NUM_GENERATIONS, "batch_size": BATCH_SIZE,
            "gpu": "H100-NVL-94GB"},
    tags=["grpo", "v1", "on-sft-v7"])
print("wandb ONLINE")

# Format prompts for chat template
def format_prompt(example):
    messages = [{"role": "user", "content": example["prompt"]}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n"
    return {"prompt": text}

dataset = dataset.map(format_prompt)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_fn,
    processing_class=tokenizer,
)

print(f"Starting GRPO: {NUM_EPOCHS} epochs, {NUM_SAMPLES} samples, {NUM_GENERATIONS} generations/sample")
trainer.train()
print("GRPO training complete!")

# =====================================================================
# Save
# =====================================================================
OUTPUT_DIR = "/workspace/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)

zip_path = "/workspace/submission.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            zf.write(fpath, fname)
size_mb = os.path.getsize(zip_path) / 1024 / 1024
print(f"submission.zip: {size_mb:.1f} MB")
wandb.finish()
print("DONE")
