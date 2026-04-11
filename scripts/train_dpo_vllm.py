"""
DPO on SFT v8 adapter using vLLM for fast generation.
Phase 1: vLLM batch inference to generate rejected responses (~2 min)
Phase 2: DPO training with TRL (~30 min)
"""

import os, sys, csv, json, random, re, subprocess, zipfile, gc
import torch

os.environ["WANDB_API_KEY"] = "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"

# =====================================================================
# Setup data
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

with open("/tmp/data/train.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)
    all_rows = list(reader)
print(f"Loaded {len(all_rows)} rows")

# Load perfect CoT for "chosen"
perfect_cot = {}
if os.path.exists("/workspace/perfect_cot_1000.jsonl"):
    with open("/workspace/perfect_cot_1000.jsonl") as f:
        for line in f:
            d = json.loads(line)
            key = d["messages"][0]["content"][:100]
            perfect_cot[key] = d["messages"][1]["content"]
    print(f"Loaded {len(perfect_cot)} perfect CoT")

def classify_puzzle(p):
    p = p[:200].lower()
    if 'bit manipulation' in p: return 'bit'
    if 'encryption' in p or 'decrypt' in p: return 'cipher'
    if 'gravitational' in p: return 'gravity'
    if 'unit conversion' in p: return 'unit'
    if 'numeral' in p: return 'numeral'
    if 'transformation rules' in p or 'wonderland' in p: return 'equation'
    return 'unknown'

random.seed(42)
by_type = {}
for r in all_rows:
    cat = classify_puzzle(r[1])
    by_type.setdefault(cat, []).append(r)

NUM_SAMPLES = 200
per_type = NUM_SAMPLES // len(by_type)
rows = []
for cat, items in by_type.items():
    sampled = random.sample(items, min(per_type, len(items)))
    rows.extend(sampled)
    print(f"  {cat}: {len(sampled)}")
random.shuffle(rows)

# =====================================================================
# Phase 1: vLLM batch inference
# =====================================================================
print("\n=== Phase 1: vLLM batch generation ===")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
ADAPTER_PATH = "/workspace/sft_adapter"

# Load vLLM with LoRA adapter
print("Loading vLLM with SFT adapter...")
llm = LLM(
    model=MODEL_PATH,
    enable_lora=True,
    max_lora_rank=32,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
)

lora_request = LoRARequest("sft_adapter", 1, ADAPTER_PATH)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
)

# Build prompts
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

prompts = []
ground_truths = []
user_messages = []
for row in rows:
    user_msg = row[1] + "\nPlease put your final answer inside \\boxed{}."
    messages = [{"role": "user", "content": user_msg}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = "<|im_start|>user\n" + user_msg + "<|im_end|>\n<|im_start|>assistant\n"
    prompts.append(text)
    ground_truths.append(row[2])
    user_messages.append(user_msg)

print(f"Generating {len(prompts)} predictions with vLLM...")
outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
print("Generation complete!")

# Free vLLM GPU memory
del llm
gc.collect()
torch.cuda.empty_cache()

# =====================================================================
# Build DPO pairs
# =====================================================================
print("\n=== Building DPO pairs ===")

def normalize_answer(s):
    s = str(s).strip()
    try:
        return str(round(float(s), 2))
    except ValueError:
        return s.lower().strip()

dpo_data = []
correct_count = 0
wrong_count = 0

for i, output in enumerate(outputs):
    generated = output.outputs[0].text
    gt = ground_truths[i]
    user_msg = user_messages[i]

    predicted_matches = re.findall(r'\\boxed\{([^}]*)\}', generated)
    predicted = predicted_matches[-1].strip() if predicted_matches else ""

    is_correct = normalize_answer(predicted) == normalize_answer(gt)

    if is_correct:
        correct_count += 1
    else:
        wrong_count += 1
        key = user_msg[:100]
        if key in perfect_cot:
            chosen = perfect_cot[key]
        else:
            chosen = "<think>\nAfter careful step-by-step analysis of the pattern from examples, I determine the answer.\n</think>\n\n\\boxed{" + gt + "}"

        dpo_data.append({
            "prompt": [{"role": "user", "content": user_msg}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": generated}],
        })

print(f"Results: {correct_count} correct, {wrong_count} wrong out of {len(outputs)}")
print(f"DPO pairs: {len(dpo_data)}")

if len(dpo_data) < 20:
    print("Few wrong predictions — augmenting with quality preference pairs...")
    for row in rows[:80]:
        user_msg = row[1] + "\nPlease put your final answer inside \\boxed{}."
        key = user_msg[:100]
        if key in perfect_cot:
            dpo_data.append({
                "prompt": [{"role": "user", "content": user_msg}],
                "chosen": [{"role": "assistant", "content": perfect_cot[key]}],
                "rejected": [{"role": "assistant", "content": "<think>\nThe answer is.\n</think>\n\n\\boxed{" + row[2] + "}"}],
            })
    print(f"Augmented to {len(dpo_data)} pairs")

with open("/workspace/dpo_pairs.jsonl", "w") as f:
    for item in dpo_data:
        f.write(json.dumps(item) + "\n")
print(f"Saved DPO pairs")

# =====================================================================
# Phase 2: DPO Training
# =====================================================================
print("\n=== Phase 2: DPO Training ===")
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"])

from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import Dataset as HFDataset

print("Loading model for DPO...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Merge SFT adapter
if os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload()
    print("SFT adapter merged")

dataset = HFDataset.from_list(dpo_data)

lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)

training_args = DPOConfig(
    output_dir="/workspace/dpo_output",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    beta=0.1,
    loss_type="sigmoid",
    warmup_steps=5,
    lr_scheduler_type="cosine",
    logging_steps=1,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",
    run_name="dpo-v1-on-sft-v8-vllm",
    remove_unused_columns=False,
    max_length=4096,
    max_prompt_length=2048,
)

wandb.init(project="nemotron-reasoning", name="dpo-v1-on-sft-v8-vllm",
    config={"method": "DPO", "base": "sft-v8", "lr": 5e-7,
            "beta": 0.1, "epochs": 2, "pairs": len(dpo_data),
            "correct": correct_count, "wrong": wrong_count},
    tags=["dpo", "vllm", "on-sft-v8"])

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

print(f"Starting DPO: 2 epochs, {len(dpo_data)} pairs")
trainer.train()
print("DPO complete!")

OUTPUT_DIR = "/workspace/dpo_adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)

zip_path = "/workspace/submission.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            zf.write(fpath, fname)
print(f"submission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
wandb.finish()
print("DONE")
