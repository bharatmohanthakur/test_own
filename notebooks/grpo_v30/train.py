"""
GRPO v30 — Load v29 SFT adapter (Donald-templated SFT) + Donald per-step rewards
Carries forward all v28 fixes (chunked-log-softmax monkey-patch, Mamba is_fast_path patch).
NEW in v30:
  - Loads bharatmohan/nemotron-sft-v29 (Donald-templated SFT) instead of v22
  - 7 reward functions per Donald playbook:
    correctness (new strict metric), format, pipeline_step (per labeled stage),
    contamination penalty, thrash penalty, ver_honesty (tiered), champagne bonus
  - Training data: bharatmohan/nemotron-training-v29 (500 templated examples)
Test with 20 GRPO samples first (per testing_policy).
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

# GRPO config — DRAMATICALLY reduced for speed
# v28 ran 50 steps × 1024 tokens × 2 gens at ~8.5 min/step = 7+ hrs total
# v30 target: ~1.5 hrs total. Use shorter completion + fewer steps.
GRPO_LR = 5e-6
GRPO_STEPS = 20       # reduced from 50
GRPO_GENERATIONS = 2
GRPO_BATCH = 1
GRPO_GRAD_ACCUM = 4
GRPO_TEMP = 0.9
GRPO_MAX_COMPLETION = 512  # reduced from 1024 (Donald templated traces are shorter)
GRPO_MAX_PROMPT = 512

# ============================================================
# FIND V22 SFT ADAPTER
# ============================================================
print("Finding v29 SFT adapter...")
sft_adapter_path = None
for pattern in [
    "/kaggle/input/notebooks/bharatmohan/nemotron-sft-v29/nemotron-lora-adapter",
    "/kaggle/input/notebooks/*/nemotron-sft-v29/nemotron-lora-adapter",
    "/kaggle/input/nemotron-sft-v29/nemotron-lora-adapter",
    "/kaggle/input/nemotron-sft-v29/sft_adapter",
    "/kaggle/input/*/nemotron-lora-adapter",
    "/kaggle/input/*/*/nemotron-lora-adapter",
    "/kaggle/input/*/*/*/nemotron-lora-adapter",
    "/kaggle/input/*/sft_adapter",
    "/kaggle/input/nemotron-sft-v29",
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
    print("ERROR: v29 SFT adapter not found!")
    sys.exit(1)

print(f"Found v29 adapter at: {sft_adapter_path}")

# ============================================================
# LOAD GRPO DATA — use v29 templated dataset
# ============================================================
print("Loading GRPO data...")
grpo_examples = []
for pat in [
    "/kaggle/input/nemotron-training-v29/training_v29.jsonl",
    "/kaggle/input/datasets/bharatmohan/nemotron-training-v29/training_v29.jsonl",
    "/kaggle/input/*/training_v29.jsonl",
    "/kaggle/input/*/*/training_v29.jsonl",
    "/kaggle/input/*/*/*/training_v29.jsonl",
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

# Load v29 SFT adapter ON TOP of base model
print(f"Loading v29 SFT adapter from {sft_adapter_path}...")
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
# GRPO SETUP — Donald per-step rewards
# ============================================================

SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

def _verify(stored, predicted):
    """New strict metric (Apr 2026): binary as string, others as float."""
    stored = str(stored).strip()
    predicted = str(predicted).strip()
    if not predicted: return False
    if re.fullmatch(r'[01]+', stored):
        return predicted.lower() == stored.lower()
    try:
        return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored.lower()

def _extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else ""

def _get_text(c):
    return c[0]["content"] if isinstance(c, list) else c

def _detect_type(prompt_text):
    fl = prompt_text.split('\n')[0].lower() if prompt_text else ''
    if 'bit manipulation' in fl or '8-bit' in fl: return 'binary'
    if 'roman' in fl or 'numeral' in fl: return 'roman'
    if 'unit' in fl or 'measurement' in fl: return 'unit'
    if 'gravitational' in fl: return 'gravity'
    if 'encryption' in fl or 'cipher' in fl: return 'cipher'
    if 'transformation rules' in fl: return 'equation'
    return 'other'

PIPELINE_LABELS = {
    'binary':   ['SCAN', 'VER', 'CONCAT', 'ANS'],
    'cipher':   ['LEN:', 'TABLE', 'VER', 'DECRYPT', 'CHECK', 'ANS'],
    'gravity':  ['SOLVE', 'RATE', 'VER', 'APPLY', 'ANS'],
    'unit':     ['SOLVE', 'RATE', 'VER', 'APPLY', 'ANS'],
    'roman':    ['DECOMPOSE', 'CAT', 'VER', 'ANS'],
    'equation': ['PARSE', 'SCAN', 'LOCK', 'VER', 'APPLY', 'ANS'],
}

CONTAMINATION = {
    'binary':   ['LEN:', 'TABLE:', 'DECRYPT:', 'VOCAB', 'roman', 'gravity', 'RATE', 'cipher'],
    'cipher':   ['B0:', 'XOR', 'AND', 'NOT', 'roman', 'RATE', 'gravity', 'CONST'],
    'gravity':  ['B0:', 'XOR', 'TABLE:', 'roman', 'cipher', 'binary', 'DECRYPT'],
    'unit':     ['B0:', 'XOR', 'TABLE:', 'roman', 'cipher', 'binary', 'DECRYPT'],
    'roman':    ['B0:', 'XOR', 'AND', 'TABLE:', 'cipher', 'gravity', 'RATE', 'DECRYPT'],
    'equation': ['LEN:', 'TABLE:', 'DECRYPT:', 'roman', 'gravity', 'RATE'],
}

THRASH = ['let me try', 'hmm', 'actually,', 'wait,', "i'm not sure", 'maybe', 'perhaps', 'i think i', 'or perhaps']

def _prompt_text(p):
    if isinstance(p, list):
        return p[0].get('content', '') if p else ''
    return p

def correctness_reward(prompts, completions, answer, **kwargs):
    return [2.0 if _verify(a, _extract_boxed(_get_text(c))) else 0.0 for c, a in zip(completions, answer)]

def format_reward(completions, **kwargs):
    out = []
    for c in completions:
        text = _get_text(c)
        s = 0.0
        if '\\boxed{' in text: s += 0.5
        if '<think>' in text and '</think>' in text: s += 0.3
        if '\\boxed{' in text and '<think>' in text: s += 0.2
        out.append(s)
    return out

def pipeline_step_reward(prompts, completions, answer, **kwargs):
    out = []
    for p, c in zip(prompts, completions):
        ptype = _detect_type(_prompt_text(p))
        labels = PIPELINE_LABELS.get(ptype, [])
        text = _get_text(c)
        present = sum(1 for l in labels if l in text)
        out.append(0.1 * present)
    return out

def contamination_penalty(prompts, completions, **kwargs):
    out = []
    for p, c in zip(prompts, completions):
        ptype = _detect_type(_prompt_text(p))
        markers = CONTAMINATION.get(ptype, [])
        text_lower = _get_text(c).lower()
        hits = sum(1 for m in markers if m.lower() in text_lower)
        out.append(-0.2 * hits)
    return out

def thrash_penalty(completions, **kwargs):
    out = []
    for c in completions:
        text_lower = _get_text(c).lower()
        hits = sum(1 for m in THRASH if m in text_lower)
        out.append(-0.15 * hits)
    return out

def ver_honesty_reward(prompts, completions, answer, **kwargs):
    out = []
    for c, a in zip(completions, answer):
        text = _get_text(c)
        boxed = _extract_boxed(text)
        is_correct = _verify(a, boxed)
        ver_pass = bool(re.search(r'VER.*?(PASS|YES)', text, re.IGNORECASE))
        ver_fail = bool(re.search(r'VER.*?(FAIL|NO)', text, re.IGNORECASE))
        if not (ver_pass or ver_fail):
            out.append(0.0)
        elif is_correct and ver_pass:
            out.append(0.5)   # honest YES
        elif (not is_correct) and ver_fail:
            out.append(0.2)   # honest NO
        elif is_correct and ver_fail:
            out.append(-0.1)  # confused NO
        else:
            out.append(-0.5)  # lying YES (worst)
    return out

def champagne_bonus(prompts, completions, answer, **kwargs):
    out = []
    for p, c, a in zip(prompts, completions, answer):
        prompt_text = _prompt_text(p)
        text = _get_text(c)
        text_lower = text.lower()
        boxed = _extract_boxed(text)
        is_correct = _verify(a, boxed)
        ptype = _detect_type(prompt_text)
        labels = PIPELINE_LABELS.get(ptype, [])
        all_steps = all(l in text for l in labels)
        no_thrash = not any(m in text_lower for m in THRASH)
        markers = CONTAMINATION.get(ptype, [])
        no_contam = not any(m.lower() in text_lower for m in markers)
        out.append(5.0 if (is_correct and all_steps and no_thrash and no_contam) else 0.0)
    return out

ALL_REWARDS = [
    correctness_reward,
    format_reward,
    pipeline_step_reward,
    contamination_penalty,
    thrash_penalty,
    ver_honesty_reward,
    champagne_bonus,
]
print(f"Donald reward functions registered: {[f.__name__ for f in ALL_REWARDS]}")

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
print(f"GRPO v30 — Donald-templated SFT loaded, GRPO with 7 Donald rewards")
print(f"{'='*60}")
print(f"  Prompts: {len(grpo_dataset)}, Steps: {GRPO_STEPS}")
print(f"  Generations: {GRPO_GENERATIONS}, Max completion: {GRPO_MAX_COMPLETION}")
print(f"{'='*60}\n")

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=ALL_REWARDS,
    args=training_args,
    train_dataset=grpo_dataset,
)

# ============================================================
# v28 FIX: monkey-patch chunked_hidden_states_selective_log_softmax
# Unsloth's text-only branch passes model().logits (vocab_size dim) into a function
# that expects hidden_states (hidden_size dim). Patch it to detect and fall through.
# ============================================================
def _v28_patch_unsloth_grpo():
    target_modules = []
    for name, mod in list(sys.modules.items()):
        if "UnslothGRPOTrainer" in name and hasattr(mod, "chunked_hidden_states_selective_log_softmax"):
            target_modules.append((name, mod))
    if not target_modules:
        print("v28 patch: no UnslothGRPOTrainer module found in sys.modules — skipping")
        return
    for name, mod in target_modules:
        original_hidden = mod.chunked_hidden_states_selective_log_softmax
        original_simple = getattr(mod, "chunked_selective_log_softmax", None)
        if original_simple is None:
            print(f"v28 patch: {name} has no chunked_selective_log_softmax fallback — skipping")
            continue

        def make_fixed(orig_hidden, orig_simple):
            def fixed_chunked_hidden_states_selective_log_softmax(
                hidden_states, lm_head, index,
                chunks=4,
                logit_scale_multiply=0.0,
                logit_scale_divide=0.0,
                logit_softcapping=0.0,
                temperature=1.0,
            ):
                # If shape[-1] is already vocab_size (matches lm_head.shape[0]), it's logits.
                # The original expects hidden_states where shape[-1] == lm_head.shape[1] (hidden_dim).
                hidden_last = hidden_states.shape[-1]
                lm_rows = lm_head.shape[0]
                lm_cols = lm_head.shape[1] if lm_head.ndim >= 2 else lm_rows
                if hidden_last == lm_rows and hidden_last != lm_cols:
                    # Already logits — apply scale/softcap/temperature manually then call simple path
                    logits = hidden_states
                    if logit_scale_multiply != 0.0:
                        logits = logits * logit_scale_multiply
                    if logit_scale_divide != 0.0:
                        logits = logits / logit_scale_divide
                    if logit_softcapping != 0.0:
                        logits = logits * torch.tanh(logits / logit_softcapping)
                    return orig_simple(logits, index, temperature)
                return orig_hidden(
                    hidden_states, lm_head, index,
                    chunks=chunks,
                    logit_scale_multiply=logit_scale_multiply,
                    logit_scale_divide=logit_scale_divide,
                    logit_softcapping=logit_softcapping,
                    temperature=temperature,
                )
            return fixed_chunked_hidden_states_selective_log_softmax

        mod.chunked_hidden_states_selective_log_softmax = make_fixed(original_hidden, original_simple)
        print(f"v28 patch applied to {name}: chunked_hidden_states_selective_log_softmax wrapped with shape guard")

_v28_patch_unsloth_grpo()

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
