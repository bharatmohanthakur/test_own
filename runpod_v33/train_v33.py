"""
GRPO v33 — Donald 7-reward GRPO from v22 SFT adapter
Diagnosis from v31 inference debug: model hallucinates on binary/equation
because it learned "I identify the pattern" (lying YES) under simple correctness reward.

v33 fix: rich per-step rewards that PUNISH hallucination and REWARD per-step work
- correctness (+2 if \\boxed{} matches)
- format (+0.5/0.3/0.2 for \\boxed{} + <think> tags)
- pipeline_step (+0.1 per labeled step like SCAN/VER/CONCAT/ANS)
- contamination penalty (-0.2 per cross-template marker)
- thrash penalty (-0.3 per marker, HEAVY — targets "I identify the pattern" exact phrase)
- VER honesty tiered (-0.5 for lying YES, +0.5 for honest YES)
- champagne bonus (+5 for fully-correct + clean trace)

Same proven config as v31: max_steps=10, save_steps=2, num_gen=2, max_completion=512.
Same v28 monkey-patch for chunked-log-softmax.
"""

# CRITICAL: disable torch.compile BEFORE any imports
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# wandb tracking
os.environ["WANDB_API_KEY"] = "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"
os.environ["WANDB_PROJECT"] = "nemotron-reasoning"
os.environ["WANDB_NAME"] = "v33-grpo-donald-7reward"
os.environ["WANDB_RUN_ID"] = "v33-grpo-donald-resume"  # consistent ID for resume
os.environ["WANDB_RESUME"] = "allow"

import sys, shutil, gc, math, zipfile, time, json, glob, random, re

import torch
import torch._dynamo
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True
_orig_compile = torch.compile
def _no_compile(model=None, *args, **kwargs):
    if model is None:
        return lambda f: f
    return model
torch.compile = _no_compile
print("torch.compile disabled globally")

# ============================================================
# IMPORTS
# ============================================================
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset as HFDataset
from peft import PeftModel

# ============================================================
# CONFIG
# ============================================================
WORKSPACE = "/workspace"
MODEL_PATH = os.path.join(WORKSPACE, "model")
ADAPTER_V22 = os.path.join(WORKSPACE, "adapter_v22")
DATA_PATH = os.path.join(WORKSPACE, "data", "grpo_goldilocks.jsonl")
OUTPUT_DIR = os.path.join(WORKSPACE, "output_v33")
ADAPTER_OUT = os.path.join(OUTPUT_DIR, "nemotron-lora-adapter")
GRPO_OUT = os.path.join(OUTPUT_DIR, "grpo_output")
os.makedirs(ADAPTER_OUT, exist_ok=True)
os.makedirs(GRPO_OUT, exist_ok=True)

MAX_SEQ_LEN = 4096
RANDOM_SEED = 42

# GRPO config — TIGHT (v28 OOMed at step 9 with 50 steps × 1024 completion)
GRPO_LR = 5e-6
GRPO_STEPS = 10
GRPO_GENERATIONS = 2
GRPO_BATCH = 1
GRPO_GRAD_ACCUM = 4
GRPO_TEMP = 0.9
GRPO_MAX_COMPLETION = 512
GRPO_MAX_PROMPT = 512
GRPO_SAVE_STEPS = 2

# ============================================================
# LOAD GOLDILOCKS DATA
# ============================================================
print(f"Loading GRPO data from {DATA_PATH} ...")
grpo_examples = []
with open(DATA_PATH) as fh:
    for line in fh:
        grpo_examples.append(json.loads(line))
print(f"Loaded {len(grpo_examples)} Goldilocks prompts")

import collections
print(f"  Categories: {dict(collections.Counter(e.get('category','?') for e in grpo_examples))}")

random.seed(RANDOM_SEED)
random.shuffle(grpo_examples)

# ============================================================
# LOAD BASE MODEL + V22 ADAPTER
# ============================================================
print(f"Loading base model from {MODEL_PATH} ...")
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

# Patch Mamba fast_path on imported modules
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name:
        if hasattr(mod, "is_fast_path_available"):
            mod.is_fast_path_available = False
            print(f"Patched {name}: is_fast_path_available = False")

# Load v22 SFT adapter on top of base model
print(f"Loading v22 SFT adapter from {ADAPTER_V22} ...")
model = PeftModel.from_pretrained(model, ADAPTER_V22, is_trainable=True)
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
# REWARDS — Donald Galliano playbook 7-reward function
# Designed to PUNISH "I identify the pattern" hallucination (v31 failure mode)
# Designed to REWARD per-step bit-serial / char-by-char work
# Max reward for perfect Donald-style trace: 9.0 (vs 2.5 for v31's correctness+format)
# ============================================================
SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

def _verify(stored, predicted):
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

def _prompt_text(p):
    if isinstance(p, list):
        return p[0].get('content', '') if p else ''
    return p

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

# CRITICAL: catch v31's exact failure modes (model says "I identify..." then guesses)
THRASH_MARKERS = [
    'let me try', 'hmm', 'actually,', 'wait,', "i'm not sure",
    'maybe', 'perhaps', 'i think i', 'or perhaps',
    # v31-specific anti-patterns:
    'i identify the pattern', 'i identify the rule',
    'after analyzing all',
    'by comparing inputs and outputs',
    'i can determine the transformation',
    'i can determine the rule',
    'a more complex multi-step',
    'applying the identified rule',
    'applying to target',
]

# REWARD 1: correctness (the main signal)
def correctness_reward(prompts, completions, answer, **kwargs):
    return [2.0 if _verify(a, _extract_boxed(_get_text(c))) else 0.0 for c, a in zip(completions, answer)]

# REWARD 2: format
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

# REWARD 3: per-step labeled pipeline
def pipeline_step_reward(prompts, completions, answer, **kwargs):
    out = []
    for p, c in zip(prompts, completions):
        ptype = _detect_type(_prompt_text(p))
        labels = PIPELINE_LABELS.get(ptype, [])
        text = _get_text(c)
        present = sum(1 for l in labels if l in text)
        out.append(0.1 * present)
    return out

# REWARD 4: contamination penalty
def contamination_penalty(prompts, completions, **kwargs):
    out = []
    for p, c in zip(prompts, completions):
        ptype = _detect_type(_prompt_text(p))
        markers = CONTAMINATION.get(ptype, [])
        text_lower = _get_text(c).lower()
        hits = sum(1 for m in markers if m.lower() in text_lower)
        out.append(-0.2 * hits)
    return out

# REWARD 5: thrash penalty (HEAVY weight to kill "I identify..." hallucination)
def thrash_penalty(completions, **kwargs):
    out = []
    for c in completions:
        text_lower = _get_text(c).lower()
        hits = sum(1 for m in THRASH_MARKERS if m in text_lower)
        out.append(-0.3 * hits)  # heavier than Donald's -0.15 because v31 showed this is the dominant failure
    return out

# REWARD 6: VER honesty (tiered)
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
            out.append(0.5)   # honest YES (best)
        elif (not is_correct) and ver_fail:
            out.append(0.2)   # honest NO
        elif is_correct and ver_fail:
            out.append(-0.1)  # confused NO
        else:
            out.append(-0.5)  # lying YES (worst)
    return out

# REWARD 7: champagne bonus
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
        no_thrash = not any(m in text_lower for m in THRASH_MARKERS)
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
print(f"Donald 7-reward registered: {[f.__name__ for f in ALL_REWARDS]}")
print(f"Heavy thrash penalty (-0.3 per marker) targets v31's 'I identify the pattern' failure mode")

# ============================================================
# BUILD GRPO DATASET
# ============================================================
grpo_data = []
for ex in grpo_examples:
    # Goldilocks data has {prompt, answer} (built by data/training_v31/grpo_goldilocks.jsonl)
    prompt_text = ex.get("prompt", "") if "prompt" in ex else ex["messages"][0]["content"]
    if not prompt_text.endswith(SUFFIX):
        prompt_text += SUFFIX
    grpo_data.append({
        "prompt": [{"role": "user", "content": prompt_text}],
        "answer": ex.get("answer", ""),
    })

grpo_dataset = HFDataset.from_list(grpo_data)
print(f"GRPO dataset: {len(grpo_dataset)} prompts")

gc.collect()
torch.cuda.empty_cache()

# ============================================================
# GRPO TRAINING ARGS
# ============================================================
training_args = GRPOConfig(
    output_dir=GRPO_OUT,
    learning_rate=GRPO_LR,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    save_strategy="steps",
    per_device_train_batch_size=GRPO_BATCH,
    gradient_accumulation_steps=GRPO_GRAD_ACCUM,
    num_generations=GRPO_GENERATIONS,
    max_prompt_length=GRPO_MAX_PROMPT,
    max_completion_length=GRPO_MAX_COMPLETION,
    max_grad_norm=0.1,
    temperature=GRPO_TEMP,
    max_steps=GRPO_STEPS,
    save_steps=GRPO_SAVE_STEPS,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",
    use_vllm=False,
    remove_unused_columns=False,
    seed=RANDOM_SEED,
    loss_type="grpo",
)

print(f"\n{'='*60}")
print(f"GRPO v33 (RunPod) — v22 adapter + Goldilocks + Donald 7-reward, max_steps={GRPO_STEPS}, save_steps={GRPO_SAVE_STEPS}")
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
# v28 FIX: chunked_hidden_states_selective_log_softmax shape guard
# ============================================================
def _v28_patch():
    targets = [(n, m) for n, m in list(sys.modules.items())
               if "UnslothGRPOTrainer" in n and hasattr(m, "chunked_hidden_states_selective_log_softmax")]
    if not targets:
        print("v28 patch: no UnslothGRPOTrainer module found in sys.modules — skipping")
        return
    for name, mod in targets:
        orig_hidden = mod.chunked_hidden_states_selective_log_softmax
        orig_simple = getattr(mod, "chunked_selective_log_softmax", None)
        if orig_simple is None:
            continue

        def make_fixed(orig_h, orig_s):
            def fixed(hidden_states, lm_head, index, chunks=4,
                      logit_scale_multiply=0.0, logit_scale_divide=0.0,
                      logit_softcapping=0.0, temperature=1.0):
                hidden_last = hidden_states.shape[-1]
                lm_rows = lm_head.shape[0]
                lm_cols = lm_head.shape[1] if lm_head.ndim >= 2 else lm_rows
                if hidden_last == lm_rows and hidden_last != lm_cols:
                    logits = hidden_states
                    if logit_scale_multiply != 0.0:
                        logits = logits * logit_scale_multiply
                    if logit_scale_divide != 0.0:
                        logits = logits / logit_scale_divide
                    if logit_softcapping != 0.0:
                        logits = logits * torch.tanh(logits / logit_softcapping)
                    return orig_s(logits, index, temperature)
                return orig_h(hidden_states, lm_head, index, chunks=chunks,
                              logit_scale_multiply=logit_scale_multiply,
                              logit_scale_divide=logit_scale_divide,
                              logit_softcapping=logit_softcapping,
                              temperature=temperature)
            return fixed

        mod.chunked_hidden_states_selective_log_softmax = make_fixed(orig_hidden, orig_simple)
        print(f"v28 patch applied to {name}")

_v28_patch()

# ============================================================
# TRAIN with OOM recovery — resume from checkpoint if present
# ============================================================
start_time = time.time()

# Find latest checkpoint to resume from (if v33 was killed mid-run)
ckpts = sorted(glob.glob(os.path.join(GRPO_OUT, "checkpoint-*")),
               key=lambda p: int(p.rsplit("-", 1)[-1]))
resume_ckpt = ckpts[-1] if ckpts else None
if resume_ckpt:
    print(f"Resuming from checkpoint: {resume_ckpt}")
else:
    print("No checkpoint found, starting fresh")

print("Starting GRPO...", flush=True)
try:
    if resume_ckpt:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()
    elapsed = (time.time() - start_time) / 60
    print(f"\nGRPO complete in {elapsed:.1f} min")
except (RuntimeError, torch.OutOfMemoryError) as e:
    elapsed = (time.time() - start_time) / 60
    print(f"\n!!! GRPO failed after {elapsed:.1f} min: {type(e).__name__}: {e}")
    print("Falling through to checkpoint recovery...")
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# SAVE & ZIP (best effort)
# ============================================================
try:
    model.save_pretrained(ADAPTER_OUT)
    tokenizer.save_pretrained(ADAPTER_OUT)
    print(f"Saved current model state to {ADAPTER_OUT}")
except Exception as e:
    print(f"In-memory save failed ({e}), falling back to latest checkpoint...")
    ckpts = sorted(glob.glob(os.path.join(GRPO_OUT, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[-1]))
    if ckpts:
        latest = ckpts[-1]
        print(f"Using latest checkpoint: {latest}")
        for fname in ["adapter_config.json", "adapter_model.safetensors"]:
            src = os.path.join(latest, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(ADAPTER_OUT, fname))
                print(f"  Copied {fname} from {latest}")
    else:
        print("ERROR: no checkpoint to fall back to")

# Patch adapter_config for inference
config_path = os.path.join(ADAPTER_OUT, "adapter_config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["inference_mode"] = True
    cfg["lora_dropout"] = 0.0
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

# Zip submission
zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in ["adapter_config.json", "adapter_model.safetensors"]:
        fpath = os.path.join(ADAPTER_OUT, fname)
        if os.path.exists(fpath):
            zf.write(fpath, fname)
            print(f"  Added {fname}")

if os.path.exists(zip_path):
    print(f"\nsubmission.zip: {os.path.getsize(zip_path)/1024/1024:.1f} MB")
print("DONE")
