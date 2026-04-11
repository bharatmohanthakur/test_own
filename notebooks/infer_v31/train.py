"""
Inference debug for v31 GRPO adapter.
Loads Nemotron-3-Nano-30B + bharatmohan/nemotron-v31-grpo-adapter,
runs prediction on 6 test prompts (one of each puzzle type),
prints the actual model outputs vs expected answers.

Goal: see what GRPO actually produces — is it
  - getting the format right but the math wrong?
  - producing too-long traces and cutting off the \\boxed{}?
  - drifting from the v22 baseline behavior?
  - hallucinating?
This is the diagnosis we need before deciding what to fix in v32+.
"""
import os, sys, subprocess, glob, json, re, math, time, gc

# Critical patches first
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch._dynamo
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True
torch.compile = lambda model=None, *a, **k: (model if model is not None else (lambda f: f))
print("torch.compile disabled")

# Install from offline wheels — same as Kaggle Nemotron pattern
print("Installing packages...")
PKG_DIR = None
for c in [
    "/kaggle/input/datasets/mayukh18/nemotron-packages/packages",
    "/kaggle/input/nemotron-packages/packages",
    "/kaggle/input/mayukh18/nemotron-packages/packages",
]:
    if os.path.exists(c):
        PKG_DIR = c
        break

if PKG_DIR:
    subprocess.run(
        f"pip install -q --no-index --find-links {PKG_DIR} "
        f"unsloth trl peft transformers datasets accelerate bitsandbytes",
        shell=True
    )
    for pat in ["causal_conv1d*.whl", "mamba_ssm*.whl"]:
        wheels = sorted(glob.glob(os.path.join(PKG_DIR, "..", pat)) + glob.glob(os.path.join(PKG_DIR, pat)))
        if wheels:
            subprocess.run(f"pip install -q {wheels[-1]}", shell=True)
else:
    print("ERROR: nemotron-packages not found")
    sys.exit(1)

import kagglehub
from unsloth import FastLanguageModel
from peft import PeftModel

# ============================================================
# FIND v31 ADAPTER
# ============================================================
print("Finding v31 GRPO adapter...")
ADP = None
for pat in [
    "/kaggle/input/nemotron-v31-grpo-adapter",
    "/kaggle/input/datasets/bharatmohan/nemotron-v31-grpo-adapter",
    "/kaggle/input/*/nemotron-v31-grpo-adapter",
    "/kaggle/input/*",
    "/kaggle/input/*/*",
]:
    for d in glob.glob(pat):
        if os.path.isdir(d):
            cfg = os.path.join(d, "adapter_config.json")
            sft = os.path.join(d, "adapter_model.safetensors")
            if os.path.exists(cfg) and os.path.exists(sft):
                ADP = d
                break
    if ADP: break

if not ADP:
    print("ERROR: v31 adapter not found")
    sys.exit(1)
print(f"v31 adapter at: {ADP}")

# ============================================================
# LOAD MODEL + ADAPTER
# ============================================================
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
print(f"Model: {MODEL_PATH}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=8192,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    dtype=None,
    unsloth_force_compile=False,
    attn_implementation="eager",
)

# Patch Mamba fast path
for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name and hasattr(mod, "is_fast_path_available"):
        mod.is_fast_path_available = False

print(f"Loading v31 adapter from {ADP}...")
model = PeftModel.from_pretrained(model, ADP, is_trainable=False)
for module in model.modules():
    if hasattr(module, "is_fast_path_available"):
        module.is_fast_path_available = False

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("Model + adapter ready")

# Switch to inference mode
FastLanguageModel.for_inference(model)

# ============================================================
# TEST PROMPTS — one per category, with known answers
# ============================================================
SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# Real prompts from train.csv (we know the answers)
TESTS = [
    {
        "category": "binary",
        "expected": "10010111",
        "prompt": """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
01010001 -> 11011101
00001001 -> 01101101
00010101 -> 01010101
11111111 -> 10000001
10011101 -> 01000101
00111011 -> 00001001
10111101 -> 00000101
00100110 -> 10110011

Now, determine the output for: 00110100"""
    },
    {
        "category": "cipher",
        "expected": "cat imagines book",
        "prompt": """In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:
ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley
pqrsfv pqorzg wvgwpo trgbjo -> dragon dreams inside castle
gbcpovb tqorbog bxo zrswtrj pffq -> student creates the magical door
bxo sfjpov pqrsfv dfjjfig -> the golden dragon follows
nqwvtogg qorpg bxo zegboqwfcg gotqob -> princess reads the mysterious secret
Now, decrypt the following text: trb wzrswvog hffk"""
    },
    {
        "category": "gravity",
        "expected": "154.62",
        "prompt": """In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:
For t = 1.37s, distance = 14.92 m
For t = 4.27s, distance = 144.96 m
For t = 3.28s, distance = 85.54 m
For t = 3.67s, distance = 107.09 m
For t = 1.78s, distance = 25.19 m
Now, determine the falling distance for t = 4.41s given d = 0.5*g*t^2."""
    },
    {
        "category": "roman",
        "expected": "XXXVIII",
        "prompt": """In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:
11 -> XI
15 -> XV
94 -> XCIV
19 -> XIX
Now, write the number 38 in the Wonderland numeral system."""
    },
    {
        "category": "unit",
        "expected": "16.65",
        "prompt": """In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:
10.08 m becomes 6.69
17.83 m becomes 11.83
35.85 m becomes 23.79
17.06 m becomes 11.32
31.54 m becomes 20.93
Now, convert the following measurement: 25.09 m"""
    },
    {
        "category": "equation",
        "expected": "@&",
        "prompt": """In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
`!*[{ = '"[`
\\'*'> = ![@
\\'-!` = \\\\
`!*\\& = '@'{
Now, determine the result for: [[-!'"""
    },
]

# ============================================================
# RUN INFERENCE
# ============================================================
def verify(stored, predicted):
    stored = str(stored).strip()
    predicted = str(predicted).strip()
    if not predicted: return False
    if re.fullmatch(r'[01]+', stored):
        return predicted.lower() == stored.lower()
    try:
        return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except:
        return predicted.lower() == stored.lower()

def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else ""

print("\n" + "="*80)
print("INFERENCE OUTPUTS")
print("="*80)

for i, test in enumerate(TESTS):
    print(f"\n{'='*80}")
    print(f"TEST {i+1}: {test['category'].upper()}")
    print(f"Expected: {test['expected']!r}")
    print(f"{'='*80}")

    prompt_text = test['prompt'] + SUFFIX
    messages = [{"role": "user", "content": prompt_text}]
    chat = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    inputs = tokenizer(chat, return_tensors="pt").to("cuda")

    print(f"\n[generating ~7680 tokens...]")
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=7680,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - t0
    full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract just the assistant response (after the chat template's assistant marker)
    response = full
    for marker in ["<|im_start|>assistant", "assistant\n", "<assistant>"]:
        if marker in response:
            response = response.split(marker, 1)[-1]
            break

    boxed = extract_boxed(response)
    correct = verify(test['expected'], boxed)
    print(f"\nGen time: {elapsed:.1f}s, response length: {len(response)} chars")
    print(f"Boxed answer: {boxed!r}")
    print(f"CORRECT: {correct}")
    print(f"\n--- FULL RESPONSE (first 4000 chars) ---")
    print(response[:4000])
    print(f"--- END (truncated at 4000 chars, full was {len(response)}) ---")
    sys.stdout.flush()

    # Free memory
    del inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "="*80)
print("DONE")
print("="*80)
