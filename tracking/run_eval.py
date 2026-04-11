"""Reusable batched vLLM evaluation runner (runs on a RunPod / RTX Pro 6000 pod).

Loads the Nemotron base model once, then runs batched generation with each LoRA adapter
passed on the command line. Saves full prompts (not previews) so the local analyzer
can re-verify every trace against ground truth.

Designed to be SCP'd to the pod and invoked with:
    python3 run_eval.py \
        --model /workspace/model \
        --train /workspace/data/train.csv \
        --adapters v31:/workspace/adapter_v31 v30:/workspace/adapter_v30 \
        --n-per-type 10 \
        --out /workspace/eval_YYYYMMDD.json

Setup matches Kaggle eval exactly:
    temperature=0.0, top_p=1.0, max_tokens=7680, max_model_len=8192
    FLASHINFER attention backend (Blackwell sm_120 compat)
"""
import argparse
import csv
import json
import os
import random
import re
import time
from collections import defaultdict, Counter

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'


def classify(p: str) -> str:
    fl = p.split('\n')[0].lower()
    if 'bit manipulation' in fl or '8-bit' in fl: return 'binary'
    if 'roman' in fl or 'numeral' in fl: return 'roman'
    if 'unit' in fl or 'measurement' in fl: return 'unit'
    if 'gravitational' in fl: return 'gravity'
    if 'encryption' in fl or 'cipher' in fl: return 'cipher'
    if 'transformation rules' in fl: return 'equation'
    return 'other'


def extract_boxed(text: str) -> str:
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else ""


def wrap_prompt(text: str) -> str:
    if not text.endswith(SUFFIX.strip()):
        text = text + SUFFIX
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def load_sampled_prompts(train_csv: str, n_per_type: int, seed: int = 42):
    with open(train_csv) as f:
        rows = list(csv.DictReader(f))
    by_type = defaultdict(list)
    for r in rows:
        t = classify(r['prompt'])
        if t == 'other':
            continue
        by_type[t].append(r)

    random.seed(seed)
    sampled = []
    for t, items in sorted(by_type.items()):
        random.shuffle(items)
        sampled.extend(items[:n_per_type])
    return sampled


def parse_adapter_arg(arg: str):
    """'name:path' → (name, path)"""
    name, path = arg.split(':', 1)
    return name.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to base model (Nemotron)")
    ap.add_argument("--train", required=True, help="path to train.csv")
    ap.add_argument("--adapters", nargs="+", required=True,
                    help="list of name:path pairs, e.g. v31:/workspace/adapter_v31")
    ap.add_argument("--n-per-type", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=7680)
    args = ap.parse_args()

    print("Importing vLLM...", flush=True)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapters = [parse_adapter_arg(a) for a in args.adapters]
    print(f"Adapters to evaluate: {[a[0] for a in adapters]}")

    # Load prompts
    sampled = load_sampled_prompts(args.train, args.n_per_type, args.seed)
    print(f"Sampled {len(sampled)} prompts, mix: {Counter(classify(r['prompt']) for r in sampled)}")

    prompts = [wrap_prompt(r['prompt']) for r in sampled]
    expected = [r['answer'] for r in sampled]
    types = [classify(r['prompt']) for r in sampled]

    # Load model once with all LoRAs enabled
    print("Loading model...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
        enable_lora=True,
        max_loras=max(2, len(adapters)),
        max_lora_rank=32,
        trust_remote_code=True,
        enforce_eager=True,
        attention_backend="FLASHINFER",
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # Run each adapter
    adapter_outputs = {}
    for i, (name, path) in enumerate(adapters):
        print(f"\n=== Running {name} ({path}) on {len(prompts)} prompts ===", flush=True)
        t0 = time.time()
        outputs = llm.generate(
            prompts, sp,
            lora_request=LoRARequest(name, i + 1, path),
        )
        print(f"{name} generated in {time.time()-t0:.1f}s")
        adapter_outputs[name] = [o.outputs[0].text for o in outputs]

    # Build results rows with FULL prompts (critical for local analyzer)
    rows_out = []
    for i, r in enumerate(sampled):
        row = {
            "type": types[i],
            "expected": expected[i],
            "prompt": r['prompt'],  # FULL prompt, not preview
        }
        for name, texts in adapter_outputs.items():
            row[f"{name}_text"] = texts[i]
            row[f"{name}_box"] = extract_boxed(texts[i])
        rows_out.append(row)

    # Metadata
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": len(sampled),
        "n_per_type": args.n_per_type,
        "seed": args.seed,
        "adapters": [{"name": n, "path": p} for n, p in adapters],
        "sampling": {
            "temperature": 0.0, "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
        },
        "attention_backend": "FLASHINFER",
    }

    dump = {"meta": meta, "rows": rows_out}
    with open(args.out, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nSaved {len(rows_out)} rows + metadata to {args.out}")
    print("DONE")


if __name__ == '__main__':
    main()
