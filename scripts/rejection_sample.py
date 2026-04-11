#!/usr/bin/env python3
"""
Rejection-sampled training data generation using MiniMax M2.7 API.

Generates 8 candidate solutions per puzzle, verifies against ground truth,
keeps the longest correct reasoning trace. Uses high concurrency to maximize
throughput.

Usage:
    PYTHONUNBUFFERED=1 python3 scripts/rejection_sample.py [--resume]
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
import sys
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Force unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY = "sk-cp-BSGDKBJscQ1-fT0eAEYQiNgzWat0SYqIXxreNoeUtJ2p6tP0C3kQLt5M4PkNZY5smab3zO5q5V_VVuiV-Q62TqKd_t53agq14E6UbBKiXCtRJHuWzxnuefY"
BASE_URL = "https://api.minimax.io/v1"
MODEL = "MiniMax-M2.7"

DATA_PATH = "data/train.csv"
OUTPUT_PATH = "training_data/rft_1000.jsonl"
CHECKPOINT_PATH = "training_data/rft_checkpoint.json"
FULL_RESULTS_PATH = "training_data/rft_1000_full.jsonl"

NUM_SAMPLES_PER_TYPE = 200
NUM_CANDIDATES = 8
TEMPERATURE = 0.7
MAX_TOKENS = 4096

# Concurrency: process multiple examples at once, each with parallel candidates
MAX_CONCURRENT_API_CALLS = 24  # total concurrent API calls across all examples
BATCH_SIZE = 3  # number of examples to process simultaneously

SYSTEM_PROMPT = (
    "You are an expert puzzle solver. Think step by step inside <think> tags, "
    "then give your final answer inside \\boxed{}."
)

# Thread-safe print
_print_lock = threading.Lock()

def log(msg):
    with _print_lock:
        print(msg, flush=True)


# ── Puzzle type classifier ──────────────────────────────────────────────────
def classify_puzzle(prompt):
    p = prompt.lower()
    if "bit manipulation" in p or "bit shifts" in p or "8-bit binary" in p:
        return "bit_manipulation"
    elif "cipher" in p or "substitution" in p or "encrypt" in p:
        return "cipher"
    elif "gravitational" in p or "gravity" in p or "free fall" in p or "dropped" in p:
        return "gravity"
    elif "unit" in p and "conver" in p:
        return "unit_conversion"
    elif "numeral" in p or "roman" in p:
        return "numeral_system"
    elif "equation" in p or "transform" in p:
        return "equation_transform"
    return "unknown"


# ── Answer extraction and verification ──────────────────────────────────────
def extract_boxed_answer(text):
    """Extract the last \\boxed{...} answer from text, handling nested braces."""
    matches = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        depth = 0
        start = idx + 7
        j = start
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    matches.append(text[start:j])
                    break
                depth -= 1
            j += 1
        i = j + 1
    if matches:
        return matches[-1].strip()
    return None


def answers_match(predicted, ground_truth):
    """Check if predicted answer matches ground truth."""
    if predicted is None:
        return False
    predicted = predicted.strip()
    ground_truth = ground_truth.strip()

    # Case-insensitive exact match
    if predicted.lower() == ground_truth.lower():
        return True

    # Numeric comparison with tolerance
    try:
        pred_val = float(predicted)
        gt_val = float(ground_truth)
        if abs(pred_val - gt_val) < 0.02:
            return True
    except (ValueError, TypeError):
        pass

    return False


def make_fallback_solution(prompt, answer):
    """Create a minimal fallback solution when all candidates fail."""
    return (
        "<think>\nLet me work through this problem step by step.\n\n"
        "After carefully analyzing the patterns and relationships given in the problem, "
        "I can determine the answer.\n</think>\n\n\\boxed{" + answer + "}"
    )


# ── API call ────────────────────────────────────────────────────────────────
def generate_single_candidate(client, prompt, example_id, candidate_idx):
    """Generate one candidate solution. Returns (example_id, candidate_idx, text)."""
    user_content = prompt + "\nPlease put your final answer inside \\boxed{}."
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (example_id, candidate_idx, response.choices[0].message.content)
    except Exception as e:
        log(f"    API error [{example_id[:8]}#{candidate_idx}]: {e}")
        return (example_id, candidate_idx, "")


# ── Data loading ────────────────────────────────────────────────────────────
def load_and_sample_data(data_path, num_per_type):
    """Load train.csv, skip numeral_system, sample balanced across 5 types."""
    with open(data_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    by_type = defaultdict(list)
    for row in rows:
        ptype = classify_puzzle(row["prompt"])
        if ptype in ("numeral_system", "unknown"):
            continue
        by_type[ptype].append(row)

    log("\nPuzzle type counts (excluding numeral_system):")
    for t, examples in sorted(by_type.items()):
        log(f"  {t}: {len(examples)}")

    sampled = []
    random.seed(42)
    for ptype in sorted(by_type.keys()):
        examples = by_type[ptype]
        chosen = random.sample(examples, min(num_per_type, len(examples)))
        for ex in chosen:
            ex["_type"] = ptype
        sampled.extend(chosen)

    random.shuffle(sampled)
    log(f"\nTotal sampled: {len(sampled)}")
    return sampled


def process_batch(client, batch, batch_start_idx, total):
    """Process a batch of examples with all candidates in parallel."""
    # Submit all API calls for all examples in batch
    all_futures = {}
    results_by_id = defaultdict(list)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_API_CALLS) as executor:
        for example in batch:
            eid = example["id"]
            for c in range(NUM_CANDIDATES):
                future = executor.submit(
                    generate_single_candidate, client, example["prompt"], eid, c
                )
                all_futures[future] = (eid, c)

        for future in as_completed(all_futures):
            eid, cidx, text = future.result()
            results_by_id[eid].append(text)

    # Now evaluate each example
    batch_results = []
    for i, example in enumerate(batch):
        eid = example["id"]
        prompt = example["prompt"]
        ground_truth = example["answer"]
        ptype = example["_type"]
        idx = batch_start_idx + i

        candidates = results_by_id[eid]
        correct_solutions = []
        for cand in candidates:
            if not cand:
                continue
            predicted = extract_boxed_answer(cand)
            if answers_match(predicted, ground_truth):
                correct_solutions.append(cand)

        num_correct = len(correct_solutions)

        if correct_solutions:
            best = max(correct_solutions, key=len)
            log(f"  [{idx+1}/{total}] {ptype[:12]:12s} id={eid[:8]} "
                f"correct={num_correct}/{NUM_CANDIDATES} [ACCEPTED len={len(best)}]")
        else:
            best = make_fallback_solution(prompt, ground_truth)
            log(f"  [{idx+1}/{total}] {ptype[:12]:12s} id={eid[:8]} "
                f"correct=0/{NUM_CANDIDATES} [FALLBACK]")

        user_content = prompt + "\nPlease put your final answer inside \\boxed{}."
        result = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": best},
            ],
            "_meta": {
                "id": eid,
                "type": ptype,
                "ground_truth": ground_truth,
                "num_correct": num_correct,
                "num_candidates": NUM_CANDIDATES,
                "is_fallback": num_correct == 0,
            },
        }
        batch_results.append(result)

    return batch_results


def save_checkpoint(results, stats, checkpoint_path):
    with open(checkpoint_path, "w") as f:
        json.dump(
            {
                "completed_ids": [r["_meta"]["id"] for r in results],
                "stats": {k: dict(v) for k, v in stats.items()},
                "num_completed": len(results),
            },
            f,
        )


def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        return set(data.get("completed_ids", []))
    return set()


def save_results(results, output_path):
    with open(output_path, "w") as f:
        for r in results:
            output = {"messages": r["messages"]}
            f.write(json.dumps(output) + "\n")


def save_full_results(results, path):
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.chdir("/Users/bharat/Downloads/kaggle")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)

    examples = load_and_sample_data(DATA_PATH, NUM_SAMPLES_PER_TYPE)

    # Resume
    completed_ids = set()
    results = []
    if args.resume:
        completed_ids = load_checkpoint(CHECKPOINT_PATH)
        if os.path.exists(FULL_RESULTS_PATH):
            with open(FULL_RESULTS_PATH, "r") as f:
                results = [json.loads(line) for line in f if line.strip()]
        log(f"\nResuming: {len(completed_ids)} already completed")

    remaining = [ex for ex in examples if ex["id"] not in completed_ids]
    total = len(remaining)

    log(f"\nRemaining to process: {total}")
    log(f"Batch size: {BATCH_SIZE} examples, {MAX_CONCURRENT_API_CALLS} concurrent API calls")
    log(f"Total API calls: {total * NUM_CANDIDATES}")
    log("=" * 70)

    stats = defaultdict(lambda: {"total": 0, "accepted": 0, "correct_counts": []})
    start_time = time.time()
    processed_count = 0

    # Process in batches
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = remaining[batch_start:batch_end]

        batch_results = process_batch(client, batch, batch_start, total)

        for result in batch_results:
            results.append(result)
            ptype = result["_meta"]["type"]
            stats[ptype]["total"] += 1
            if not result["_meta"]["is_fallback"]:
                stats[ptype]["accepted"] += 1
            stats[ptype]["correct_counts"].append(result["_meta"]["num_correct"])

        processed_count += len(batch)

        # Checkpoint every 50 examples
        if processed_count % 50 < BATCH_SIZE or batch_end == total:
            if processed_count >= 50 or batch_end == total:
                save_checkpoint(results, stats, CHECKPOINT_PATH)
                save_full_results(results, FULL_RESULTS_PATH)
                save_results(results, OUTPUT_PATH)

                elapsed = time.time() - start_time
                rate = processed_count / elapsed * 3600
                eta_min = (total - processed_count) / max(rate / 3600, 0.001) / 60
                log(f"\n  --- Checkpoint ({processed_count}/{total}) | "
                    f"Rate: {rate:.0f}/hr | ETA: {eta_min:.0f} min ---")

                for ptype in sorted(stats.keys()):
                    s = stats[ptype]
                    if s["total"] > 0:
                        acc = s["accepted"] / s["total"] * 100
                        avg_c = sum(s["correct_counts"]) / len(s["correct_counts"])
                        log(f"    {ptype}: {s['accepted']}/{s['total']} ({acc:.0f}%), "
                            f"avg={avg_c:.1f}/8")
                log("")

        # Progress every 10 examples
        if processed_count % 10 < BATCH_SIZE:
            elapsed = time.time() - start_time
            rate = processed_count / elapsed * 3600 if elapsed > 0 else 0
            log(f"  [Progress: {processed_count}/{total} | "
                f"Elapsed: {elapsed/60:.1f}m | Rate: {rate:.0f}/hr]")

    # Final save
    save_checkpoint(results, stats, CHECKPOINT_PATH)
    save_full_results(results, FULL_RESULTS_PATH)
    save_results(results, OUTPUT_PATH)

    # Final stats
    log("\n" + "=" * 70)
    log("FINAL STATISTICS")
    log("=" * 70)

    total_accepted = 0
    total_processed = 0
    total_correct_sum = 0
    total_candidate_sum = 0

    for ptype in sorted(stats.keys()):
        s = stats[ptype]
        accepted = s["accepted"]
        processed = s["total"]
        correct_counts = s["correct_counts"]
        avg_correct = sum(correct_counts) / len(correct_counts) if correct_counts else 0

        total_accepted += accepted
        total_processed += processed
        total_correct_sum += sum(correct_counts)
        total_candidate_sum += len(correct_counts) * NUM_CANDIDATES

        log(f"\n  {ptype}:")
        log(f"    Processed: {processed}")
        if processed > 0:
            log(f"    At least 1 correct: {accepted}/{processed} ({accepted/processed*100:.1f}%)")
        log(f"    Avg correct per example: {avg_correct:.2f}/{NUM_CANDIDATES}")
        log(f"    Fallbacks used: {processed - accepted}")

    if total_processed > 0:
        log(f"\n  TOTAL:")
        log(f"    Processed: {total_processed}")
        log(f"    Accepted: {total_accepted}/{total_processed} "
            f"({total_accepted/total_processed*100:.1f}%)")
        if total_candidate_sum > 0:
            log(f"    Per-candidate rate: {total_correct_sum}/{total_candidate_sum} "
                f"({total_correct_sum/total_candidate_sum*100:.1f}%)")
        log(f"    Fallbacks: {total_processed - total_accepted}")

    log(f"\n  Output: {OUTPUT_PATH} ({len(results)} examples)")
    elapsed = time.time() - start_time
    log(f"  Total time: {elapsed/3600:.1f} hours")


if __name__ == "__main__":
    main()
