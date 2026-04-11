"""Analyze a raw inference dump and produce per-type metrics + failure report.

Usage:
    python3 analyze.py <results.json> [--adapter NAME] [--out report.md]

Input JSON format: list of dicts with fields
    type, expected, prompt (or prompt_preview), {adapter}_text, {adapter}_box
For single-adapter dumps, the adapter key should be passed explicitly.
"""
import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from verifiers import verify_row
from train_loader import load_train


def pick_adapter_fields(rows, adapter_hint=None):
    """Auto-detect which adapter keys are in the dump.
    Returns list of adapter names, e.g. ['v31', 'v30'] or ['v33']."""
    if not rows:
        return []
    keys = set(rows[0].keys())
    adapters = set()
    for k in keys:
        if k.endswith('_text') and not k.startswith('_'):
            adapters.add(k[:-5])
    if adapter_hint and adapter_hint in adapters:
        return [adapter_hint]
    return sorted(adapters)


def enrich_prompts(rows):
    """If rows only have prompt_preview, look up full prompts from train.csv by (type, expected)."""
    need_full = any('prompt' not in r for r in rows)
    if not need_full:
        return rows
    _, _, by_key = load_train()
    for r in rows:
        if 'prompt' in r:
            continue
        key = (r['type'], str(r['expected']))
        full = by_key.get(key)
        if full:
            r['prompt'] = full['prompt']
        else:
            r['prompt'] = r.get('prompt_preview', '')
    return rows


def analyze_adapter(rows, adapter):
    """Run per-row verifier for one adapter; return per-type metrics."""
    per_type = defaultdict(lambda: {
        "n": 0, "correct": 0, "failures": Counter(),
        "rows": [],
    })
    # Aggregate cipher-specific metrics
    cipher_agg = {
        "ver_honesty_dist": Counter(),
        "table_emitted": 0, "table_acc_sum": 0.0, "table_acc_n": 0,
        "vocab_fill_used": 0,
    }
    binary_agg = {"difficulty": Counter(), "matches_simple_truth": 0, "solvable_simple": 0}
    equation_agg = {"subtype": Counter(), "ran_cipher_template": 0, "attempted_arith": 0}
    drift_flags = defaultdict(int)  # type → count of cipher_template_drift

    text_key = f"{adapter}_text"
    box_key = f"{adapter}_box"

    for r in rows:
        t = r['type']
        prompt = r.get('prompt', r.get('prompt_preview', ''))
        trace = r.get(text_key, '')
        boxed = r.get(box_key, '')
        expected = r.get('expected', '')

        result = verify_row(t, prompt, expected, trace, boxed)
        per_type[t]["n"] += 1
        per_type[t]["correct"] += int(result["correct"])
        per_type[t]["failures"][result["failure_mode"]] += 1
        per_type[t]["rows"].append({**r, "_verdict": result})

        if t == 'cipher':
            cipher_agg["ver_honesty_dist"][result["quality_flags"].get("ver_honesty", "NONE")] += 1
            if result["quality_flags"].get("table_emitted"):
                cipher_agg["table_emitted"] += 1
                acc = result["quality_flags"].get("table_accuracy")
                if acc is not None:
                    cipher_agg["table_acc_sum"] += acc
                    cipher_agg["table_acc_n"] += 1
            if result["quality_flags"].get("vocab_fill_used"):
                cipher_agg["vocab_fill_used"] += 1
        elif t == 'binary':
            diff = result["ground_truth"].get("difficulty", "?")
            binary_agg["difficulty"][diff] += 1
            if result["quality_flags"].get("answer_matches_truth"):
                binary_agg["matches_simple_truth"] += 1
            if result["quality_flags"].get("solvable_with_simple_gates"):
                binary_agg["solvable_simple"] += 1
        elif t == 'equation':
            equation_agg["subtype"][result["ground_truth"].get("subtype", "?")] += 1
            if result["quality_flags"].get("ran_cipher_template"):
                equation_agg["ran_cipher_template"] += 1
            if result["quality_flags"].get("attempted_arithmetic"):
                equation_agg["attempted_arith"] += 1

        if result["failure_mode"] == "cipher_template_drift":
            drift_flags[t] += 1

    return {
        "per_type": {t: {**d, "failures": dict(d["failures"]),
                         "acc": d["correct"] / max(1, d["n"])}
                     for t, d in per_type.items()},
        "cipher_agg": {**cipher_agg,
                       "ver_honesty_dist": dict(cipher_agg["ver_honesty_dist"]),
                       "avg_table_accuracy": (cipher_agg["table_acc_sum"] / cipher_agg["table_acc_n"])
                       if cipher_agg["table_acc_n"] else None},
        "binary_agg": {**binary_agg, "difficulty": dict(binary_agg["difficulty"])},
        "equation_agg": {**equation_agg, "subtype": dict(equation_agg["subtype"])},
        "drift_flags": dict(drift_flags),
    }


def format_report(rows, results_by_adapter, adapters):
    out = []
    out.append(f"# Tracking Report — {len(rows)} samples")
    out.append("")

    # ========================================
    # Tier 1: Per-type accuracy
    # ========================================
    out.append("## Tier 1 — Accuracy per type")
    out.append("")
    header = "| type     | " + " | ".join(adapters) + " |"
    sep = "|----------|" + "|".join(["--------"] * len(adapters)) + "|"
    out.append(header)
    out.append(sep)
    types = sorted({t for a in adapters for t in results_by_adapter[a]["per_type"].keys()})
    for t in types:
        row = f"| {t:<8} | "
        cells = []
        for a in adapters:
            d = results_by_adapter[a]["per_type"].get(t, {"correct": 0, "n": 0, "acc": 0})
            cells.append(f"{d['correct']}/{d['n']} ({d['acc']*100:.0f}%)")
        row += " | ".join(cells) + " |"
        out.append(row)

    # Total
    out.append("")
    out.append("**Totals:**")
    for a in adapters:
        tot = sum(d["correct"] for d in results_by_adapter[a]["per_type"].values())
        n = sum(d["n"] for d in results_by_adapter[a]["per_type"].values())
        out.append(f"- {a}: {tot}/{n} = {tot/max(1,n):.3f}")
    out.append("")

    # ========================================
    # Tier 3: Cipher deep-dive
    # ========================================
    out.append("## Tier 3 — Cipher step-derivation quality")
    out.append("")
    for a in adapters:
        c = results_by_adapter[a]["cipher_agg"]
        n_cipher = results_by_adapter[a]["per_type"].get("cipher", {}).get("n", 0)
        out.append(f"### {a}")
        out.append(f"- cipher samples: {n_cipher}")
        out.append(f"- table emitted: {c['table_emitted']}/{n_cipher}")
        if c.get("avg_table_accuracy") is not None:
            out.append(f"- avg table accuracy (when emitted): {c['avg_table_accuracy']:.2f}")
        out.append(f"- vocab fill used: {c['vocab_fill_used']}/{n_cipher}")
        out.append(f"- VER honesty distribution: {c['ver_honesty_dist']}")
        out.append("")

    # ========================================
    # Binary difficulty
    # ========================================
    out.append("## Binary difficulty breakdown")
    out.append("")
    for a in adapters:
        b = results_by_adapter[a]["binary_agg"]
        n_bin = results_by_adapter[a]["per_type"].get("binary", {}).get("n", 0)
        out.append(f"### {a}")
        out.append(f"- difficulty mix: {b['difficulty']}")
        out.append(f"- solvable with simple gates (constant/identity/NOT/2-input/majority): {b['solvable_simple']}/{n_bin}")
        out.append(f"- answer matches derived truth: {b['matches_simple_truth']}/{n_bin}")
        out.append("")

    # ========================================
    # Equation subtype + template misfire
    # ========================================
    out.append("## Equation: subtype + cipher-template misfire")
    out.append("")
    for a in adapters:
        e = results_by_adapter[a]["equation_agg"]
        n_eq = results_by_adapter[a]["per_type"].get("equation", {}).get("n", 0)
        out.append(f"### {a}")
        out.append(f"- subtype mix: {e['subtype']}")
        out.append(f"- ran cipher template on equation: {e['ran_cipher_template']}/{n_eq}")
        out.append(f"- attempted arithmetic: {e['attempted_arith']}/{n_eq}")
        out.append("")

    # ========================================
    # Failure mode breakdown per type
    # ========================================
    out.append("## Failure modes (per type, per adapter)")
    out.append("")
    for t in types:
        out.append(f"### {t}")
        for a in adapters:
            d = results_by_adapter[a]["per_type"].get(t, {"failures": {}})
            out.append(f"- {a}: {d['failures']}")
        out.append("")

    # ========================================
    # Drift flags (any type got cipher_template_drift mode)
    # ========================================
    out.append("## Template drift (non-cipher types running cipher template)")
    out.append("")
    for a in adapters:
        df = results_by_adapter[a]["drift_flags"]
        if df:
            out.append(f"- {a}: {df}")
        else:
            out.append(f"- {a}: none")
    out.append("")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", help="path to raw inference dump (v30/v31 compare JSON)")
    ap.add_argument("--adapter", default=None, help="specific adapter name in dump (else all found)")
    ap.add_argument("--out", default=None, help="output markdown path")
    args = ap.parse_args()

    rows = json.load(open(args.results_json))
    rows = enrich_prompts(rows)
    adapters = pick_adapter_fields(rows, args.adapter)
    print(f"Detected adapters: {adapters}")

    results_by_adapter = {}
    for a in adapters:
        print(f"Analyzing {a}...")
        results_by_adapter[a] = analyze_adapter(rows, a)

    report = format_report(rows, results_by_adapter, adapters)

    out_path = args.out or args.results_json.replace('.json', '_report.md')
    Path(out_path).write_text(report)
    print(f"Report saved to {out_path}")
    print()
    print(report)


if __name__ == '__main__':
    main()
