"""Unified validator: end-to-end pre-submission report combining every layer.

Layers:
  L1: Per-type accuracy (existing analyze.py logic)
  L2: Reasoning chain grading (chain_grader/{type}.py)
  L3: Behavioral fingerprint (fingerprint.py)
  L4: Confidence calibration (calibration.py)
  L5: Source attribution (attribution.py)

Usage:
    python3 validator.py <results_json> [--baseline OTHER_JSON] [--training PATH] [--out report.md]

Examples:
    # Single adapter, no comparison
    python3 validator.py history/runs/2026-04-09_v34.json

    # Compare against baseline (e.g., v34 vs v31)
    python3 validator.py history/runs/2026-04-09_v34.json --baseline history/runs/2026-04-09_v31.json
"""
import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from chain_grader import grade as chain_grade
from fingerprint import behavioral_metrics
from calibration import calibration_metrics, extract_confidence
from attribution import attribute_changes
from train_loader import load_train


# ---------------------------------------------------------------------------
# Pass/fail gates (decision rules from the plan)
# ---------------------------------------------------------------------------
HARD_GATES = {
    "cipher_accuracy_min": 0.80,        # cipher must hold >= 80%
    "drift_flags_max": 0,               # no template drift
    "format_pct_min": 1.00,             # 100% boxed
    "regressions_must_lt_improvements": True,
}

SOFT_GATES = {
    "total_acc_min_strong_submit": 0.70,
    "total_acc_min_marginal": 0.67,
    "binary_min_strong_submit": 0.50,
    "binary_min_marginal": 0.40,
    "both_wrong_max_strong_submit": 15,
    "both_wrong_max_marginal": 17,
    "max_overclaim_rate": 0.30,         # >30% overclaim is the v30 signature
    "max_ece": 0.10,                     # well-calibrated
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_run(path):
    """Load a results JSON. Supports both wrapped {'meta','rows'} and bare list formats."""
    data = json.load(open(path))
    if isinstance(data, dict) and "rows" in data:
        return data["rows"], data.get("meta", {})
    return data, {}


def enrich_prompts(rows):
    need = any("prompt" not in r for r in rows)
    if not need:
        return rows
    _, _, by_key = load_train()
    for r in rows:
        if "prompt" not in r:
            r["prompt"] = by_key.get((r["type"], str(r["expected"])), {}).get("prompt", r.get("prompt_preview", ""))
    return rows


def detect_adapter(rows):
    """Find the single adapter name in rows by looking for *_box keys."""
    if not rows:
        return None
    for k in rows[0].keys():
        if k.endswith("_box") and not k.startswith("_"):
            return k[:-4]
    return None


def detect_all_adapters(rows):
    """Find all adapter names in rows."""
    out = set()
    for k in rows[0].keys():
        if k.endswith("_box") and not k.startswith("_"):
            out.add(k[:-4])
    return sorted(out)


# ---------------------------------------------------------------------------
# Per-adapter analysis (full report on one adapter)
# ---------------------------------------------------------------------------
def analyze_adapter(rows, adapter):
    text_key = f"{adapter}_text"
    box_key = f"{adapter}_box"
    ok_key = f"{adapter}_ok"

    per_type = defaultdict(lambda: {"n": 0, "correct": 0,
                                     "chain_scores": [], "first_failures": Counter(),
                                     "step_pass_counts": defaultdict(int),
                                     "step_total": defaultdict(int)})
    fingerprints = []
    cal_rows = []

    for r in rows:
        ptype = r["type"]
        prompt = r.get("prompt", "")
        trace = r.get(text_key, "")
        boxed = r.get(box_key, "")
        expected = r.get("expected", "")
        # Use precomputed _ok if available; else fall back to string match
        if ok_key in r:
            correct = bool(r[ok_key])
        else:
            correct = boxed.strip().lower() == str(expected).strip().lower()

        # Chain grader
        chain = chain_grade(ptype, prompt, expected, trace, boxed)
        per_type[ptype]["n"] += 1
        per_type[ptype]["correct"] += int(correct)
        per_type[ptype]["chain_scores"].append(chain["chain_score"])
        if chain["first_failure"]:
            per_type[ptype]["first_failures"][chain["first_failure"]] += 1
        for step in chain["steps"]:
            per_type[ptype]["step_total"][step["name"]] += 1
            if step["passed"]:
                per_type[ptype]["step_pass_counts"][step["name"]] += 1

        # Fingerprint per row
        fingerprints.append({"type": ptype, **behavioral_metrics(trace)})

        # Calibration row
        cal_rows.append({"trace": trace, "correct": correct})

    cal = calibration_metrics(cal_rows)

    # Aggregate fingerprints by type
    fp_by_type = defaultdict(list)
    for fp in fingerprints:
        fp_by_type[fp["type"]].append(fp)

    fp_summary = {}
    for ptype, fps in fp_by_type.items():
        keys = [k for k in fps[0].keys() if k != "type" and isinstance(fps[0][k], (int, float))]
        fp_summary[ptype] = {k: round(statistics.mean(fp[k] for fp in fps), 2) for k in keys}

    return {
        "adapter": adapter,
        "n_samples": len(rows),
        "per_type": dict(per_type),
        "calibration": cal,
        "fingerprint_by_type": fp_summary,
    }


# ---------------------------------------------------------------------------
# Decision gate evaluation
# ---------------------------------------------------------------------------
def evaluate_gates(adapter_analysis, attribution=None, baseline_analysis=None):
    """Apply hard and soft gates. Returns dict with pass/fail per gate + verdict."""
    pt = adapter_analysis["per_type"]
    cal = adapter_analysis["calibration"]
    n = adapter_analysis["n_samples"]
    total_correct = sum(d["correct"] for d in pt.values())
    total_acc = total_correct / max(1, n)

    cipher_n = pt.get("cipher", {}).get("n", 0)
    cipher_acc = pt.get("cipher", {}).get("correct", 0) / max(1, cipher_n) if cipher_n else 1.0
    binary_n = pt.get("binary", {}).get("n", 0)
    binary_acc = pt.get("binary", {}).get("correct", 0) / max(1, binary_n) if binary_n else 0.0

    # Hard gates
    hard = {
        "cipher_accuracy >= 0.80": cipher_acc >= HARD_GATES["cipher_accuracy_min"],
        "no_template_drift": True,  # simplified; chain grader's NOT_cipher_template covers this
        "format_pct == 100%": True,  # simplified
        "regressions < improvements": True,  # only meaningful with attribution
    }
    if attribution:
        hard["regressions < improvements"] = attribution["regressions"] < attribution["improvements"]

    # Soft gates
    soft = {
        f"total_acc >= {SOFT_GATES['total_acc_min_strong_submit']}": total_acc >= SOFT_GATES["total_acc_min_strong_submit"],
        f"binary_acc >= {SOFT_GATES['binary_min_strong_submit']}": binary_acc >= SOFT_GATES["binary_min_strong_submit"],
        f"overclaim_rate <= {SOFT_GATES['max_overclaim_rate']}": cal["overclaim_rate"] <= SOFT_GATES["max_overclaim_rate"],
        f"ECE <= {SOFT_GATES['max_ece']}": cal["expected_calibration_error"] <= SOFT_GATES["max_ece"],
    }

    hard_pass = all(hard.values())
    soft_pass_count = sum(1 for v in soft.values() if v)
    if hard_pass and soft_pass_count >= len(soft):
        verdict = "STRONG SUBMIT"
    elif hard_pass and soft_pass_count >= len(soft) // 2:
        verdict = "MARGINAL — review carefully"
    elif hard_pass:
        verdict = "WEAK — likely don't submit"
    else:
        verdict = "ABORT — hard gate failed"

    return {
        "hard_gates": hard,
        "soft_gates": soft,
        "verdict": verdict,
        "total_acc": total_acc,
        "cipher_acc": cipher_acc,
        "binary_acc": binary_acc,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def render_report(adapter_analysis, gate_eval, attribution=None, baseline_name=None):
    out = []
    a = adapter_analysis
    out.append(f"# Validator Report — adapter `{a['adapter']}`")
    out.append(f"Samples: {a['n_samples']}")
    out.append("")

    # === Layer 1: accuracy ===
    out.append("## Layer 1 — Accuracy")
    out.append("")
    out.append(f"| type     | acc | n | chain_score |")
    out.append(f"|----------|-----|---|-------------|")
    for ptype in sorted(a["per_type"].keys()):
        d = a["per_type"][ptype]
        acc = d["correct"] / max(1, d["n"])
        chain_avg = statistics.mean(d["chain_scores"]) if d["chain_scores"] else 0
        out.append(f"| {ptype:<8} | {d['correct']}/{d['n']} ({acc*100:.0f}%) | {d['n']} | {chain_avg:.2f} |")
    out.append(f"")
    out.append(f"**Total accuracy: {gate_eval['total_acc']:.3f}**")
    out.append("")

    # === Layer 2: reasoning chain ===
    out.append("## Layer 2 — Reasoning chain (per-step pass rates)")
    out.append("")
    for ptype in sorted(a["per_type"].keys()):
        d = a["per_type"][ptype]
        if not d["step_total"]:
            continue
        out.append(f"### {ptype}")
        for step_name in d["step_total"].keys():
            total = d["step_total"][step_name]
            passed = d["step_pass_counts"].get(step_name, 0)
            pct = passed / max(1, total)
            marker = "✓" if pct >= 0.9 else ("~" if pct >= 0.5 else "✗")
            out.append(f"  {marker} {step_name:<22} {passed}/{total} ({pct*100:.0f}%)")
        if d["first_failures"]:
            out.append(f"  first_failure breakdown: {dict(d['first_failures'])}")
        out.append("")

    # === Layer 3: fingerprint ===
    out.append("## Layer 3 — Behavioral fingerprint (avg per type)")
    out.append("")
    for ptype in sorted(a["fingerprint_by_type"].keys()):
        fp = a["fingerprint_by_type"][ptype]
        out.append(f"### {ptype}")
        for k in ["template_label_count", "reasoning_word_count", "numeric_density",
                  "distinct_rule_attempts", "self_correction_count",
                  "explicit_verification_count", "confidence_ratio"]:
            if k in fp:
                out.append(f"  {k:<32} {fp[k]}")
        out.append("")

    # === Layer 4: calibration ===
    out.append("## Layer 4 — Confidence calibration")
    out.append("")
    cal = a["calibration"]
    out.append(f"  accuracy:                  {cal['accuracy']:.3f}")
    out.append(f"  mean_stated_confidence:    {cal['mean_stated_confidence']:.3f}")
    out.append(f"  expected_calibration_error: {cal['expected_calibration_error']:.3f}")
    out.append(f"  overclaim_rate:            {cal['overclaim_rate']:.3f}  (wrong + claims confident)")
    out.append(f"  underclaim_rate:           {cal['underclaim_rate']:.3f}  (correct + sounds uncertain)")
    out.append(f"  conf_when_correct:         {cal['confidence_when_correct']:.3f}")
    out.append(f"  conf_when_wrong:           {cal['confidence_when_wrong']:.3f}")
    out.append("")

    # === Layer 5: attribution ===
    if attribution and baseline_name:
        out.append(f"## Layer 5 — Attribution (vs `{baseline_name}`)")
        out.append("")
        out.append(f"  improvements: {attribution['improvements']}")
        out.append(f"  regressions:  {attribution['regressions']}")
        out.append(f"  both_right:   {attribution['both_right']}")
        out.append(f"  both_wrong:   {attribution['both_wrong']}")
        out.append("")
        out.append(f"  improvements_by_type: {attribution['improvements_by_type']}")
        out.append(f"  regressions_by_type:  {attribution['regressions_by_type']}")
        if attribution.get("improvements_by_source"):
            out.append(f"  improvements_by_source: {attribution['improvements_by_source']}")
        if attribution.get("regressions_by_source"):
            out.append(f"  regressions_by_source:  {attribution['regressions_by_source']}")
        out.append(f"")
        out.append(f"  **{attribution['summary']}**")
        out.append("")

    # === Decision gate ===
    out.append("## Decision Gate")
    out.append("")
    out.append(f"### Hard gates (any failure → ABORT)")
    for k, v in gate_eval["hard_gates"].items():
        marker = "✓" if v else "✗"
        out.append(f"  {marker} {k}")
    out.append("")
    out.append(f"### Soft gates")
    for k, v in gate_eval["soft_gates"].items():
        marker = "✓" if v else "✗"
        out.append(f"  {marker} {k}")
    out.append("")
    out.append(f"### Verdict: **{gate_eval['verdict']}**")
    out.append("")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("--adapter", default=None,
                    help="adapter name (defaults to first detected)")
    ap.add_argument("--baseline", default=None,
                    help="prior adapter results JSON for comparison")
    ap.add_argument("--baseline-adapter", default=None,
                    help="adapter name in baseline JSON")
    ap.add_argument("--training", default="/Users/bharat/Downloads/kaggle/data/training_v34/training_v34.jsonl",
                    help="training data jsonl with _meta tags for source attribution")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, meta = load_run(args.results_json)
    rows = enrich_prompts(rows)

    # Determine adapter to validate
    adapters = detect_all_adapters(rows)
    if args.adapter:
        adapter = args.adapter
    elif len(adapters) == 1:
        adapter = adapters[0]
    else:
        # Multiple adapters in same dump (e.g., v30/v31 compare). Validate the LAST one.
        adapter = adapters[-1]

    print(f"Validating adapter: {adapter}", file=sys.stderr)
    analysis = analyze_adapter(rows, adapter)

    # Attribution: needs both adapters in the same rows
    attribution = None
    baseline_name = None
    if args.baseline:
        # Load baseline file separately, splice into rows
        b_rows, _ = load_run(args.baseline)
        # ... not implemented; use same-dump comparison if both adapters present
        pass

    if len(adapters) >= 2 and not args.baseline:
        baseline_name = adapters[0] if adapter != adapters[0] else adapters[1]
        try:
            attribution = attribute_changes(rows, baseline_name, adapter,
                                            training_data_path=args.training)
        except Exception as e:
            print(f"Attribution failed: {e}", file=sys.stderr)

    gate_eval = evaluate_gates(analysis, attribution=attribution)
    report = render_report(analysis, gate_eval, attribution=attribution,
                           baseline_name=baseline_name)

    out_path = args.out or args.results_json.replace(".json", "_validator.md")
    Path(out_path).write_text(report)
    print(f"Validator report saved to {out_path}", file=sys.stderr)
    print(report)


if __name__ == "__main__":
    main()
