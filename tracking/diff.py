"""Diff two adapters from the same raw JSON dump.

Usage:
    python3 diff.py <results.json> <adapter_a> <adapter_b>
    python3 diff.py /tmp/compare_v30_v31.json v30 v31

Surfaces:
- Per-type delta (acc_a - acc_b)
- Regressions (a right, b wrong) with prompt + both traces truncated
- Improvements (a wrong, b right)
- Both wrong (the hard core — candidates for next training data)
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from verifiers import verify_row
from train_loader import load_train


def enrich_prompts(rows):
    need_full = any('prompt' not in r for r in rows)
    if not need_full:
        return rows
    _, _, by_key = load_train()
    for r in rows:
        if 'prompt' not in r:
            key = (r['type'], str(r['expected']))
            r['prompt'] = by_key.get(key, {}).get('prompt', r.get('prompt_preview', ''))
    return rows


def grade_all(rows, adapter):
    verdicts = []
    for r in rows:
        v = verify_row(r['type'], r.get('prompt', ''), r.get('expected', ''),
                       r.get(f'{adapter}_text', ''), r.get(f'{adapter}_box', ''))
        verdicts.append(v)
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("adapter_a")
    ap.add_argument("adapter_b")
    ap.add_argument("--show", type=int, default=3, help="num failing cases to show per bucket")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = json.load(open(args.results_json))
    rows = enrich_prompts(rows)

    a_v = grade_all(rows, args.adapter_a)
    b_v = grade_all(rows, args.adapter_b)

    out = []
    out.append(f"# Diff: {args.adapter_a} vs {args.adapter_b}")
    out.append(f"Samples: {len(rows)}")
    out.append("")

    # Per-type accuracy delta
    per_type = defaultdict(lambda: {"a": 0, "b": 0, "n": 0})
    for r, av, bv in zip(rows, a_v, b_v):
        t = r['type']
        per_type[t]["a"] += int(av["correct"])
        per_type[t]["b"] += int(bv["correct"])
        per_type[t]["n"] += 1

    out.append("## Per-type accuracy delta")
    out.append("")
    out.append(f"| type     | {args.adapter_a:>6} | {args.adapter_b:>6} | delta  |")
    out.append("|----------|--------|--------|--------|")
    for t in sorted(per_type.keys()):
        d = per_type[t]
        a_pct = d["a"] / d["n"]
        b_pct = d["b"] / d["n"]
        out.append(f"| {t:<8} | {d['a']}/{d['n']:<4} | {d['b']}/{d['n']:<4} | {b_pct-a_pct:+.2f}  |")
    tot_a = sum(d["a"] for d in per_type.values())
    tot_b = sum(d["b"] for d in per_type.values())
    tot_n = sum(d["n"] for d in per_type.values())
    out.append(f"| TOTAL    | {tot_a}/{tot_n:<4} | {tot_b}/{tot_n:<4} | {(tot_b-tot_a)/tot_n:+.3f} |")
    out.append("")

    # Bucket rows
    regressions = []  # a correct, b wrong
    improvements = []  # a wrong, b correct
    both_right = 0
    both_wrong = []

    for r, av, bv in zip(rows, a_v, b_v):
        if av["correct"] and bv["correct"]:
            both_right += 1
        elif av["correct"] and not bv["correct"]:
            regressions.append((r, av, bv))
        elif not av["correct"] and bv["correct"]:
            improvements.append((r, av, bv))
        else:
            both_wrong.append((r, av, bv))

    out.append("## Case breakdown")
    out.append(f"- both_right:  {both_right}")
    out.append(f"- both_wrong:  {len(both_wrong)}")
    out.append(f"- regressions: {len(regressions)}  ({args.adapter_a}→right, {args.adapter_b}→wrong)")
    out.append(f"- improvements: {len(improvements)}  ({args.adapter_a}→wrong, {args.adapter_b}→right)")
    out.append("")

    # Regression type breakdown
    out.append("### Regressions by type & failure mode")
    reg_by_type = defaultdict(lambda: defaultdict(int))
    for r, av, bv in regressions:
        reg_by_type[r['type']][bv['failure_mode']] += 1
    for t, modes in reg_by_type.items():
        out.append(f"- {t}: {dict(modes)}")
    out.append("")

    # Improvement breakdown
    out.append("### Improvements by type & prior failure mode")
    imp_by_type = defaultdict(lambda: defaultdict(int))
    for r, av, bv in improvements:
        imp_by_type[r['type']][av['failure_mode']] += 1
    for t, modes in imp_by_type.items():
        out.append(f"- {t}: {dict(modes)}")
    out.append("")

    # Both wrong breakdown — the hard core
    out.append("### Both wrong by type (hardest cases → next training data candidates)")
    bw_by_type = defaultdict(int)
    for r, _, _ in both_wrong:
        bw_by_type[r['type']] += 1
    for t, n in sorted(bw_by_type.items(), key=lambda x: -x[1]):
        out.append(f"- {t}: {n}")
    out.append("")

    # Show sample cases
    def show_case(label, bucket, n=args.show):
        out.append(f"## Sample {label} (top {n})")
        out.append("")
        for i, (r, av, bv) in enumerate(bucket[:n]):
            out.append(f"### {label} #{i+1}: {r['type']}")
            out.append(f"**Expected:** `{r['expected']}`")
            out.append(f"**{args.adapter_a} box:** `{r.get(f'{args.adapter_a}_box', '')}`  "
                       f"→ {'✓' if av['correct'] else '✗'} ({av['failure_mode']})")
            out.append(f"**{args.adapter_b} box:** `{r.get(f'{args.adapter_b}_box', '')}`  "
                       f"→ {'✓' if bv['correct'] else '✗'} ({bv['failure_mode']})")
            prompt = r.get('prompt', '')[:400]
            out.append(f"**Prompt:**")
            out.append("```")
            out.append(prompt)
            out.append("```")
            out.append("")

    show_case("Regression", regressions)
    show_case("Improvement", improvements)
    show_case("Both-wrong", both_wrong)

    report = "\n".join(out)
    out_path = args.out or f"/Users/bharat/Downloads/kaggle/tracking/reports/diff_{args.adapter_a}_vs_{args.adapter_b}.md"
    Path(out_path).write_text(report)
    print(f"Diff saved to {out_path}")
    print()
    # Print summary
    print("\n".join(out[:40]))


if __name__ == '__main__':
    main()
