"""Append a run's summary metrics to history.jsonl for trajectory tracking.

Usage:
    python3 history.py <results.json>   # runs analyzer, appends each adapter's summary
    python3 history.py --show            # print trajectory table
"""
import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from analyze import analyze_adapter, pick_adapter_fields, enrich_prompts

HISTORY_PATH = Path(__file__).parent / "history" / "history.jsonl"
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)


def append_run(results_json: str, kaggle_score: float = None, notes: str = None):
    """Analyze results_json and append one line per adapter to history.jsonl."""
    data = json.load(open(results_json))
    # Support both wrapped {"meta", "rows"} and bare list formats
    if isinstance(data, dict) and "rows" in data:
        rows = data["rows"]
        meta = data.get("meta", {})
    else:
        rows = data
        meta = {}
    rows = enrich_prompts(rows)
    adapters = pick_adapter_fields(rows)

    for a in adapters:
        result = analyze_adapter(rows, a)
        per_type = {t: {"correct": d["correct"], "n": d["n"], "acc": d["acc"]}
                    for t, d in result["per_type"].items()}
        total_correct = sum(d["correct"] for d in result["per_type"].values())
        total_n = sum(d["n"] for d in result["per_type"].values())

        entry = {
            "timestamp": meta.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "adapter": a,
            "source_file": str(Path(results_json).name),
            "n_samples": total_n,
            "local_total_acc": total_correct / max(1, total_n),
            "kaggle_score": kaggle_score,
            "per_type": per_type,
            "cipher": {
                "avg_table_accuracy": result["cipher_agg"].get("avg_table_accuracy"),
                "vocab_fill_used": result["cipher_agg"]["vocab_fill_used"],
                "ver_honesty_dist": result["cipher_agg"]["ver_honesty_dist"],
            },
            "binary": {
                "difficulty": result["binary_agg"]["difficulty"],
                "matches_simple_truth": result["binary_agg"]["matches_simple_truth"],
            },
            "equation": {
                "ran_cipher_template": result["equation_agg"]["ran_cipher_template"],
                "attempted_arith": result["equation_agg"]["attempted_arith"],
            },
            "drift_flags": result["drift_flags"],
            "notes": notes,
        }
        with HISTORY_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"Appended {a}: {total_correct}/{total_n} = {entry['local_total_acc']:.3f}")


def show_history():
    if not HISTORY_PATH.exists():
        print("No history yet.")
        return
    entries = [json.loads(l) for l in HISTORY_PATH.read_text().splitlines() if l.strip()]
    if not entries:
        print("No history yet.")
        return

    types = ["binary", "cipher", "equation", "gravity", "roman", "unit"]
    print(f"{'timestamp':<20} {'adapter':<8} {'local':>6} {'kaggle':>7} " +
          " ".join(f"{t[:4]:>5}" for t in types) + "  notes")
    print("-" * 110)
    for e in entries:
        pt = e.get("per_type", {})
        acc_cells = []
        for t in types:
            d = pt.get(t, {})
            if d.get("n", 0):
                acc_cells.append(f"{d['acc']*100:>4.0f}%")
            else:
                acc_cells.append(" - ")
        kagscore = f"{e['kaggle_score']:.3f}" if e.get("kaggle_score") is not None else "-"
        print(f"{e['timestamp']:<20} {e['adapter']:<8} {e['local_total_acc']:>6.3f} "
              f"{kagscore:>7} " + " ".join(acc_cells) + f"  {e.get('notes') or ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", nargs="?")
    ap.add_argument("--kaggle-score", type=float, default=None,
                    help="public Kaggle score for this adapter (optional, for correlation)")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--show", action="store_true", help="print trajectory table")
    args = ap.parse_args()

    if args.show:
        show_history()
        return

    if not args.results_json:
        ap.error("results_json required (or use --show)")

    append_run(args.results_json, args.kaggle_score, args.notes)
    print()
    show_history()


if __name__ == '__main__':
    main()
