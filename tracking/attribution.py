"""Source attribution for adapter-vs-adapter comparisons.

Given two adapter runs on the same prompts, identify which rows improved or
regressed, aggregate by puzzle type, and (optionally) cross-reference against
training data tagged with `_meta.source` so we can say "these binary improvements
came from v34_binary_solver IDENTITY/CONSTANT rules".

See data/generate_v34.py for the _meta schema produced per training example.
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Optional

# Allow running as a script from the tracking directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from verifiers.binary import parse_prompt as parse_binary_prompt, classify_bit


# ---------------------------------------------------------------------------
# Answer matching (permissive for gravity/unit; strict elsewhere)
# ---------------------------------------------------------------------------
def _num(s: str) -> Optional[float]:
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def _answer_matches(puzzle_type: str, expected: str, got: str) -> bool:
    """Match the same logic as the original /tmp/infer_compare.py verify():
    - binary string match for [01]+ stored answers
    - math.isclose(rel_tol=1e-2, abs_tol=1e-5) for any numeric pair
    - case-insensitive string equality otherwise
    """
    if expected is None or got is None:
        return False
    e = str(expected).strip()
    g = str(got).strip()
    if not g:
        return False
    # Binary string match (8-bit binary answers)
    import re
    if re.fullmatch(r'[01]+', e):
        return g.lower() == e.lower()
    # Numeric tolerance (gravity/unit/anything that parses as float)
    try:
        return math.isclose(float(e), float(g), rel_tol=1e-2, abs_tol=1e-5)
    except (TypeError, ValueError):
        pass
    return g.lower() == e.lower()


# ---------------------------------------------------------------------------
# Row prompt retrieval
# ---------------------------------------------------------------------------
def _row_prompt(row: dict) -> str:
    """Prefer full `prompt`, fall back to `prompt_preview`. Returns '' if neither."""
    return row.get("prompt") or row.get("prompt_preview") or ""


# ---------------------------------------------------------------------------
# Binary rule-kind overlap
# ---------------------------------------------------------------------------
def _row_binary_rule_kinds(prompt: str) -> set:
    """Return the set of per-bit rule kinds for a binary row, or empty set if
    unparseable (e.g. truncated prompt_preview)."""
    if not prompt:
        return set()
    pairs, _target = parse_binary_prompt(prompt)
    if not pairs:
        return set()
    kinds = set()
    for k in range(8):
        rule = classify_bit(pairs, k)
        if rule is not None:
            kinds.add(rule["kind"])
    return kinds


def _binary_rule_overlap(prompt: str, training_source_rule_kinds: set) -> bool:
    row_kinds = _row_binary_rule_kinds(prompt)
    if not row_kinds or not training_source_rule_kinds:
        return False
    return len(row_kinds & training_source_rule_kinds) > 0


# ---------------------------------------------------------------------------
# Training data loader
# ---------------------------------------------------------------------------
def _load_training_sources(training_data_path: str) -> dict:
    """Group training examples by (source, puzzle_type).

    Returns:
        {
            source_name: {
                "puzzle_types": Counter,
                "binary_rule_kinds": set,  # union of all rule_kinds across binary examples
                "total": int,
            },
            ...
        }
    """
    sources = defaultdict(lambda: {
        "puzzle_types": Counter(),
        "binary_rule_kinds": set(),
        "total": 0,
    })
    with open(training_data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = ex.get("_meta") or {}
            src = meta.get("source", "unknown")
            ptype = meta.get("puzzle_type", "unknown")
            rule_kinds = meta.get("rule_kinds") or []
            bucket = sources[src]
            bucket["puzzle_types"][ptype] += 1
            bucket["total"] += 1
            if ptype == "binary":
                bucket["binary_rule_kinds"].update(rule_kinds)
    return dict(sources)


# ---------------------------------------------------------------------------
# Source attribution for a single row
# ---------------------------------------------------------------------------
def _attribute_row(row: dict, training_sources: dict) -> list:
    """Return list of source names that could plausibly account for this row's
    outcome, based on puzzle_type (and, for binary, rule-kind overlap)."""
    ptype = row.get("type", "unknown")
    attributed = []
    for src, info in training_sources.items():
        if info["puzzle_types"].get(ptype, 0) == 0:
            continue
        if ptype == "binary":
            # For binary, require rule-kind overlap IF the training source tracks rule kinds
            # AND the row's prompt is parseable. Otherwise fall back to pure type match.
            src_kinds = info["binary_rule_kinds"]
            if src_kinds:
                prompt = _row_prompt(row)
                row_kinds = _row_binary_rule_kinds(prompt)
                if row_kinds and not (row_kinds & src_kinds):
                    continue
                # If row prompt isn't parseable (truncated preview), fall through and
                # credit the source by type match alone.
            attributed.append(src)
        else:
            attributed.append(src)
    return attributed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def attribute_changes(rows, adapter_a: str, adapter_b: str,
                      training_data_path: Optional[str] = None) -> dict:
    """Compare two adapter runs on the same prompts and attribute score changes.

    Args:
        rows: list of dicts, each containing BOTH adapters' outputs. Expected keys:
            'type', 'expected', 'prompt' (or 'prompt_preview'),
            '<adapter_a>_box', '<adapter_b>_box' (and optionally _text, _ok).
        adapter_a, adapter_b: short adapter names (e.g. 'v30', 'v31').
        training_data_path: optional path to a training_v*.jsonl file where each
            example has _meta.source / puzzle_type / rule_kinds. If provided,
            improvements/regressions are cross-referenced to training sources.

    Returns:
        dict with keys: n_samples, improvements, regressions, both_right, both_wrong,
        improvements_by_type, regressions_by_type, improvements_by_source,
        regressions_by_source, summary.
    """
    a_box = f"{adapter_a}_box"
    b_box = f"{adapter_b}_box"

    training_sources = None
    if training_data_path:
        training_sources = _load_training_sources(training_data_path)

    improvements = []
    regressions = []
    both_right = 0
    both_wrong = 0

    a_ok_key = f"{adapter_a}_ok"
    b_ok_key = f"{adapter_b}_ok"

    for row in rows:
        ptype = row.get("type", "unknown")
        expected = row.get("expected", "")
        a_ans = row.get(a_box, "") or ""
        b_ans = row.get(b_box, "") or ""
        # Prefer precomputed ok flags from the JSON (matches the original verify exactly).
        # Fall back to recomputing for new dumps that don't ship _ok.
        a_ok = row[a_ok_key] if a_ok_key in row else _answer_matches(ptype, expected, a_ans)
        b_ok = row[b_ok_key] if b_ok_key in row else _answer_matches(ptype, expected, b_ans)
        if a_ok and b_ok:
            both_right += 1
        elif not a_ok and not b_ok:
            both_wrong += 1
        elif b_ok and not a_ok:
            improvements.append(row)
        elif a_ok and not b_ok:
            regressions.append(row)

    improvements_by_type = Counter(r.get("type", "unknown") for r in improvements)
    regressions_by_type = Counter(r.get("type", "unknown") for r in regressions)

    improvements_by_source = None
    regressions_by_source = None
    if training_sources is not None:
        improvements_by_source = Counter()
        regressions_by_source = Counter()
        for row in improvements:
            for src in _attribute_row(row, training_sources):
                improvements_by_source[src] += 1
        for row in regressions:
            for src in _attribute_row(row, training_sources):
                regressions_by_source[src] += 1

    # ------------------------------------------------------------------
    # Build a human-readable summary
    # ------------------------------------------------------------------
    parts = []
    net = len(improvements) - len(regressions)
    sign = "+" if net >= 0 else ""
    parts.append(f"{adapter_a}->{adapter_b}: net {sign}{net} ({len(improvements)} up, {len(regressions)} down)")

    if improvements_by_type:
        imp_str = ", ".join(f"+{c} {t}" for t, c in improvements_by_type.most_common())
        parts.append(imp_str)
    if regressions_by_type:
        reg_str = ", ".join(f"-{c} {t}" for t, c in regressions_by_type.most_common())
        parts.append(reg_str)

    if improvements_by_source:
        src_str = ", ".join(
            f"{c} via {s}" for s, c in improvements_by_source.most_common()
        )
        parts.append(f"improvements attributed: {src_str}")
    if regressions_by_source:
        src_str = ", ".join(
            f"{c} via {s}" for s, c in regressions_by_source.most_common()
        )
        parts.append(f"regressions attributed: {src_str}")

    summary = " | ".join(parts)

    return {
        "n_samples": len(rows),
        "improvements": len(improvements),
        "regressions": len(regressions),
        "both_right": both_right,
        "both_wrong": both_wrong,
        "improvements_by_type": dict(improvements_by_type),
        "regressions_by_type": dict(regressions_by_type),
        "improvements_by_source": dict(improvements_by_source) if improvements_by_source is not None else None,
        "regressions_by_source": dict(regressions_by_source) if regressions_by_source is not None else None,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# CLI for ad hoc use
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Attribute score changes between two adapters")
    ap.add_argument("rows_json", help="Path to JSON list with both adapters' outputs per row")
    ap.add_argument("adapter_a", help="Baseline adapter name (e.g. v30)")
    ap.add_argument("adapter_b", help="New adapter name (e.g. v31)")
    ap.add_argument("--training", help="Training JSONL with _meta tags", default=None)
    args = ap.parse_args()

    with open(args.rows_json) as f:
        rows = json.load(f)
    result = attribute_changes(rows, args.adapter_a, args.adapter_b,
                               training_data_path=args.training)
    print(json.dumps(result, indent=2))
