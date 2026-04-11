"""Confidence calibration metrics for model reasoning traces.

Given a list of ``{trace, correct, ...}`` dicts, these metrics measure
whether the model's stated confidence (extracted heuristically from the
wording of its trace) matches its actual accuracy.

All metrics are deterministic and use only the standard library.

Public API:
    calibration_metrics(rows: list[dict]) -> dict
    extract_confidence(trace: str) -> float    # exposed for reuse/testing
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Confidence markers (kept in-sync with fingerprint.py's lists, but copied
# locally so this file has no dependency on fingerprint.py).
# ---------------------------------------------------------------------------

# Two marker sets: english hedge words AND structured-trace verification claims.
# Technical reasoning traces (Donald-template) almost never use English hedges,
# so we need structured signals to extract real confidence.

_CONFIDENCE_HIGH_MARKERS = [
    # English hedges
    "certainly", "obviously", "clearly", "definitely",
    "must be", "is exactly", "i'm sure", "without doubt",
    # Structured-trace verification claims (Donald-template style)
    "ver: pass", "ver pass", "check: pass", "check pass",
    "ver: yes", "verification: pass", "matches all examples",
    "consistent across", "holds for every example",
    "all checks pass", "verified",
]

_CONFIDENCE_LOW_MARKERS = [
    # English hedges
    "i think", "maybe", "perhaps", "might be",
    "i'm not sure", "possibly", "could be", "appears to",
    # Structured-trace failure / uncertainty signals
    "ver: fail", "ver fail", "check: fail", "check fail",
    "ver: no", "couldn't determine", "unclear", "ambiguous",
    "doesn't match", "failed to verify", "no rule found",
    "inconsistent", "hardstop",
]

_DECLARATIVE_MARKERS = [
    "therefore", "i am confident", "the answer is",
    "ans:", "concat:",  # template-style commitment markers
]

_HIGH_DELTA = 0.1
_LOW_DELTA = -0.1
_DECLARATIVE_DELTA = 0.05

# Default for ambiguous traces. We bias toward 0.7 (committed) when the model
# emits a boxed answer, since that itself is a confidence claim. Empty/no-boxed
# traces default to 0.3 (uncommitted).
_DEFAULT_CONFIDENCE_BOXED = 0.7
_DEFAULT_CONFIDENCE_NOBOX = 0.3
_DEFAULT_CONFIDENCE = 0.5

_N_BINS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_occurrences(text_lower: str, markers: list[str]) -> int:
    total = 0
    for m in markers:
        if not m:
            continue
        total += text_lower.count(m)
    return total


def extract_confidence(trace: str) -> float:
    """Heuristic stated-confidence extractor.

    Starts at 0.5 and adjusts based on keyword evidence:
      +0.10 per high-confidence marker
      -0.10 per low-confidence marker
      +0.05 per declarative marker ("therefore", "i am confident",
            "the answer is")

    Final value is clamped to [0, 1]. This is a rough proxy — it captures
    whether the model's wording sounds confident, not whether it actually
    is confident in any calibrated sense.

    Args:
        trace: raw model output.

    Returns:
        Float in [0, 1]. Default 0.5 for empty / marker-free traces.
    """
    if not trace:
        return _DEFAULT_CONFIDENCE

    text_lower = trace.lower()

    # Default depends on whether the trace produced a boxed answer.
    # A model that commits to \boxed{X} is implicitly claiming high confidence,
    # even if the wording is technical and free of hedge words.
    has_boxed = "\\boxed{" in trace
    score = _DEFAULT_CONFIDENCE_BOXED if has_boxed else _DEFAULT_CONFIDENCE_NOBOX

    score += _HIGH_DELTA * _count_occurrences(text_lower, _CONFIDENCE_HIGH_MARKERS)
    score += _LOW_DELTA * _count_occurrences(text_lower, _CONFIDENCE_LOW_MARKERS)
    score += _DECLARATIVE_DELTA * _count_occurrences(
        text_lower, _DECLARATIVE_MARKERS
    )

    # Clamp
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0
    return float(score)


def _expected_calibration_error(
    confidences: list[float], correctness: list[bool]
) -> float:
    """Standard binned ECE with ``_N_BINS`` equal-width buckets on [0, 1].

    ECE = sum_b |acc_b - conf_b| * (n_b / N)
    """
    n = len(confidences)
    if n == 0:
        return 0.0

    # Bucket edges: [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    # The final bucket is closed on the right to include conf == 1.0.
    bucket_confs: list[list[float]] = [[] for _ in range(_N_BINS)]
    bucket_correct: list[list[bool]] = [[] for _ in range(_N_BINS)]

    for conf, ok in zip(confidences, correctness):
        # Map conf to bucket index in [0, _N_BINS - 1]
        idx = int(conf * _N_BINS)
        if idx >= _N_BINS:
            idx = _N_BINS - 1
        if idx < 0:
            idx = 0
        bucket_confs[idx].append(conf)
        bucket_correct[idx].append(bool(ok))

    ece = 0.0
    for b in range(_N_BINS):
        n_b = len(bucket_confs[b])
        if n_b == 0:
            continue
        mean_conf = sum(bucket_confs[b]) / n_b
        acc_b = sum(1 for c in bucket_correct[b] if c) / n_b
        ece += abs(acc_b - mean_conf) * (n_b / n)

    return float(ece)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calibration_metrics(rows: list[dict]) -> dict:
    """Compute aggregate calibration metrics over a list of graded rows.

    Args:
        rows: list of dicts, each with at minimum
              ``{"trace": str, "correct": bool}``. Additional keys are
              ignored.

    Returns:
        Flat dict with keys:
          - n_samples
          - accuracy
          - mean_stated_confidence
          - expected_calibration_error
          - overclaim_rate
          - underclaim_rate
          - confidence_when_correct
          - confidence_when_wrong
    """
    n = len(rows)
    if n == 0:
        return {
            "n_samples": 0,
            "accuracy": 0.0,
            "mean_stated_confidence": 0.0,
            "expected_calibration_error": 0.0,
            "overclaim_rate": 0.0,
            "underclaim_rate": 0.0,
            "confidence_when_correct": 0.0,
            "confidence_when_wrong": 0.0,
        }

    confidences: list[float] = []
    correctness: list[bool] = []
    for r in rows:
        trace = r.get("trace", "") or ""
        ok = bool(r.get("correct", False))
        confidences.append(extract_confidence(trace))
        correctness.append(ok)

    accuracy = sum(1 for c in correctness if c) / n
    mean_stated_confidence = sum(confidences) / n

    overclaim = sum(
        1
        for conf, ok in zip(confidences, correctness)
        if (not ok) and conf > 0.7
    )
    underclaim = sum(
        1
        for conf, ok in zip(confidences, correctness)
        if ok and conf < 0.3
    )
    overclaim_rate = overclaim / n
    underclaim_rate = underclaim / n

    correct_confs = [c for c, ok in zip(confidences, correctness) if ok]
    wrong_confs = [c for c, ok in zip(confidences, correctness) if not ok]

    confidence_when_correct = (
        sum(correct_confs) / len(correct_confs) if correct_confs else 0.0
    )
    confidence_when_wrong = (
        sum(wrong_confs) / len(wrong_confs) if wrong_confs else 0.0
    )

    ece = _expected_calibration_error(confidences, correctness)

    return {
        "n_samples": int(n),
        "accuracy": float(accuracy),
        "mean_stated_confidence": float(mean_stated_confidence),
        "expected_calibration_error": float(ece),
        "overclaim_rate": float(overclaim_rate),
        "underclaim_rate": float(underclaim_rate),
        "confidence_when_correct": float(confidence_when_correct),
        "confidence_when_wrong": float(confidence_when_wrong),
    }
