"""Behavioral fingerprint metrics for model reasoning traces.

These metrics describe HOW the model reasons, not WHETHER it is correct.
They are useful for detecting when a training run has corrupted reasoning
style even when final accuracy looks similar.

All metrics are deterministic and computed from raw trace text using only
the `re` module. All keyword matching is case-insensitive.

Public API:
    behavioral_metrics(trace: str) -> dict
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Keyword lists (all matched case-insensitively as substrings)
# ---------------------------------------------------------------------------

_DISTINCT_RULE_MARKERS = [
    "let me try",
    "trying",
    "attempt",
    "another approach",
    "alternatively",
    "or maybe",
    "what if",
]

_SELF_CORRECTION_MARKERS = [
    "wait",
    "actually",
    "let me reconsider",
    "i was wrong",
    "correction:",
    "no, actually",
    "rethinking",
    "let me check again",
]

_CONFIDENCE_HIGH_MARKERS = [
    "certainly",
    "obviously",
    "clearly",
    "definitely",
    "must be",
    "is exactly",
    "i'm sure",
    "without doubt",
]

_CONFIDENCE_LOW_MARKERS = [
    "i think",
    "maybe",
    "perhaps",
    "might be",
    "i'm not sure",
    "possibly",
    "could be",
    "appears to",
]

_EXPLICIT_VERIFICATION_MARKERS = [
    "let me verify",
    "verifying",
    "let me check",
    "checking",
    "applying to example",
    "test on example",
    "let me confirm",
]

# Matches ALL-CAPS colon labels like `LEN:`, `TABLE:`, `VER:`, `ANS:`,
# `DECRYPT:`, `SCAN:`, etc. We require the label to be at the start of a
# line or preceded by whitespace so we don't accidentally match URLs.
_TEMPLATE_LABEL_RE = re.compile(r"(?:^|\s)([A-Z][A-Z0-9_]{1,15}):")

# Numbers (integers or decimals) used for numeric density
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

# \boxed{...} content extractor. Handles nested-free simple case. If the
# content itself contains braces we greedily take everything up to the last
# closing brace before a newline/end.
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_occurrences(text_lower: str, markers: list[str]) -> int:
    """Count total (overlapping-free) occurrences of any marker in text."""
    total = 0
    for m in markers:
        if not m:
            continue
        total += text_lower.count(m)
    return total


def _extract_thinking_block(trace: str) -> str:
    """Extract the thinking block per spec.

    - If `<think>` and `</think>` both present, return text between them.
    - If only `<think>` present, return text from `<think>` up to `\\boxed`.
    - If only `</think>` present (common when the chat template has
      already injected the opening tag and the model's completion begins
      inside the thinking block), treat the trace as starting inside the
      block and return everything up to `</think>`.
    - If neither, return empty string.
    """
    if not trace:
        return ""

    open_idx = trace.find("<think>")
    close_idx = trace.find("</think>")

    if open_idx != -1:
        start = open_idx + len("<think>")
        if close_idx != -1 and close_idx >= start:
            return trace[start:close_idx]
        boxed_idx = trace.find("\\boxed", start)
        if boxed_idx != -1:
            return trace[start:boxed_idx]
        return ""

    # No opening tag. If a closing tag exists, treat the start of the
    # trace as the start of the thinking block.
    if close_idx != -1:
        return trace[:close_idx]

    return ""


def _extract_boxed_content(trace: str) -> str:
    """Return the last \\boxed{...} content, or empty string."""
    if not trace:
        return ""
    matches = _BOXED_RE.findall(trace)
    if matches:
        return matches[-1].strip()
    return ""


def _substring_overlap_ratio(needle: str, haystack: str) -> float:
    """Fraction of `needle` that appears as a contiguous substring of
    `haystack`. Returns 0-1.

    Heuristic: find the longest contiguous substring of `needle` that
    appears in `haystack`, divided by len(needle). Empty needle -> 0.0.
    """
    if not needle or not haystack:
        return 0.0

    n = len(needle)
    # Full match short-circuit
    if needle in haystack:
        return 1.0

    # Longest common substring via simple expansion around candidate starts.
    # For answer strings this is short (usually <200 chars) so O(n*m) is fine.
    best = 0
    m = len(haystack)
    for i in range(n):
        # Try to extend a match starting at position i in needle
        # against every position in haystack where needle[i] appears.
        ch = needle[i]
        j = haystack.find(ch)
        while j != -1:
            k = 0
            while (
                i + k < n
                and j + k < m
                and needle[i + k] == haystack[j + k]
            ):
                k += 1
            if k > best:
                best = k
                if best == n:
                    return 1.0
            j = haystack.find(ch, j + 1)
    return best / n


def _per_step_avg_chunk_length(block: str) -> int:
    """Average length (in chars) of non-empty newline-delimited chunks."""
    if not block:
        return 0
    chunks = [c for c in block.split("\n") if c.strip()]
    if not chunks:
        return 0
    return int(sum(len(c) for c in chunks) / len(chunks))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def behavioral_metrics(trace: str) -> dict:
    """Compute behavioral fingerprint metrics for a single trace.

    Args:
        trace: raw model output, possibly containing ``<think>...</think>``
            and ``\\boxed{...}`` markers.

    Returns:
        Flat dict of ints/floats/strs describing the reasoning style. See
        module docstring for key definitions.
    """
    trace = trace or ""
    trace_lower = trace.lower()

    # ---- thinking block range ----
    think_block = _extract_thinking_block(trace)
    reasoning_token_count = len(think_block)
    reasoning_word_count = len(think_block.split()) if think_block else 0
    reasoning_line_count = (
        think_block.count("\n") + 1 if think_block else 0
    )

    # ---- exploration / correction / confidence markers ----
    distinct_rule_attempts = _count_occurrences(
        trace_lower, _DISTINCT_RULE_MARKERS
    )
    self_correction_count = _count_occurrences(
        trace_lower, _SELF_CORRECTION_MARKERS
    )
    confidence_high_count = _count_occurrences(
        trace_lower, _CONFIDENCE_HIGH_MARKERS
    )
    confidence_low_count = _count_occurrences(
        trace_lower, _CONFIDENCE_LOW_MARKERS
    )
    confidence_ratio = confidence_high_count / (
        confidence_high_count + confidence_low_count + 1
    )

    explicit_verification_count = _count_occurrences(
        trace_lower, _EXPLICIT_VERIFICATION_MARKERS
    )

    # ---- copy-from-input ratio ----
    boxed = _extract_boxed_content(trace)
    # "prompt" here means the trace text MINUS the boxed answer itself.
    # We look at how much of the answer is already present verbatim in
    # the rest of the trace (which usually mirrors the prompt / inputs
    # that the model echoed). This is a deterministic proxy for the
    # real prompt which isn't passed in.
    trace_without_boxed = _BOXED_RE.sub("", trace)
    copy_from_input_ratio = _substring_overlap_ratio(boxed, trace_without_boxed)

    # ---- structural ----
    per_step_avg_chunk_length = _per_step_avg_chunk_length(think_block)

    char_count = len(trace)
    number_matches = _NUMBER_RE.findall(trace)
    numeric_density = (
        len(number_matches) / (char_count / 100.0) if char_count > 0 else 0.0
    )

    template_label_count = len(_TEMPLATE_LABEL_RE.findall(trace))

    return {
        "reasoning_token_count": int(reasoning_token_count),
        "reasoning_word_count": int(reasoning_word_count),
        "reasoning_line_count": int(reasoning_line_count),
        "distinct_rule_attempts": int(distinct_rule_attempts),
        "self_correction_count": int(self_correction_count),
        "confidence_high_count": int(confidence_high_count),
        "confidence_low_count": int(confidence_low_count),
        "confidence_ratio": float(confidence_ratio),
        "explicit_verification_count": int(explicit_verification_count),
        "copy_from_input_ratio": float(copy_from_input_ratio),
        "per_step_avg_chunk_length": int(per_step_avg_chunk_length),
        "numeric_density": float(numeric_density),
        "template_label_count": int(template_label_count),
    }
