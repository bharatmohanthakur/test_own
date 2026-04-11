"""Per-type verifiers. Each exposes `verify(prompt, expected, trace, boxed)` → dict."""
from . import cipher, binary, equation, gravity, unit, roman

VERIFIERS = {
    'cipher': cipher.verify,
    'binary': binary.verify,
    'equation': equation.verify,
    'gravity': gravity.verify,
    'unit': unit.verify,
    'roman': roman.verify,
}


def verify_row(puzzle_type: str, prompt: str, expected: str, trace: str, boxed: str) -> dict:
    """Dispatch to the correct verifier. Always returns a dict with at least
    {correct, failure_mode, quality_flags}."""
    fn = VERIFIERS.get(puzzle_type)
    if fn is None:
        return {
            "correct": False,
            "failure_mode": "unknown_type",
            "quality_flags": {},
            "ground_truth": {},
            "model_facts": {},
        }
    return fn(prompt, expected, trace, boxed)
