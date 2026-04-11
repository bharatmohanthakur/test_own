"""Reasoning chain graders — per-puzzle-type step-by-step grading.

Each module exposes:
    grade_chain(prompt, expected, trace, boxed) -> dict

Returns:
    {
        "type": str,
        "steps": [
            {"name": str, "passed": bool, "score": float (0-1), "detail": str},
            ...
        ],
        "first_failure": str | None,   # name of first failed step
        "pass_count": int,
        "fail_count": int,
        "chain_score": float,           # fraction of steps passed
    }

Step names per type are STABLE across adapters (so we can compare iterations).
Each step's `passed` is binary; `score` is graded (e.g., table_accuracy 0.79).
"""
from . import cipher, binary, equation, gravity, unit, roman

GRADERS = {
    "cipher": cipher.grade_chain,
    "binary": binary.grade_chain,
    "equation": equation.grade_chain,
    "gravity": gravity.grade_chain,
    "unit": unit.grade_chain,
    "roman": roman.grade_chain,
}


def grade(puzzle_type: str, prompt: str, expected: str, trace: str, boxed: str) -> dict:
    fn = GRADERS.get(puzzle_type)
    if fn is None:
        return {
            "type": puzzle_type,
            "steps": [],
            "first_failure": "unknown_type",
            "pass_count": 0,
            "fail_count": 0,
            "chain_score": 0.0,
        }
    return fn(prompt, expected, trace, boxed)
