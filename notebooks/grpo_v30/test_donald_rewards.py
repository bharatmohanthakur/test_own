"""
Test the Donald reward functions on diverse synthetic inputs.
Run: python3 notebooks/grpo_v30/test_donald_rewards.py
"""
import sys
sys.path.insert(0, '/Users/bharat/Downloads/kaggle/notebooks/grpo_v30')
import donald_rewards as dr


def test(name, expected, actual):
    if isinstance(expected, list) and isinstance(actual, list):
        ok = all(abs(e - a) < 0.001 for e, a in zip(expected, actual))
    else:
        ok = expected == actual
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: expected={expected} actual={actual}")
    return ok


def make_prompt(text):
    return [{"role": "user", "content": text}]


def make_completion(text):
    return [{"role": "assistant", "content": text}]


# ============================================================
def test_correctness():
    print("\n--- correctness_reward ---")
    # Gravity-style: float comparison with 1e-2 rel tol
    p = [make_prompt("In Alice's Wonderland, the gravitational constant has been secretly changed.")]
    c = [make_completion("<think>SOLVE...</think>\\boxed{132.42}")]
    test("correct float (132.42 vs stored 132.44)", [2.0], dr.correctness_reward(p, c, ["132.44"]))

    c = [make_completion("\\boxed{999.99}")]
    test("wrong float", [0.0], dr.correctness_reward(p, c, ["132.44"]))

    # Binary: strict string match
    p = [make_prompt("In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.")]
    c = [make_completion("\\boxed{00000001}")]
    test("correct binary", [2.0], dr.correctness_reward(p, c, ["00000001"]))

    c = [make_completion("\\boxed{00000010}")]
    test("wrong binary (one bit off)", [0.0], dr.correctness_reward(p, c, ["00000001"]))

    # Cipher: string comparison
    p = [make_prompt("In Alice's Wonderland, secret encryption rules are used on text.")]
    c = [make_completion("\\boxed{cat imagines book}")]
    test("correct cipher", [2.0], dr.correctness_reward(p, c, ["cat imagines book"]))


def test_format():
    print("\n--- format_reward ---")
    test("box only", [0.5], dr.format_reward([make_completion("\\boxed{42}")]))
    test("think only", [0.3], dr.format_reward([make_completion("<think>x</think>")]))
    test("both", [1.0], dr.format_reward([make_completion("<think>x</think>\\boxed{42}")]))
    test("none", [0.0], dr.format_reward([make_completion("just text")]))


def test_pipeline_step():
    print("\n--- pipeline_step_reward ---")
    p = [make_prompt("In Alice's Wonderland, the gravitational constant has been secretly changed.")]
    # Gravity expects: SOLVE, RATE, VER, APPLY, ANS
    c = [make_completion("<think>SOLVE: x\nRATE = 9.8\nVER: PASS\nAPPLY: y\nANS: 42</think>")]
    test("all gravity steps", [0.5], dr.pipeline_step_reward(p, c, ["42"]))

    c = [make_completion("<think>VER: PASS</think>")]
    test("only VER", [0.1], dr.pipeline_step_reward(p, c, ["42"]))


def test_contamination():
    print("\n--- contamination_penalty ---")
    # Gravity prompt with binary contamination
    p = [make_prompt("In Alice's Wonderland, the gravitational constant has been secretly changed.")]
    c = [make_completion("<think>XOR these bits: 010 XOR 011</think>\\boxed{42}")]
    # XOR is in gravity contamination markers
    result = dr.contamination_penalty(p, c)
    print(f"  result: {result} (expect negative for XOR contamination)")
    assert result[0] < 0, "XOR should be flagged as contamination in gravity"
    print("  PASS")


def test_thrash():
    print("\n--- thrash_penalty ---")
    c = [make_completion("Hmm, let me try this. Maybe... actually, wait.")]
    result = dr.thrash_penalty(c)
    print(f"  result: {result} (expect strongly negative)")
    assert result[0] < -0.5, f"Multiple thrash markers should be heavily penalized, got {result[0]}"
    print("  PASS")


def test_ver_honesty():
    print("\n--- ver_honesty_reward ---")
    p = [make_prompt("gravitational constant")]

    # Honest YES: correct + VER PASS
    c = [make_completion("<think>VER: PASS</think>\\boxed{42}")]
    test("honest YES (correct+PASS)", [0.5], dr.ver_honesty_reward(p, c, ["42"]))

    # Honest NO: wrong + VER FAIL
    c = [make_completion("<think>VER: FAIL</think>\\boxed{99}")]
    test("honest NO (wrong+FAIL)", [0.2], dr.ver_honesty_reward(p, c, ["42"]))

    # Confused NO: correct + VER FAIL
    c = [make_completion("<think>VER: FAIL</think>\\boxed{42}")]
    test("confused NO (correct+FAIL)", [-0.1], dr.ver_honesty_reward(p, c, ["42"]))

    # Lying YES: wrong + VER PASS
    c = [make_completion("<think>VER: PASS</think>\\boxed{99}")]
    test("lying YES (wrong+PASS)", [-0.5], dr.ver_honesty_reward(p, c, ["42"]))

    # No VER
    c = [make_completion("\\boxed{42}")]
    test("no VER", [0.0], dr.ver_honesty_reward(p, c, ["42"]))


def test_champagne():
    print("\n--- champagne_bonus ---")
    p = [make_prompt("In Alice's Wonderland, the gravitational constant has been secretly changed.")]
    # Perfect: correct + all steps + no thrash + no contam
    c = [make_completion("<think>SOLVE: x\nRATE = 9.8\nVER: PASS\nAPPLY: y\nANS: 42</think>\\boxed{42}")]
    test("perfect gravity trace", [5.0], dr.champagne_bonus(p, c, ["42"]))

    # Wrong answer: no champagne
    c = [make_completion("<think>SOLVE: x\nRATE = 9.8\nVER: PASS\nAPPLY: y\nANS: 99</think>\\boxed{99}")]
    test("wrong answer", [0.0], dr.champagne_bonus(p, c, ["42"]))

    # Correct but missing step
    c = [make_completion("<think>SOLVE: x\nRATE = 9.8\nANS: 42</think>\\boxed{42}")]
    test("missing VER step", [0.0], dr.champagne_bonus(p, c, ["42"]))


def main():
    print("=" * 60)
    print("Testing Donald reward functions")
    print("=" * 60)
    test_correctness()
    test_format()
    test_pipeline_step()
    test_contamination()
    test_thrash()
    test_ver_honesty()
    test_champagne()
    print("\n" + "=" * 60)
    print("All tests done")


if __name__ == "__main__":
    main()
