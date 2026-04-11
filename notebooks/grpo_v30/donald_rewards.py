"""
Donald Galliano playbook reward functions for GRPO.

Per Donald (notes/donald-galliano-playbook.md):
- Per-step partial credit at every labeled stage
- Tiered VER honesty: lying YES (worst) > confused NO > honest NO > honest YES (best)
- Champagne bonus on fully-correct (+5 superlinear)
- Penalize contamination markers (other templates' language)
- Penalize thrash markers ("hmm", "let me try", "actually")

Each reward function takes (prompts, completions, answer, **kwargs) and returns a list of floats.
Compatible with TRL GRPOTrainer reward_funcs interface.
"""
import re
import math


# ============================================================
# UTILITIES
# ============================================================
def _get_completion_text(c):
    return c[0]["content"] if isinstance(c, list) else c

def _verify_answer(stored, predicted):
    """Match the new strict metric (Apr 2026): binary as string, others as float."""
    stored = str(stored).strip()
    predicted = str(predicted).strip()
    if not predicted:
        return False
    if re.fullmatch(r'[01]+', stored):
        return predicted.lower() == stored.lower()
    try:
        return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored.lower()

def _extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else ""


# ============================================================
# CONTAMINATION MARKERS (cross-template language)
# ============================================================
CONTAMINATION_MARKERS = {
    'binary':   ['LEN:', 'TABLE:', 'DECRYPT:', 'VOCAB', 'roman', 'gravity', 'RATE', 'cipher'],
    'cipher':   ['B0:', 'XOR', 'AND', 'NOT', 'roman', 'RATE', 'gravity', 'CONST'],
    'gravity':  ['B0:', 'XOR', 'TABLE:', 'roman', 'cipher', 'binary', 'DECRYPT'],
    'unit':     ['B0:', 'XOR', 'TABLE:', 'roman', 'cipher', 'binary', 'DECRYPT'],
    'roman':    ['B0:', 'XOR', 'AND', 'TABLE:', 'cipher', 'gravity', 'RATE', 'DECRYPT'],
    'equation': ['LEN:', 'TABLE:', 'DECRYPT:', 'roman', 'gravity', 'RATE'],
}

# Thrash markers — universal
THRASH_MARKERS = [
    'let me try', 'hmm', 'actually,', 'wait,', "i'm not sure",
    'maybe', 'perhaps', 'i think i', 'or perhaps',
]

def _detect_type(prompt_text):
    """Identify puzzle type from prompt first line."""
    first_line = prompt_text.split('\n')[0].lower() if prompt_text else ''
    if 'bit manipulation' in first_line or '8-bit' in first_line: return 'binary'
    if 'roman' in first_line or 'numeral' in first_line: return 'roman'
    if 'unit' in first_line or 'measurement' in first_line: return 'unit'
    if 'gravitational' in first_line: return 'gravity'
    if 'encryption' in first_line or 'cipher' in first_line: return 'cipher'
    if 'transformation rules' in first_line: return 'equation'
    return 'other'


# ============================================================
# REWARD 1: correctness (the main signal)
# ============================================================
def correctness_reward(prompts, completions, answer, **kwargs):
    """+2 if final \\boxed{} answer matches stored answer (new strict metric), else 0."""
    rewards = []
    for c, a in zip(completions, answer):
        text = _get_completion_text(c)
        boxed = _extract_boxed(text)
        rewards.append(2.0 if _verify_answer(a, boxed) else 0.0)
    return rewards


# ============================================================
# REWARD 2: format compliance
# ============================================================
def format_reward(completions, **kwargs):
    """+0.5 if \\boxed{} present, +0.3 if <think> present, +0.2 if both."""
    rewards = []
    for c in completions:
        text = _get_completion_text(c)
        score = 0.0
        if '\\boxed{' in text:
            score += 0.5
        if '<think>' in text and '</think>' in text:
            score += 0.3
        if '\\boxed{' in text and '<think>' in text:
            score += 0.2
        rewards.append(score)
    return rewards


# ============================================================
# REWARD 3: per-step labeled pipeline (binary, cipher, gravity, unit, roman)
# ============================================================
PIPELINE_LABELS = {
    'binary':   ['SCAN', 'VER', 'CONCAT', 'ANS'],
    'cipher':   ['LEN:', 'TABLE', 'VER', 'DECRYPT', 'CHECK', 'ANS'],
    'gravity':  ['SOLVE', 'RATE', 'VER', 'APPLY', 'ANS'],
    'unit':     ['SOLVE', 'RATE', 'VER', 'APPLY', 'ANS'],
    'roman':    ['DECOMPOSE', 'CAT', 'VER', 'ANS'],
    'equation': ['PARSE', 'SCAN', 'LOCK', 'VER', 'APPLY', 'ANS'],
}

def pipeline_step_reward(prompts, completions, answer, **kwargs):
    """+0.1 per labeled step present in the trace, scaled by puzzle type."""
    rewards = []
    for p, c in zip(prompts, completions):
        # Extract prompt text
        if isinstance(p, list):
            prompt_text = p[0].get('content', '') if p else ''
        else:
            prompt_text = p
        ptype = _detect_type(prompt_text)
        labels = PIPELINE_LABELS.get(ptype, [])
        text = _get_completion_text(c)
        present = sum(1 for label in labels if label in text)
        rewards.append(0.1 * present)
    return rewards


# ============================================================
# REWARD 4: contamination penalty
# ============================================================
def contamination_penalty(prompts, completions, **kwargs):
    """-0.2 per contamination marker (language from other templates)."""
    rewards = []
    for p, c in zip(prompts, completions):
        if isinstance(p, list):
            prompt_text = p[0].get('content', '') if p else ''
        else:
            prompt_text = p
        ptype = _detect_type(prompt_text)
        markers = CONTAMINATION_MARKERS.get(ptype, [])
        text = _get_completion_text(c)
        text_lower = text.lower()
        hits = sum(1 for m in markers if m.lower() in text_lower)
        rewards.append(-0.2 * hits)
    return rewards


# ============================================================
# REWARD 5: thrash penalty
# ============================================================
def thrash_penalty(completions, **kwargs):
    """-0.15 per thrash marker (model spiraling instead of committing)."""
    rewards = []
    for c in completions:
        text = _get_completion_text(c).lower()
        hits = sum(1 for m in THRASH_MARKERS if m in text)
        rewards.append(-0.15 * hits)
    return rewards


# ============================================================
# REWARD 6: VER honesty (tiered)
# ============================================================
def ver_honesty_reward(prompts, completions, answer, **kwargs):
    """
    Detect VER step outcome and answer correctness, score the combination.
    Honest YES = correct + VER says PASS = +0.5
    Honest NO = wrong + VER says FAIL = +0.2 (caught own error)
    Confused NO = correct + VER says FAIL = -0.1 (rejected the right answer)
    Lying YES = wrong + VER says PASS = -0.5 (worst — rubber-stamped wrong answer)
    No VER = 0
    """
    rewards = []
    for c, a in zip(completions, answer):
        text = _get_completion_text(c)
        boxed = _extract_boxed(text)
        is_correct = _verify_answer(a, boxed)

        # Detect VER outcome
        ver_pass = bool(re.search(r'VER.*?(PASS|YES)', text, re.IGNORECASE))
        ver_fail = bool(re.search(r'VER.*?(FAIL|NO)', text, re.IGNORECASE))

        if not (ver_pass or ver_fail):
            rewards.append(0.0)
        elif is_correct and ver_pass:
            rewards.append(0.5)   # honest YES
        elif (not is_correct) and ver_fail:
            rewards.append(0.2)   # honest NO
        elif is_correct and ver_fail:
            rewards.append(-0.1)  # confused NO
        else:
            rewards.append(-0.5)  # lying YES
    return rewards


# ============================================================
# REWARD 7: champagne bonus (fully correct + clean trace)
# ============================================================
def champagne_bonus(prompts, completions, answer, **kwargs):
    """
    Superlinear bonus +5 for: correct answer + all pipeline steps present + no thrash + no contamination.
    """
    rewards = []
    for p, c, a in zip(prompts, completions, answer):
        if isinstance(p, list):
            prompt_text = p[0].get('content', '') if p else ''
        else:
            prompt_text = p
        text = _get_completion_text(c)
        text_lower = text.lower()

        boxed = _extract_boxed(text)
        is_correct = _verify_answer(a, boxed)

        ptype = _detect_type(prompt_text)
        labels = PIPELINE_LABELS.get(ptype, [])
        all_steps_present = all(label in text for label in labels)

        no_thrash = not any(m in text_lower for m in THRASH_MARKERS)
        markers = CONTAMINATION_MARKERS.get(ptype, [])
        no_contam = not any(m.lower() in text_lower for m in markers)

        rewards.append(5.0 if (is_correct and all_steps_present and no_thrash and no_contam) else 0.0)
    return rewards


# ============================================================
# Convenience: bundle of all reward functions
# ============================================================
ALL_REWARDS = [
    correctness_reward,
    format_reward,
    pipeline_step_reward,
    contamination_penalty,
    thrash_penalty,
    ver_honesty_reward,
    champagne_bonus,
]
