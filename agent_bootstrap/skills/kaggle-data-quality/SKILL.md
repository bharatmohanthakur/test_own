---
name: kaggle-data-quality
description: Training data quality checks for the Nemotron competition. Use before training any new adapter. Enforces boxed-after-think format, ground-truth answer match, quality-over-quantity, and the 6-tier data quality metrics.
---

# kaggle-data-quality — Data quality gate

## THE RULE: quality > quantity. 504 verified > 9,500 raw.

User has stated this twice. More examples with mediocre traces will NOT help. Focus effort on:
- Deeper reasoning chains
- Self-verification steps
- Hypothesis testing
- Distillation from stronger models (Grok 4.20, Claude, DeepSeek R1, MinMax)

Target sample count: **≤ 2,000** unless proven otherwise.

## MANDATORY format check (SFT data)

Every example MUST have `\boxed{ANSWER}` AFTER `</think>`. If < 95% do, training will fail catastrophically (v29 had 3.2% correct format → scored 0.22).

```python
import json, re
path = "data/your_training.jsonl"
lines = [json.loads(l) for l in open(path)]
good, no_box, box_in_think = 0, 0, 0
for x in lines:
    text = x.get("text") or x.get("messages", [{}])[-1].get("content", "")
    after_think = text.split("</think>")[-1] if "</think>" in text else ""
    if "\\boxed{" in after_think:
        good += 1
    elif "\\boxed{" not in text:
        no_box += 1
    else:
        box_in_think += 1
n = len(lines)
print(f"Total: {n}")
print(f"✓ boxed AFTER </think>: {good} ({good/n:.1%})  [TARGET ≥ 95%]")
print(f"✗ boxed inside <think> only: {box_in_think} ({box_in_think/n:.1%})")
print(f"✗ no \\boxed{{}} at all: {no_box} ({no_box/n:.1%})")
assert good/n >= 0.95, f"FORMAT FAIL: only {good/n:.1%} correct. Fix before training."
```

## MANDATORY ground-truth match

Every `\boxed{}` answer must match the `train.csv` ground truth for that example's prompt.

**Past disaster**: `perfect_cot_1000.jsonl` had wrong answers (6.41 instead of 57.0) because prompt matching was by prefix. v8 scored 0.60 (down from 0.69).

```python
import pandas as pd, json, re
gt = pd.read_csv("train.csv")
# Build gt lookup by prompt hash or exact text (NOT prefix)
gt_map = {row.prompt.strip(): row.answer for _, row in gt.iterrows()}

lines = [json.loads(l) for l in open("data/your_training.jsonl")]
miss, wrong, ok = 0, 0, 0
for x in lines:
    prompt = x.get("prompt") or extract_prompt(x["text"])  # extract user msg
    if prompt.strip() not in gt_map:
        miss += 1
        continue
    expected = str(gt_map[prompt.strip()]).strip()
    m = re.search(r"\\boxed\{([^}]+)\}", x["text"].split("</think>")[-1])
    actual = m.group(1).strip() if m else None
    if actual == expected: ok += 1
    else: wrong += 1; print(f"MISMATCH: expected={expected} got={actual}")

print(f"OK: {ok}  WRONG: {wrong}  MISSING: {miss}")
assert wrong == 0, "Wrong labels in training data — fix before training"
```

## The 6 puzzle types — coverage check

Hidden test is balanced across: `binary`, `cipher`, `cryptarithm`, `gravity`, `roman`, `unit_conversion`. Your training data should have ≥ 50 examples per type. If any type is < 50, the model will regress on it.

```python
from collections import Counter
types = Counter(x.get("type","unknown") for x in lines)
for t,c in types.most_common(): print(f"{t}: {c}")
# Flag if any of the 6 types has < 50
```

## 6-tier quality metrics (from `data_quality_metrics.md`)

1. **Answer accuracy**: `\boxed{}` matches ground truth (above)
2. **Format compliance**: `\boxed{}` after `</think>` (above)
3. **Step derivation**: the trace actually derives the answer (not just asserts it)
4. **VER honesty**: `VER=PASS` lines aren't lying about intermediate checks
5. **No template leakage**: generic phrases like "Let me verify:" repeated identically across examples ⇒ template overfit risk
6. **Regression tracking**: when you regenerate data, diff old vs new. If 30% of traces change → diagnose before training.

## Teacher model quality (for distilled CoT)

| Teacher | Quality | Speed | Cost |
|---|---|---|---|
| Grok 4.20 | Best (per `data_quality_findings.md`) | medium | high |
| DeepSeek R1 | Good for math/code | fast | low |
| Claude Opus | Excellent format fidelity | slow | high |
| MinMax 2.7 | Okay, cheap fallback | fast | low |

Never use: teachers that match the target model (base Nemotron-3-Nano-30B already trained on Cascade-2; don't distill from that family).

## Diagnose-first rule

Before generating MORE data, run the current model on the training set and **analyze errors by category**:
- Wrong operation identified? → targeted operation-classification examples
- Wrong calculation? → step-by-step arithmetic examples
- Format issue (no `\boxed{}`)? → format-focused examples with minimal reasoning
- Confident-wrong VER=PASS? → honest self-verification examples

Don't throw generic data at a known failure mode.

## Memory pointers
- `data_quality_findings.md` — less curated > more raw; Grok best teacher
- `data_quality_metrics.md` — 6-tier checklist
- `feedback_data_verification.md` — never match prompts by prefix
- `feedback_data_quality.md` — quality > quantity
- `feedback_diagnose_first.md` — analyze errors, then target data
- `v29_disaster.md` — 3.2% format correctness → 0.22 score
- `donald_playbook.md` — per-type trace templates
