# Metric Update — April 2026 (Critical)

**Source**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687798
**Author**: Ryan Holbrook (KAGGLE STAFF — competition organizer)
**Posted**: ~Apr 2026

## What changed

Around **March 28, 2026** the evaluation metric was updated to fix a bug where binary answers were matching as **floats** instead of strict **strings**. This caused a **0.3-0.4 point drop on new submissions**.

The new `verify()` function:

```python
import re, math

def verify(stored_answer: str, predicted: str) -> bool:
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # If the answer is a binary string, compare strictly as strings
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()
```

## Why this matters for us

Our submission timeline:
- Mar 21: SFT v7 = **0.69** (best, scored under OLD metric, float coercion friendly)
- Apr 2: v15 (raw 9496 ex) = 0.53
- Apr 2: v7 resub #1 = 0.66
- Apr 2: v7 resub #2 = 0.65
- Apr 6: 4x submissions = 0.53, 0.58, 0.58, 0.65

**The 0.69 → 0.65 drop on the SAME v7 adapter is partly the metric update.**

If the rescore re-runs old submissions under the new metric, our 0.69 will likely become 0.65-0.66.

## The binary format mismatch problem (Sangram Patil, rank 184)

Sangram reported that the **test set has binary labels that are NOT always 8-bit**. Examples cited:
- `1`
- `111101`
- `1000100`
- `10111111`

But train.csv binary answers are **uniformly 8-bit (1602/1602 verified Apr 7)**.

> "The given ground truth label is 1 only, but the expected output is 00000001. So I think the labels were created according to this format. Also, some of the ground truth labels in the bit manipulation tasks are not 8-bit. Even if we train the model to predict 8-bit outputs, we need the ground truth labels to be in the same format."

### Implication
With strict string match for binary `[01]+` answers:
- A model trained to output `00000001` (8-bit padded) **will fail** if the stored answer is `1`
- A model trained to output `1` (no padding) will fail if the stored answer is `00000001`

### Possible mitigations (need to test which the test set actually expects)
1. **Output 8-bit padded** — matches train format, may fail on stripped-label tests
2. **Output stripped (int-like)** — `int(binary_str, 2)` then back to bin, no leading zeros — may fail on padded-label tests
3. **Hedge in training**: 50/50 mix of padded and stripped labels in SFT
4. **Ask organizer in discussion** to clarify the test set format

### Tong Hui Kang (rank 7) related observation
`verify("0234", "234") == True` because numeric answers go through float comparison. So **numerical answers can have leading zeros stripped** without penalty. But **binary answers cannot** (strict string match).

## Action items

1. **Update verify() in our training data filter** to use the new strict binary string match. Drop training examples whose model output predictions wouldn't pass the new verifier.
2. **Check old SFT v22 binary outputs** — are they 8-bit padded? Yes (matches train.csv).
3. **Add a binary-format hedging task**: generate training examples with both padded and stripped labels for the same input, so the model learns to match the label format.
4. **Use the new verify() locally** when computing GRPO correctness rewards.

## Next steps (in priority order)

- Use the new `verify()` in v28+ correctness reward function
- Check the few non-8-bit binary outputs Sangram cited — search train.csv for any short binary labels (none found, all are 8-bit)
- Generate a small validation split with both padded and stripped binary to A/B test the format
- Re-evaluate v22 base score after the rescore lands (will help us see the new baseline)
